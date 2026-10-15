"""Tests for the WBC Studio core (no browser): clip IO / QA, dataset browsing, frames, jobs."""

from __future__ import annotations

import dataclasses
import json
import struct
from pathlib import Path

import mjlab  # noqa: F401 — load mjlab.tasks entry points before wbc_mjlab task modules
import numpy as np
import pytest

from wbc_mjlab.data_paths import repo_root
from wbc_mjlab.studio import clips as clipmod
from wbc_mjlab.studio import jobs as jobmod

SAMPLES = repo_root() / "data" / "g1" / "samples"


@pytest.fixture(scope="module")
def core():
  from wbc_mjlab.studio.core import StudioCore

  return StudioCore(robot="g1", device="cpu")


def test_raw_header_counts_csv_rows() -> None:
  csv = SAMPLES / "backflip.csv"
  head = clipmod.raw_header(csv)
  rows = len(csv.read_text().strip().splitlines())
  assert head == {"frames": rows, "fps": 30.0, "dof": 29}


def test_npz_header_reads_shape_without_loading(tmp_path: Path) -> None:
  path = tmp_path / "clip.npz"
  np.savez(
    path, joint_pos=np.zeros((17, 29), np.float32), fps=np.array([50.0], np.float32)
  )
  assert clipmod.npz_header(path) == {"frames": 17, "dof": 29, "fps": 50.0}


def test_load_raw_clip_and_analyze(core) -> None:
  clip = clipmod.load_clip(SAMPLES / "walk1_subject1.csv", core)
  assert clip.kind == "raw" and clip.fps == 30.0
  assert clip.body_pos.shape == (clip.frames, core.model.nbody, 3)
  feet, sole = core.data._robot_feet()  # noqa: SLF001
  qa = clipmod.analyze(clip, core.model, feet=feet, sole_z=sole)
  assert qa["frames"] == clip.frames
  assert any("resamples to 50 Hz" in i["text"] for i in qa["issues"])
  # issues with a curve name the Plots channels that show it, and where it starts
  for issue in qa["issues"]:
    if "plot" in issue:
      assert issue["plot"]["channels"]
      frame = issue["plot"].get("frame")
      assert frame is None or 0 <= frame < clip.frames


def test_wildcard_and_substring_search() -> None:
  from wbc_mjlab.studio.data import DataService

  names = ["walk1_subject1", "run1_subject2", "Loop_Forward_Walk_001", "dance"]
  assert DataService._match(names, "*walk*") == [
    "walk1_subject1",
    "Loop_Forward_Walk_001",
  ]
  assert DataService._match(names, "subject run") == ["run1_subject2"]
  assert DataService._match(names, "") == names


def test_listing_samples(core) -> None:
  listing = core.data.listing("samples", q="walk", status="all", page=0, size=10)
  assert listing["total"] >= 1
  row = next(r for r in listing["rows"] if r["name"] == "walk1_subject1")
  assert row["raw"] == "walk1_subject1.csv"


def test_preview_raw_clip_and_frame_encoding(core) -> None:
  from wbc_mjlab.studio.web.server import encode_frame, model_payload

  core.data.load("samples", "walk1_subject1", source="raw")
  state = core.data.state()
  assert state["active"] and state["kind"] == "raw"
  core.data.set_frame(10)
  assert core.scene.hud["frame"] == 10
  buf = encode_frame(core, with_overlays=True)
  head_len = struct.unpack("<I", buf[:4])[0]
  header = json.loads(buf[4 : 4 + head_len])
  assert header["n"] == 1 and header["nbody"] == core.model.nbody
  assert any(o["key"] == "data/path" for o in header["overlays"])
  meta, blob = model_payload(core.model)
  assert meta["nbody"] == core.model.nbody and len(blob) == meta["bytes"]
  core.data.close()


def test_finite_sanitizes_nan() -> None:
  from wbc_mjlab.studio.web.server import _finite

  out = _finite({"a": float("nan"), "b": [1.0, float("inf")], "c": np.float32(2.0)})
  assert json.dumps(out) == '{"a": null, "b": [1.0, null], "c": 2.0}'


def test_job_progress_parsing(tmp_path: Path) -> None:
  log = tmp_path / "train.log"
  log.write_text(
    "Learning iteration 41/100\nMean reward: 12.5\nMean episode length: 300.0\n"
    "Logging experiment in directory: /x/logs/rsl_rl/wbc_g1/run\n"
  )
  job = jobmod.Job(name="t", kind="train", pid=0, log=str(log), started=0.0, command=[])
  p = job.progress()
  assert p["iteration"] == 41 and p["total"] == 100
  assert p["fraction"] == pytest.approx(0.42)
  assert p["log_dir"].endswith("wbc_g1/run")

  conv = tmp_path / "convert.log"
  conv.write_text(
    "Starting FK pool: 4 env(s), 3 clip(s)\n[INFO] Exported clip: a\n[INFO] Exported clip: b\n"
  )
  cj = jobmod.Job(
    name="c",
    kind="convert",
    pid=0,
    log=str(conv),
    started=0.0,
    command=[],
    meta={"mirror": True},
  )
  assert cj.progress() == {"done": 2, "total": 6, "fraction": pytest.approx(1 / 3)}


def test_checkpoint_iteration() -> None:
  assert jobmod.checkpoint_iteration("logs/x/model_349999.pt") == 349999
  assert jobmod.checkpoint_iteration("model.pt") == -1


@pytest.mark.slow
def test_play_session_pins_and_loops() -> None:
  import time

  import torch

  if not torch.cuda.is_available():
    pytest.skip("needs CUDA")
  from wbc_mjlab.studio.core import StudioCore

  c = StudioCore(robot="g1", device="cuda:0")
  if jobmod.find_run(jobmod.DEMO_LABEL) is None or not any(SAMPLES.glob("npz/*.npz")):
    pytest.skip("needs the demo checkpoint and converted samples")
  c.play.configure(jobmod.DEMO_LABEL, "model.pt", motion=str(SAMPLES), num_envs=2)
  while c.busy:
    time.sleep(0.2)
  for _ in range(
    5
  ):  # steps before pinning: pin() then writes buffers the steps created
    c.play.step_once()
  c.play.set_clip("walk1_subject1")
  for _ in range(30):
    c.play.step_once()
  hud = c.scene.hud
  assert hud["title"] == "walk1_subject1" and hud["pinned"] and hud["frame"] == 30
  assert hud["metrics"]["error_body_pos"] < 0.2
  assert c.scene.snapshot()["ghost_pos"].shape[0] == 2
  c.play.push(speed=1.0)
  c.play.step_once()
  c.play.stop()


def test_token_auth(core) -> None:
  from fastapi.testclient import TestClient

  from wbc_mjlab.studio.web.server import create_app

  app = create_app(core, token="secret")
  remote = TestClient(app, client=("192.168.1.20", 5000))
  assert remote.get("/api/state").status_code == 401
  assert remote.get("/?token=wrong").status_code == 401
  r = remote.get("/?token=secret", follow_redirects=False)
  assert r.status_code == 303 and r.cookies.get("wbc_studio_token") == "secret"
  assert remote.get("/api/state").status_code == 200  # cookie kept by the client
  local = TestClient(app, client=("127.0.0.1", 5000))
  assert local.get("/api/state").status_code == 200


# ---------------------------------------------------------------------- presets
@pytest.fixture
def preset_dir(tmp_path, monkeypatch):
  monkeypatch.setenv("WBC_MJLAB_CONFIG_DIR", str(tmp_path))
  return tmp_path / "presets"


def test_config_dir_is_per_user(monkeypatch, tmp_path) -> None:
  from wbc_mjlab.studio import presets

  monkeypatch.delenv("WBC_MJLAB_CONFIG_DIR", raising=False)
  monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
  assert presets.presets_dir() == tmp_path / "xdg" / "wbc-mjlab" / "presets"
  monkeypatch.setenv("WBC_MJLAB_CONFIG_DIR", str(tmp_path / "mine"))
  assert presets.modules_dir() == tmp_path / "mine" / "modules"


def test_preset_roundtrip_applies_overrides(preset_dir) -> None:
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from wbc_mjlab.studio import presets
  from wbc_mjlab.tasks import register_all_wbc_tasks

  register_all_wbc_tasks()
  schema = presets.describe(load_env_cfg("Wbc-G1"), load_rl_cfg("Wbc-G1"))
  assert {r["name"] for r in schema["rewards"]} >= {"motion_body_pos", "survival"}
  keys = {f["key"] for f in schema["command"]}
  assert {"rsi.bin_width_s", "rsi.sampling_mode", "env.episode_length_s"} <= keys
  assert any(f["key"] == "algorithm.learning_rate" for f in schema["agent"])

  ov = {
    "rewards": {
      "motion_body_pos": {"weight": 2.5, "params": {"sigma_per_keybody": 0.15}},
      "anti_shake": {"enabled": False},
    },
    "terminations": {"anchor_pos": {"params": {"threshold": 0.5}}},
    "events": {"push_robot": {"params": {"velocity_range.x": [-1.0, 1.0]}}},
    "command": {"rsi.bin_width_s": 2.0, "rsi.sampling_mode": "uniform"},
    "agent": {"algorithm.learning_rate": 5e-4},
  }
  presets.save("zz_pytest", "Wbc-G1", ov, "test")
  assert presets.count_changes(presets.load("zz_pytest")["overrides"]) == 8
  tid = presets.register("zz_pytest")
  assert tid == "Wbc-G1--zz_pytest"
  env, rl = load_env_cfg(tid), load_rl_cfg(tid)
  assert env.rewards["motion_body_pos"].weight == 2.5
  assert env.rewards["motion_body_pos"].params["sigma_per_keybody"] == 0.15
  assert "anti_shake" not in env.rewards
  assert env.terminations["anchor_pos"].params["threshold"] == 0.5
  assert env.events["push_robot"].params["velocity_range"]["x"] == (-1.0, 1.0)
  assert env.commands["motion"].rsi.bin_width_s == 2.0
  assert rl.algorithm.learning_rate == 5e-4
  assert rl.experiment_name == "wbc_g1--zz_pytest"
  assert presets.task_for_experiment("wbc_g1--zz_pytest") == tid
  # the base task is untouched
  assert load_env_cfg("Wbc-G1").rewards["motion_body_pos"].weight == 1.0
  # editing the file re-registers
  presets.save("zz_pytest", "Wbc-G1", {"rewards": {"survival": {"weight": 3.0}}})
  presets.register("zz_pytest")
  assert load_env_cfg(tid).rewards["survival"].weight == 3.0
  presets.delete("zz_pytest")
  assert presets.list_presets() == []


def test_se_module_reproduces_the_se_task() -> None:
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.studio import presets
  from wbc_mjlab.tasks import register_all_wbc_tasks

  register_all_wbc_tasks()
  se = {m["name"]: m for m in presets.list_modules("g1")}["se"]
  assert se["builtin"] and se["source"] == "Wbc-G1-SE vs Wbc-G1"
  env, _ = presets.build("Wbc-G1", {}, modules=["se"])
  ref = load_env_cfg("Wbc-G1-SE")
  for g in ("actor", "critic"):
    assert list(env.observations[g].terms) == list(ref.observations[g].terms)
  lin = env.observations["actor"].terms["base_lin_vel"]
  assert lin.params == ref.observations["actor"].terms["base_lin_vel"].params


def test_observation_overrides_modules_and_snapshot(preset_dir) -> None:
  import yaml
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.studio import presets

  obs = {
    "actor": {
      "history_length": 3,
      "terms": {
        "actions": {"enabled": False},
        "joint_vel": {"noise_scale": 2.0},
        "ref_body_pos": {"add": True, "bodies": ["pelvis", "torso_link"]},
      },
    },
    "critic": {"terms": {"assistive_gain": {"enabled": False}}},
  }
  presets.save_module(
    "zz_noisy",
    {"observations": {"actor": {"terms": {"joint_pos": {"noise_scale": 0.0}}}}},
  )
  assert [m["name"] for m in presets.list_modules("g1")] == ["se", "zz_noisy"]
  with pytest.raises(ValueError):
    presets.save_module("se", {})
  presets.save("zz_obs", "Wbc-G1", {"observations": obs}, modules=["se", "zz_noisy"])
  assert presets.count_changes({"observations": obs}) == 6
  tid = presets.register("zz_obs")
  actor = load_env_cfg(tid).observations["actor"]
  assert actor.history_length == 3
  assert (
    "actions" not in actor.terms and "ref_base_height" not in actor.terms
  )  # own + se
  assert "base_lin_vel" in actor.terms  # se
  assert actor.terms["joint_vel"].noise.n_max == pytest.approx(3.0)
  assert actor.terms["joint_pos"].noise is None  # zz_noisy
  assert actor.terms["ref_body_pos"].params["body_names"] == ("pelvis", "torso_link")
  assert list(actor.terms)[-1] == "ref_body_pos"
  assert "assistive_gain" not in load_env_cfg(tid).observations["critic"].terms
  schema = presets.describe(*presets.build("Wbc-G1", {}, modules=["se"]))
  names = [t["name"] for t in schema["observations"]["groups"][0]["terms"]]
  assert "motion_anchor_ori_error" in names and "projected_gravity" not in names
  assert {"name", "dim", "deployable"} <= set(presets.library("g1", 29, 14)[0])
  assert not presets.deployable("assistive_gain") and presets.deployable("ref_body_pos")

  # a run keeps its resolved preset: it rebuilds after the module and preset are gone
  snap = presets.resolve("zz_obs")
  run = preset_dir.parent / "run"
  (run / "params").mkdir(parents=True)
  (run / "params" / presets.SNAPSHOT).write_text(yaml.safe_dump(snap))
  presets.delete_module("zz_noisy")
  presets.delete("zz_obs")
  with pytest.raises(OSError):  # the preset itself is gone
    presets.resolve("zz_obs")
  presets.register("zz_obs", snapshot=presets.snapshot_of(run))
  assert load_env_cfg(tid).observations["actor"].terms["joint_pos"].noise is None


def test_preset_names_are_validated(preset_dir) -> None:
  from wbc_mjlab.studio import presets

  with pytest.raises(ValueError):
    presets.save("../evil", "Wbc-G1", {})


@pytest.mark.slow
def test_play_live_reward_overrides() -> None:
  import time

  import torch

  if not torch.cuda.is_available():
    pytest.skip("needs CUDA")
  from wbc_mjlab.studio.core import StudioCore

  c = StudioCore(robot="g1", device="cuda:0")
  if jobmod.find_run(jobmod.DEMO_LABEL) is None or not any(SAMPLES.glob("npz/*.npz")):
    pytest.skip("needs the demo checkpoint and converted samples")
  c.play.configure(jobmod.DEMO_LABEL, "model.pt", motion=str(SAMPLES), num_envs=1)
  while c.busy:
    time.sleep(0.2)
  rm = c.play.session.env.reward_manager
  assert (
    c.play.apply_rewards(
      {"motion_body_pos": {"weight": 10.0}, "anti_shake": {"enabled": False}}
    )
    >= 1
  )
  assert rm.get_term_cfg("motion_body_pos").weight == 10.0
  assert rm.get_term_cfg("anti_shake").weight == 0.0
  for _ in range(3):
    c.play.step_once()
  c.play.apply_rewards({})  # back to the session's own weights
  assert rm.get_term_cfg("motion_body_pos").weight == 1.0
  c.play.stop()


# ---------------------------------------------------------------------- custom robots
def _robot_folder(cfg_dir: Path, robot_id: str) -> Path:
  import shutil

  from wbc_mjlab.robots.g1.constants import G1_XML

  dest = cfg_dir / "robots" / robot_id
  dest.mkdir(parents=True)
  shutil.copy(G1_XML.parent / "g1_23dof.xml", dest / "model.xml")
  shutil.copytree(G1_XML.parent / "assets", dest / "assets")
  return dest


def test_robot_suggestions_match_g1() -> None:
  from wbc_mjlab.robots import custom
  from wbc_mjlab.robots.g1.constants import (
    G1_ANCHOR_BODY_NAME,
    G1_MOTION_BODY_NAMES,
    G1_XML,
    MOTION_Z_DEBIAS_FOOT_SOLE_Z,
  )

  info = custom.inspect_model(G1_XML)
  s = info["suggested"]
  assert sorted(s["keybodies"]) == sorted(G1_MOTION_BODY_NAMES)
  assert s["anchor_body"] == G1_ANCHOR_BODY_NAME
  assert s["feet_bodies"] == ["left_ankle_roll_link", "right_ankle_roll_link"]
  assert s["foot_sole_z"] == pytest.approx(MOTION_Z_DEBIAS_FOOT_SOLE_Z, abs=0.003)
  assert s["foot_sites"] == ["left_foot", "right_foot"]
  assert info["dof"] == 29 and all(j["stiffness"] > 0 for j in info["joints"])
  assert custom.task_id_for("h1") == "Wbc-H1"
  assert custom.task_id_for("my_bot") == "Wbc-My-Bot"
  assert custom.task_id_for("h2x") == "Wbc-H2X"


def test_custom_robot_save_register_delete(preset_dir) -> None:
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.motion.robot_assets import get_robot_mjcf_path
  from wbc_mjlab.robots import custom
  from wbc_mjlab.tasks import list_wbc_task_ids

  folder = _robot_folder(preset_dir.parent, "zz_bot")
  info = custom.inspect_model(folder / "model.xml")
  s = info["suggested"]
  spec = custom.CustomRobotSpec(
    id="zz_bot",
    model="model.xml",
    anchor_body=s["anchor_body"],
    keybodies=s["keybodies"],
    ee_bodies=s["ee_bodies"],
    feet_bodies=s["feet_bodies"],
    foot_sites=s["foot_sites"],
    foot_sole_z=s["foot_sole_z"],
    init_height=info["init_height"],
    joints={
      j["name"]: custom.JointGains(
        j["effort"], j["stiffness"], j["damping"], j["armature"]
      )
      for j in info["joints"]
    },
  )
  check = custom.settle_test(spec)
  assert check["ok"], check["checks"]
  bad = custom.settle_test(
    dataclasses.replace(spec, init_height=spec.init_height - 0.2)
  )
  assert not bad["ok"]
  with pytest.raises(ValueError):
    custom.save_spec(dataclasses.replace(spec, keybodies=["no_such_body"]))
  custom.save_spec(spec)
  try:
    assert {"Wbc-Zz-Bot", "Wbc-Zz-Bot-SE", "Wbc-Zz-Bot-Terrain-Scan"} <= set(
      list_wbc_task_ids()
    )
    rough = load_env_cfg("Wbc-Zz-Bot-Terrain-Scan")
    assert rough.scene.terrain.terrain_type == "generator"
    assert "height_scan" in rough.observations["actor"].terms
    scan = next(x for x in rough.scene.sensors if x.name == "height_scan")
    assert scan.frame.name == "pelvis"
    assert (
      load_env_cfg(
        "Wbc-Zz-Bot-Terrain-Scan", play=True
      ).scene.terrain.max_init_terrain_level
      is None
    )
    assert get_robot_mjcf_path("zz_bot") == spec.prepared
    env = load_env_cfg("Wbc-Zz-Bot")
    assert env.commands["motion"].body_names == tuple(spec.keybodies)
    assert len(env.scene.entities["robot"].articulation.actuators) == 23
    assert (
      env.observations["actor"].terms["base_ang_vel"].params["sensor_name"]
      == "robot/imu_ang_vel"
    )
    se = load_env_cfg("Wbc-Zz-Bot-SE").observations["actor"].terms
    assert "base_lin_vel" in se and "ref_base_height" not in se
    assert custom.load_spec("zz_bot").keybodies == spec.keybodies
    assert [r.id for r in custom.list_specs()] == ["zz_bot"]
  finally:
    custom.delete("zz_bot")
  assert not [t for t in list_wbc_task_ids() if t.startswith("Wbc-Zz-Bot")]
  assert not folder.exists()


def test_prepare_cleans_models(tmp_path) -> None:
  import mujoco

  from wbc_mjlab.robots import custom

  (tmp_path / "bot.xml").write_text(
    """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
    <body name="base" pos="0 0 1"><geom type="box" size=".1 .1 .1"/>
      <body name="leg"><joint name="hip" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -.5" size=".04"/></body></body></worldbody>
    <actuator><motor joint="hip" ctrlrange="-1 1" gear="40"/></actuator>
    <keyframe><key name="home" qpos="0.3"/></keyframe></mujoco>"""
  )
  spec, info = custom.prepare(tmp_path / "bot.xml")
  m = spec.compile()
  assert m.nu == 0 and m.nkey == 0 and m.ngeom == 2
  assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
  names = {
    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)
  }
  assert {"imu_ang_vel", "imu_lin_vel", "root_angmom"} <= names
  assert info["efforts"] == {"hip": 40.0}
  (tmp_path / "bot.urdf").write_text(
    '<robot name="r"><link name="a"><visual><geometry><mesh filename="package://r/m.dae"/></geometry></visual></link></robot>'
  )
  with pytest.raises(ValueError, match="dae"):
    custom.prepare(tmp_path / "bot.urdf")


def test_terrain_preset_knobs_and_sensor_guard() -> None:
  from wbc_mjlab.studio import presets

  env, agent = presets.build("Wbc-G1-Terrain-Scan", {})
  schema = presets.describe(env, agent)
  keys = {f["key"] for f in schema["terrain"]}
  assert {"curriculum.start_row", "curriculum.num_rows", "rough.noise_range"} <= keys
  scan = next(
    t
    for t in schema["observations"]["groups"][0]["terms"]
    if t["name"] == "height_scan"
  )
  assert scan["dim"] == 54 and scan["deployable"]
  ov = {"terrain": {"curriculum.start_row": 3, "rough.noise_range": [0.0, 0.2]}}
  env, _ = presets.build("Wbc-G1-Terrain-Scan", ov)
  assert env.scene.terrain.max_init_terrain_level == 3
  assert env.scene.terrain.terrain_generator.sub_terrains["rough"].noise_range == (
    0.0,
    0.2,
  )
  assert presets.count_changes(ov) == 2
  play, _ = presets.build("Wbc-G1-Terrain-Scan", ov, play=True)
  assert play.scene.terrain.max_init_terrain_level is None
  flat, _ = presets.build("Wbc-G1", {})
  assert presets.describe(flat, agent)["terrain"] == []
  lib = {x["name"]: x for x in presets.library("g1", 29, 14, flat)}
  assert lib["height_scan"]["requires"] == "height_scan"
  with pytest.raises(ValueError, match="Terrain-Scan"):
    presets.build(
      "Wbc-G1", {"observations": {"actor": {"terms": {"height_scan": {"add": True}}}}}
    )
