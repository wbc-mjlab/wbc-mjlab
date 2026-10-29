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
  assert "phase" not in p
  # the world model's first update compiles silently for minutes: say so
  with log.open("a") as f:
    f.write("[INFO] wm: first update: compiling (minutes; cached for later runs)\n")
  assert "compiling" in job.progress()["phase"]
  with log.open("a") as f:
    f.write("[INFO] wm: first update: compiled in 212 s\n")
  assert "phase" not in job.progress()

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
  if jobmod.find_run(jobmod.DEMO_LABEL) is None:
    pytest.skip("needs the demo checkpoint")
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
  # terminations flip live, in the same session: a term that always fires (an anchor height
  # threshold below zero) ends no episode while off, every episode once on
  s = c.play.session
  tm = s.env.termination_manager
  tm.get_term_cfg("anchor_pos").params["threshold"] = -1.0
  c.play.step_once()
  assert not tm.dones.any() and not tm.get_term("anchor_pos").any()
  args = (jobmod.DEMO_LABEL, "model.pt")
  live = {"motion": str(SAMPLES), "num_envs": 2}
  assert c.play.configure(*args, **live, terminations=True) == "terminations"
  assert c.play.session is s and c.play.state()["terminations"]
  c.play.step_once()
  assert tm.terminated[:2].all()
  assert c.play.configure(*args, **live, terminations=False) == "terminations"
  c.play.step_once()
  assert not tm.dones.any()
  # the env count changes in place up to the session's capacity
  assert c.play.configure(*args, **{**live, "num_envs": 4}) == "resized"
  assert c.play.session is s and c.play.state()["num_envs"] == 4
  c.play.stop()


def test_play_env_count_bounds(core) -> None:
  from wbc_mjlab.studio.play import MAX_ENVS

  for n in (0, MAX_ENVS + 1):
    with pytest.raises(ValueError, match="environments"):
      core.play.configure(jobmod.DEMO_LABEL, "model.pt", num_envs=n)


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
  schema = presets.describe(load_env_cfg("Wbc-G1"))
  assert {r["name"] for r in schema["rewards"]} >= {"motion_body_pos", "survival"}
  keys = {f["key"] for f in schema["command"]}
  assert {"rsi.bin_width_s", "rsi.sampling_mode", "env.episode_length_s"} <= keys
  assert "agent" not in schema  # training settings are recipes

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


def test_recipes_train_any_preset_with_either_method(preset_dir) -> None:
  """Recipes (a method and its settings) are saved apart from presets; the Train form lists
  every setting of each method with the preset's value as default."""
  from wbc_mjlab.studio import presets, recipes
  from wbc_mjlab.studio.training import TrainingService
  from wbc_mjlab.tasks import register_all_wbc_tasks

  register_all_wbc_tasks()
  assert {r["name"] for r in recipes.list_recipes()} == {
    "ppo",
    "wm",
    "wm-24gb",
    "wm-encoder",
  }
  recipes.save("zz_fast", "wm", {"algorithm.horizon": 16}, "short horizon")
  r = recipes.load("zz_fast")
  assert r["algo"] == "wm" and r["settings"] == {"algorithm.horizon": 16}
  assert not r["builtin"] and "zz_fast" in {x["name"] for x in recipes.list_recipes()}
  with pytest.raises(ValueError, match="built-in"):
    recipes.save("ppo", "ppo", {})
  with pytest.raises(ValueError, match="unknown algorithm"):
    recipes.save("zz_bad", "sac", {})

  # a preset's own PPO settings from before recipes still apply, and move to a recipe
  presets.save("zz_methods", "Wbc-G1", {"agent": {"actor.hidden_dims": [256, 128]}})
  tid = presets.register("zz_methods")
  m = TrainingService(None).methods(tid)
  wm = next(x for x in m["methods"] if x["id"] == "wm")
  fields = {f["key"]: f for f in wm["settings"]}
  assert (
    fields["algorithm.horizon"]["main"] and fields["algorithm.horizon"]["value"] == 32
  )
  assert fields["actor.hidden_dims"]["value"] == [256, 128]  # the policy is shared
  assert fields["actor.hidden_dims"]["section"] == "Policy network"
  assert fields["actor.goal_terms"]["kind"] == "strings"
  assert fields["actor.goal_terms"]["visible_if"] == {
    "actor.architecture": "motion_encoder"
  }
  assert recipes.check_settings(tid, "wm", {"algorithm.goal_terms": []}) == {
    "actor.goal_terms": []
  }
  assert "algorithm.horizon" not in {
    f["key"] for f in next(x for x in m["methods"] if x["id"] == "ppo")["settings"]
  }
  recipe = presets.move_agent_to_recipe("zz_methods")
  assert recipe["algo"] == "ppo"
  assert recipe["settings"] == {"actor.hidden_dims": [256, 128]}
  assert "agent" not in presets.load("zz_methods")["overrides"]
  recipes.delete("zz_fast")
  recipes.delete("zz_methods")
  presets.delete("zz_methods")


class _Core:
  def log(self, *a) -> None:
    pass

  def emit(self, *a, **k) -> None:
    pass


def test_launch_passes_the_method_and_its_settings(monkeypatch, tmp_path) -> None:
  from wbc_mjlab.studio.training import TrainingService

  launched = {}

  def fake_launch(kind, args, *, label, meta=None):
    launched.update(args=args, meta=meta)
    return jobmod.Job(name="j", kind=kind, pid=0, log="", started=0.0, command=[])

  monkeypatch.setattr(jobmod, "launch", fake_launch)
  t = TrainingService(_Core())
  t.launch(
    "Wbc-G1",
    "samples",
    algo="wm",
    algo_settings={
      "algorithm.imagined_rollouts": 2048,
      "algorithm.model_hidden_dims": [512, 512],
    },
    recipe="wm-24gb",
  )
  args = launched["args"]
  assert args[args.index("--algo") + 1] == "wm"
  assert args[args.index("--agent.algorithm.imagined-rollouts") + 1] == "2048"
  # mjlab's CLI takes tuples as Python literals
  assert args[args.index("--agent.algorithm.model-hidden-dims") + 1] == "(512, 512)"
  assert launched["meta"]["algo"] == "wm" and launched["meta"]["recipe"] == "wm-24gb"
  with pytest.raises(ValueError, match="not a setting"):
    t.launch("Wbc-G1", "samples", algo="ppo", algo_settings={"algorithm.horizon": 8})
  with pytest.raises(ValueError, match="unknown algorithm"):
    t.launch("Wbc-G1", "samples", algo="sac")

  # a run continues with the method it was trained with
  run_dir = tmp_path / "wbc_g1" / "run"
  (run_dir / "params").mkdir(parents=True)
  (run_dir / "params" / "agent.yaml").write_text(
    "algorithm:\n  class_name: PPO\nactor:\n  hidden_dims: [64, 32]\n"
  )
  run = jobmod.Run(experiment="wbc_g1", name="run", path=run_dir)
  monkeypatch.setattr(jobmod, "find_run", lambda label: run)
  monkeypatch.setattr(jobmod.Run, "task_id", lambda self: "Wbc-G1")
  with pytest.raises(ValueError, match="trained with PPO"):
    t.launch("Wbc-G1", "samples", algo="wm", resume_run="wbc_g1/run")
  t.launch(
    "Wbc-G1",
    "samples",
    algo="ppo",
    resume_run="wbc_g1/run",
    algo_settings={"actor.hidden_dims": [512, 512]},
  )
  args = launched["args"]
  assert "--agent.resume" in args
  # the run's checkpoint fixes its network, whatever the form says
  assert args[args.index("--agent.actor.hidden-dims") + 1] == "(64, 32)"


def test_evaluate_saved_checkpoint_job_and_download(core, monkeypatch, tmp_path):
  from fastapi.testclient import TestClient

  from wbc_mjlab.studio.web.server import create_app

  run_dir = tmp_path / "wbc_g1" / "run"
  (run_dir / "params").mkdir(parents=True)
  (run_dir / "model_5.pt").touch()
  (run_dir / "model_10.pt").touch()
  run = jobmod.Run(experiment="wbc_g1", name="run", path=run_dir)
  monkeypatch.setattr(jobmod, "find_run", lambda label: run)
  launched = {}
  log = tmp_path / "evaluate.log"
  log.write_text("[EVAL] clips 3/3\n[job exit 0]\n")

  def launch(kind, args, *, label, meta):
    launched.update(args=args, meta=meta)
    return jobmod.Job(
      name="evaluation",
      kind=kind,
      pid=999999999,
      log=str(log),
      started=0.0,
      command=[],
      meta=meta,
    )

  monkeypatch.setattr(jobmod, "launch", launch)
  job = core.training.evaluate(run.label)
  assert launched["args"][0] == "wbc_mjlab.scripts.evaluate"
  assert launched["meta"]["checkpoint"] == str(run_dir / "model_10.pt")
  assert job.progress() == {"done": 3, "total": 3, "fraction": 1.0}
  with pytest.raises(ValueError, match="select a saved checkpoint"):
    core.training.evaluate(run.label, "../other.pt")
  monkeypatch.setattr(jobmod, "find_job", lambda name: job)
  result = run_dir / "benchmark_model_10.json"
  result.write_text('{"summary": {"clips": 3}}')
  client = TestClient(create_app(core), client=("127.0.0.1", 5000))
  launch_response = client.post(
    "/api/runs/evaluate",
    json={"run": run.label, "motion": "/tmp/chosen-clips", "num_envs": 4},
  )
  assert launch_response.status_code == 200
  args = launched["args"]
  assert args[args.index("--dataset-path") + 1] == "/tmp/chosen-clips"
  assert args[args.index("--num-envs") + 1] == "4"
  response = client.get("/api/jobs/evaluation/result")
  assert response.status_code == 200
  assert response.json()["summary"]["clips"] == 3
  assert "attachment" in response.headers["content-disposition"]


def test_loss_curves_follow_the_method(monkeypatch, tmp_path) -> None:
  from torch.utils.tensorboard import SummaryWriter

  from wbc_mjlab.studio.training import TrainingService

  w = SummaryWriter(str(tmp_path))
  for tag in (
    "Loss/wm_dynamics",
    "Loss/critic",
    "Loss/learning_rate",
    "Policy/mean_std",
  ):
    w.add_scalar(tag, 1.0, 0)
  w.close()
  run = jobmod.Run(experiment="x", name="r", path=tmp_path)
  monkeypatch.setattr(jobmod, "find_run", lambda label: run)
  series = TrainingService(None).curves(run="x/r", group="loss")["series"]
  assert set(series) == {"Loss/wm_dynamics", "Loss/critic", "Policy/mean_std"}


def test_job_status_follows_the_exit_code(tmp_path) -> None:
  import subprocess

  proc = subprocess.Popen(["true"])
  proc.wait()  # a pid that is gone
  log = tmp_path / "job.log"

  def status(text: str) -> str:
    log.write_text(text)
    job = jobmod.Job(
      name="j", kind="train", pid=proc.pid, log=str(log), started=0.0, command=[]
    )
    return job.status()

  assert status("Unrecognized options: 512\n[job exit 2]\n") == "failed"
  assert status("Mean reward: 1.0\n[job exit 0]\n") == "done"
  assert status("[WARN] export Error, training continues\n[job exit 0]\n") == "done"
  # jobs started before the exit marker: read from the output
  assert status("Unrecognized options: 512\n") == "failed"
  assert status("Learning iteration 9/10\n") == "done"


def test_job_log_for_the_viewer(monkeypatch, tmp_path) -> None:
  """The log viewer reads a job's output as a terminal shows it (colour codes and redrawn
  progress lines resolved), up to its last lines, and the whole file downloads."""
  import subprocess

  from wbc_mjlab.studio.training import TrainingService

  proc = subprocess.Popen(["true"])
  proc.wait()  # a pid that is gone
  log = tmp_path / "job.log"
  lines = [f"\x1b[1m Learning iteration {i}/9 \x1b[0m" for i in range(10)]
  log.write_text("\n".join([*lines, "bar 10%\rbar 100%", "[job exit 0]"]) + "\n")
  job = jobmod.Job(
    name="j", kind="train", pid=proc.pid, log=str(log), started=0.0, command=[]
  )
  monkeypatch.setattr(jobmod, "find_job", lambda name: job if name == "j" else None)
  t = TrainingService(_Core())
  d = t.job_log("j", lines=3)
  assert d["log"].split("\n") == [
    " Learning iteration 9/9 ",
    "bar 100%",
    "[job exit 0]",
  ]
  assert d["size"] == log.stat().st_size and d["status"] == "done"
  assert "\x1b" not in t.job_log("j")["log"]
  assert t.job_log_path("j") == log
  with pytest.raises(ValueError, match="no job"):
    t.job_log("missing")


def _job(name, kind, alive, *, run=None, robot=""):
  """A saved job whose process is alive (this one) or gone, its log naming ``run``."""
  import os
  import subprocess

  if alive:
    pid = os.getpid()
  else:
    proc = subprocess.Popen(["true"])
    proc.wait()
    pid = proc.pid
  log = jobmod._jobs_dir() / f"{name}.log"
  head = f"[INFO] Logging experiment in directory: {run}\n" if run else ""
  log.write_text(head + ("" if alive else "[job exit 1]\n"))
  meta = {"robot": robot} if robot else {}
  jobmod.Job(name, kind, pid, str(log), 0.0, [], meta=meta).save()


def test_clear_finished_jobs(monkeypatch, tmp_path) -> None:
  """Clearing removes finished jobs of the kind (and robot) asked for, records and logs,
  and leaves running ones."""
  monkeypatch.setattr(jobmod, "logs_dir", lambda: tmp_path)
  _job("done_train", "train", False)
  _job("live_train", "train", True)
  _job("done_g1", "convert", False, robot="g1")
  _job("done_h1", "convert", False, robot="h1")

  assert jobmod.clear_jobs("convert", "g1") == ["done_g1"]
  assert jobmod.clear_jobs("train") == ["done_train"]
  left = {p.name for p in jobmod._jobs_dir().iterdir()}
  assert left == {
    f"{n}.{s}" for n in ("live_train", "done_h1") for s in ("json", "log")
  }


def test_runs_on_disk(monkeypatch, tmp_path) -> None:
  """The storage view maps runs to the jobs that wrote them; runs are pruned to their last
  checkpoint (plus multiples of ``every``) or deleted with their jobs, never while a job
  trains them."""
  from wbc_mjlab.studio.training import TrainingService

  monkeypatch.setattr(jobmod, "logs_dir", lambda: tmp_path)
  root = tmp_path / "rsl_rl" / "exp"
  for run, its in (("a", (0, 100, 200, 300)), ("b", (0,)), ("c", (0, 100))):
    (root / run).mkdir(parents=True)
    for it in its:
      (root / run / f"model_{it}.pt").write_bytes(b"x" * 10)
  (root / "empty").mkdir()  # a job that died before its first checkpoint
  _job("failed", "train", False, run=root / "b")
  _job("training", "train", True, run=root / "c")
  t = TrainingService(_Core())

  rows = {r["label"]: r for r in t.storage()["runs"]}
  assert set(rows) == {"exp/a", "exp/b", "exp/c", "exp/empty"}
  assert rows["exp/a"]["checkpoints"] == 4 and rows["exp/a"]["last"] == 300
  assert rows["exp/a"]["prunable"] == 30 and rows["exp/empty"]["checkpoints"] == 0
  assert (rows["exp/b"]["job"], rows["exp/b"]["status"]) == ("failed", "failed")
  assert rows["exp/c"]["busy"] and not rows["exp/a"]["busy"]

  assert t.prune_runs(["exp/a"], every=200) == 10
  assert sorted(p.name for p in (root / "a").glob("*.pt")) == [
    "model_0.pt",
    "model_200.pt",
    "model_300.pt",
  ]
  assert t.prune_runs(["exp/a"]) == 20
  assert [p.name for p in (root / "a").glob("*.pt")] == ["model_300.pt"]
  with pytest.raises(ValueError, match="training it"):
    t.delete_runs(["exp/a", "exp/c"])
  assert (root / "a").is_dir()  # nothing deleted when one run is refused
  with pytest.raises(ValueError, match="no run"):
    t.delete_runs(["exp/nope"])

  assert t.delete("failed", run=True) == 10
  assert not (root / "b").exists() and jobmod.find_job("failed") is None
  assert t.delete_runs(["exp/empty"]) == 0 and not (root / "empty").exists()


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
  schema = presets.describe(presets.build("Wbc-G1", {}, modules=["se"])[0])
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


def test_reference_terms_look_ahead(preset_dir) -> None:
  """A preset sets how many reference frames a term covers (``frames``, ``stride``): the
  term's noise repeats per frame, the Studio's width and the deploy export follow."""
  from wbc_mjlab.export.tracking_params_yaml import build_wbc_tracking_params
  from wbc_mjlab.studio import presets

  ov = {
    "observations": {
      "actor": {"terms": {"ref_base_lin_vel_b": {"frames": 4, "stride": 2}}}
    }
  }
  env, _ = presets.build("Wbc-G1", ov)
  term = env.observations["actor"].terms["ref_base_lin_vel_b"]
  assert term.params == {"command_name": "motion", "frames": 4, "stride": 2}
  assert term.noise.n_max == (0.1, 0.1, 0.05) * 4
  # the other reference terms are untouched
  assert "frames" not in env.observations["actor"].terms["ref_base_ang_vel_b"].params
  rows = {
    t["name"]: t for t in presets.describe(env)["observations"]["groups"][0]["terms"]
  }
  assert (
    rows["ref_base_lin_vel_b"]["dim"] == 12
    and rows["ref_base_lin_vel_b"]["frames"] == 4
  )
  assert rows["ref_joint_pos"]["takes_frames"] and not rows["joint_pos"]["takes_frames"]
  params = build_wbc_tracking_params(env, robot_id="g1", has_state_estimation=False)
  exported = params["actor_observations"]["ref_base_lin_vel_b"]
  assert exported["dim"] == 12 and exported["params"]["frames"] == 4
  assert exported["params"]["stride"] == 2
  # Lookahead's window: separate reference terms, 10 frames each
  sm, _ = presets.build("Wbc-G1-Lookahead", {})
  sm_params = build_wbc_tracking_params(sm, robot_id="g1", has_state_estimation=False)
  obs = sm_params["actor_observations"]
  assert obs["ref_joint_pos"]["dim"] == 290 and obs["ref_anchor_shift_b"]["dim"] == 30
  assert obs["joint_pos"]["history_length"] == 10
  assert sm_params["tracking"]["wbc_command_dim"] == 10 * (2 * 29 + 9)


def test_play_refuses_a_changed_observation_layout(tmp_path) -> None:
  """A run whose actor observation differs from what its task builds now (order, frames
  ahead, history) is refused: the same width would load with scrambled inputs."""
  import yaml

  from wbc_mjlab.run_params import actor_layout_change
  from wbc_mjlab.studio import presets

  env, _ = presets.build("Wbc-G1", {})
  actor = env.observations["actor"]
  record = {
    "history_length": actor.history_length,
    "terms": {
      n: {"params": dict(t.params or {}), "history_length": t.history_length}
      for n, t in actor.terms.items()
    },
  }
  (tmp_path / "params").mkdir()
  path = tmp_path / "params" / "env.yaml"
  path.write_text(yaml.safe_dump({"observations": {"actor": record}}, sort_keys=False))
  assert actor_layout_change(tmp_path, env) == ""
  record["terms"]["ref_joint_pos"]["params"]["frames"] = 5
  path.write_text(yaml.safe_dump({"observations": {"actor": record}}, sort_keys=False))
  assert "ref_joint_pos" in actor_layout_change(tmp_path, env)
  record["terms"] = dict(reversed(list(record["terms"].items())))
  path.write_text(yaml.safe_dump({"observations": {"actor": record}}, sort_keys=False))
  assert "trained with actor terms" in actor_layout_change(tmp_path, env)
  assert actor_layout_change(tmp_path / "missing", env) == ""


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
  if jobmod.find_run(jobmod.DEMO_LABEL) is None:
    pytest.skip("needs the demo checkpoint")
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

  env, _ = presets.build("Wbc-G1-Terrain-Scan", {})
  schema = presets.describe(env)
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
  assert presets.describe(flat)["terrain"] == []
  lib = {x["name"]: x for x in presets.library("g1", 29, 14, flat)}
  assert lib["height_scan"]["requires"] == "height_scan"
  with pytest.raises(ValueError, match="Terrain-Scan"):
    presets.build(
      "Wbc-G1", {"observations": {"actor": {"terms": {"height_scan": {"add": True}}}}}
    )
