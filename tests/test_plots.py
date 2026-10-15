"""Studio Plots: clip channels (Data) and the Play recorder (robot vs reference, rewards)."""

from __future__ import annotations

import base64
import json
import struct
import time

import numpy as np
import pytest

from wbc_mjlab.data_paths import repo_root
from wbc_mjlab.studio import jobs as jobmod
from wbc_mjlab.studio import plots
from wbc_mjlab.studio.plots import PlayRecorder, encode_rows, joint_group

SAMPLES = repo_root() / "data" / "g1" / "samples"


def _rows(b64: str, cols: int) -> np.ndarray:
  return np.frombuffer(base64.b64decode(b64), dtype="<f4").reshape(-1, cols)


def _full_ref(ref: np.ndarray, cols: list[int], n: int) -> np.ndarray:
  """A Play record's reference (``ref_cols`` only) spread over all ``n`` channels."""
  out = np.full((ref.shape[0], n), np.nan, np.float32)
  out[:, cols] = ref
  return out


def test_joint_groups() -> None:
  assert joint_group("left_knee_joint") == "Left leg"
  assert joint_group("right_shoulder_pitch_joint") == "Right arm"
  assert joint_group("waist_yaw_joint") == "Torso"
  assert joint_group("L_hip_roll") == "Left leg"
  assert joint_group("mystery") == "Other"


def test_effort_limits_from_the_model_and_actuators() -> None:
  from types import SimpleNamespace

  import mujoco

  m = mujoco.MjModel.from_xml_string(
    """<mujoco><worldbody><body name="robot/a">
         <joint name="robot/hip" type="hinge" actuatorfrcrange="-50 50"/>
         <geom size="0.1"/>
         <body name="robot/b"><joint name="robot/knee" type="hinge"/><geom size="0.1"/>
         <body name="robot/c"><joint name="robot/ankle" type="hinge"/><geom size="0.1"/>
         <body name="robot/d"><joint name="robot/toe" type="hinge"/><geom size="0.1"/>
         </body></body></body></body></worldbody>
       <actuator>
         <motor joint="robot/hip" forcerange="-80 80"/>
         <motor joint="robot/knee" gear="2" forcerange="-30 30"/>
         <motor joint="robot/ankle"/>
       </actuator></mujoco>"""
  )
  unitree = SimpleNamespace(
    cfg=SimpleNamespace(effort_limit=float("inf"), y1=20.0, y2=25.0),
    target_names=["ankle"],
  )
  capped = SimpleNamespace(cfg=SimpleNamespace(effort_limit=40.0), target_names=["hip"])
  session = SimpleNamespace(
    env=SimpleNamespace(sim=SimpleNamespace(mj_model=m)),
    robot=SimpleNamespace(actuators=[unitree, capped]),
  )
  lim = plots.effort_limits(session)
  assert lim["hip"] == (-40.0, 40.0, "actuator")  # the tightest of 50 (joint), 80, 40
  assert lim["knee"] == (-60.0, 60.0, "model")  # forcerange × gear
  assert lim["ankle"][:2] == (-25.0, 25.0) and "peak" in lim["ankle"][2]
  assert "toe" not in lim


@pytest.fixture(scope="module")
def core():
  from wbc_mjlab.studio.core import StudioCore

  return StudioCore(robot="g1", device="cpu")


def test_clip_channels_with_raw_source(core) -> None:
  if not (SAMPLES / "npz" / "walk1_subject1.npz").is_file():
    pytest.skip("needs converted samples")
  core.data.load("samples", "walk1_subject1", source="npz")
  info = core.data.plot_data()
  chans = info["channels"]
  cols = len(chans)
  data = _rows(info["data"], cols)
  assert data.shape[0] == info["frames"] and np.isfinite(data).all()
  groups = {c["group"] for c in chans}
  assert {"Root position", "Root rotation", "Left leg", "Right arm", "Torso"} <= groups
  knee = next(i for i, c in enumerate(chans) if c["name"] == "left_knee_joint")
  assert chans[knee]["unit"] == "rad" and chans[knee]["lower"] < chans[knee]["upper"]
  assert data[:, 2].min() > 0.3  # root height of a walk
  # converted clip vs its raw source: resampled to the clip's frames, close to it
  assert info["ref"] and info["ref_label"] == "raw source"
  ref = _rows(info["ref"], cols)
  both = np.isfinite(ref[:, knee])
  assert both.mean() > 0.9
  assert np.abs(ref[both, knee] - data[both, knee]).mean() < 0.05
  assert core.data.plot_data() is info  # cached until another clip opens
  core.data.close()
  with pytest.raises(RuntimeError):
    core.data.plot_data()


def test_play_recorder_robot_vs_reference(monkeypatch) -> None:
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
  try:
    for _ in range(5):
      c.play.step_once()
    c.play.set_clip("walk1_subject1")
    for _ in range(40):
      c.play.step_once()
    out = c.play.plot_data()
    assert out["fresh"] and out["end"] == 45 and out["env"] == 0
    chans = out["channels"]
    n = len(chans)
    data, meta = out["data"], out["meta"]
    ref = _full_ref(out["ref"], out["ref_cols"], n)
    assert data.shape == (45, n) and meta.shape == (45, 3)
    idx = {c["id"]: i for i, c in enumerate(chans)}
    knee = idx["left_knee_joint"]
    # reference present from row 1 (row k is compared to the frame set at the end of step
    # k - 1, like the rewards), and it is each joint's own: the robot is far closer to it
    # than to the neighbouring joint's reference (policy quality aside)
    assert np.isnan(ref[0, knee]) and np.isfinite(ref[1:, knee]).all()
    joint_cols = [
      i for i, c in enumerate(chans) if c["id"] == c["name"] and c["unit"] == "rad"
    ]
    jd, jr = data[1:, joint_cols], ref[1:, joint_cols]
    own = np.abs(jd - jr).mean()
    assert own < 0.2 and own < 0.5 * np.abs(jd - np.roll(jr, 1, axis=1)).mean()
    assert np.isnan(
      ref[:, idx["tau:left_knee_joint"]]
    ).all()  # no reference for torques
    # G1's effort limits: the Unitree actuators' envelope peak (none in its MJCF)
    tau = chans[idx["tau:left_knee_joint"]]
    assert tau["lower"] == -tau["upper"] and tau["upper"] > 50 and tau["limit_note"]
    assert all(c["lower"] is not None for c in chans if c["group"] == "Joint torques")
    # the plotted torque is the motor's (the envelope clips it); the friction the Unitree
    # actuators take off it is its own group, so the joint gets torque - friction
    tq = [i for i, c in enumerate(chans) if c["group"] == "Joint torques"]
    fr = [i for i, c in enumerate(chans) if c["group"] == "Actuator friction"]
    assert len(fr) == len(tq) and np.abs(data[:, fr]).max() > 0
    upper = np.array([chans[i]["upper"] for i in tq])
    assert (np.abs(data[:, tq]) <= upper + 1e-3).all()
    net = c.play.session.robot.data.qfrc_actuator[0].cpu().numpy()
    assert np.allclose(data[-1, tq] - data[-1, fr], net, atol=1e-3)

    # the weighted terms add up to the total
    rew = [i for c, i in idx.items() if c.startswith("rew:") and c != "rew:total"]
    assert np.allclose(data[:, rew].sum(1), data[:, idx["rew:total"]], atol=1e-4)
    # the last 40 rows play the pinned clip
    names = out["clips"]
    assert {names[int(k)] for k in meta[-40:, 0]} == {"walk1_subject1"}
    # incremental fetch: only the new rows, no channel list; the new row's reference is
    # the frame the command held before the step (the one that step's rewards compared to)
    before = c.play.session.cmd.joint_pos[0].cpu().numpy()
    c.play.step_once()
    more = c.play.plot_data(since=45, epoch=out["epoch"])
    assert not more["fresh"] and (more["start"], more["end"]) == (45, 46)
    assert "channels" not in more
    joints = [
      i
      for cid, i in idx.items()
      if chans[i]["group"]
      in ("Left leg", "Right leg", "Torso", "Left arm", "Right arm")
    ]
    assert np.allclose(_full_ref(more["ref"], out["ref_cols"], n)[0, joints], before)
    # the binary form the Plots window pulls
    blob = encode_rows(more)
    (hlen,) = struct.unpack_from("<I", blob)
    head = json.loads(blob[4 : 4 + hlen])
    assert head["cols"] == [n, len(out["ref_cols"]), 3] and "data" not in head
    rows = np.frombuffer(blob[4 + hlen :], dtype="<f4")
    assert rows.size == n + len(out["ref_cols"]) + 3
    assert np.array_equal(rows[:n], more["data"][0], equal_nan=True)
    # every env of a small session is recorded: switching (even paused) shows its whole
    # record at once, under its own epoch
    c.set_focus(1)
    assert c.scene.snapshot()["hud"]["focus"] == 1  # the HUD follows while paused
    other = c.play.plot_data(since=46, epoch=out["epoch"])
    assert (
      other["fresh"] and other["env"] == 1 and (other["start"], other["end"]) == (0, 46)
    )
    assert not np.allclose(
      other["data"][:, joint_cols], c.play.plot_data(env=0)["data"][:, joint_cols]
    )
    back = c.play.plot_data(env=0, since=46, epoch=out["epoch"])
    assert not back["fresh"] and back["end"] == 46
    # a restart drops the records
    c.play.restart()
    c.play.step_once()
    again = c.play.plot_data(env=0, since=46, epoch=out["epoch"])
    assert again["fresh"] and again["end"] - again["start"] == 1
    # the window keeps the newest rows only
    small = PlayRecorder(c.play.session, c.model, fps=50.0, horizon_s=0.1)
    for _ in range(8):
      small.record()
    kept = small.fetch()
    assert (kept["start"], kept["end"]) == (3, 8) and kept["data"].shape[0] == 5
    # more envs than tracks: a newly focused env takes over the least recently used track
    # and records from then on
    monkeypatch.setattr(plots, "MAX_TRACKS", 1)
    one = PlayRecorder(c.play.session, c.model, fps=50.0)
    assert one.num_tracks == 1
    one.record()
    first = one.fetch(env=0)
    one.follow(1)
    one.record()
    switched = one.fetch(env=1, epoch=first["epoch"])
    assert switched["fresh"] and (switched["start"], switched["end"]) == (1, 2)
  finally:
    c.play.stop()
