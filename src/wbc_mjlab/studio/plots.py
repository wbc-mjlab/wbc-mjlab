"""Plot channels for the Studio's Plots window.

Two sources, one layout (``{channels: [{id, name, group, unit, lower, upper}], data, ref}``
with ``data`` / ``ref`` (rows, channels) float32 values):

- **Clip** (Data step) — :func:`clip_channels`: the open clip's root position, root roll /
  pitch / yaw, every joint (with its limits) and the feet's sole heights. A converted clip
  previewed against its raw source gets the source as ``ref`` (resampled to the clip's
  frames), drawn dashed. Sent once per clip as JSON (values little-endian, base64).
- **Play** — :class:`PlayRecorder`: a rolling record of the session's envs while the
  policy runs: the robot's root, joints, PD targets, torques and actions, the reference it
  tracks as ``ref``, per-keybody tracking errors and every reward term (weighted, as summed
  into the reward, and raw: before the weight). Row ``k`` pairs the robot after step ``k``
  with the reference that step's rewards compared it to (the command advances after the
  rewards, so that is the frame set at the end of step ``k - 1``). Recording is one
  vectorized write per step for every recorded env, on the sim device; the client pulls new
  rows as binary (:func:`encode_rows`) with every stage frame.
"""

from __future__ import annotations

import base64
import json
import re
import struct
import threading
import time

import numpy as np
import torch

HISTORY_S = 60.0  # seconds of Play kept per recording

_SIDE = (
  ("left", re.compile(r"(^|[_\W])l([_\W]|$)|left", re.I)),
  ("right", re.compile(r"(^|[_\W])r([_\W]|$)|right", re.I)),
)
_PART = (
  ("leg", re.compile(r"hip|thigh|knee|shin|calf|ankle|foot|toe|leg", re.I)),
  ("arm", re.compile(r"shoulder|elbow|wrist|hand|arm|finger|thumb|palm", re.I)),
  (
    "torso",
    re.compile(r"waist|torso|spine|pelvis|trunk|chest|abdomen|lumbar|back", re.I),
  ),
  ("head", re.compile(r"head|neck", re.I)),
)


def joint_group(name: str) -> str:
  """ "Left leg", "Right arm", "Torso", "Head" or "Other" from a joint's name."""
  part = next((label for label, p in _PART if p.search(name)), None)
  side = next((label for label, p in _SIDE if p.search(name)), None)
  if part in ("leg", "arm"):
    return f"{side.capitalize()} {part}" if side else f"{part.capitalize()}s"
  return {"torso": "Torso", "head": "Head"}.get(part or "", "Other")


def _b64(rows: np.ndarray) -> str:
  return base64.b64encode(np.ascontiguousarray(rows, dtype="<f4").tobytes()).decode()


def _rpy(quat_wxyz: np.ndarray) -> np.ndarray:
  """Roll, pitch, yaw (rad, unwrapped over time) of quaternions (T, 4)."""
  w, x, y, z = (quat_wxyz[:, i] for i in range(4))
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return np.unwrap(np.stack([roll, pitch, yaw], axis=1), axis=0)


def _rpy_t(q: torch.Tensor) -> torch.Tensor:
  """Roll, pitch, yaw of quaternions (..., 4) wxyz on the sim device (wrapped: the client
  unwraps channels marked ``wrap``)."""
  w, x, y, z = q.unbind(-1)
  return torch.stack(
    (
      torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
      torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0)),
      torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ),
    dim=-1,
  )


def _ch(cid: str, name: str, group: str, unit: str, **extra) -> dict:
  return {"id": cid, "name": name, "group": group, "unit": unit, **extra}


def _root_channels(wrap: bool = False) -> list[dict]:
  out = [_ch(f"root_{a}", f"root {a}", "Root position", "m") for a in "xyz"]
  out += [
    _ch(
      f"root_{a}",
      f"root {a}",
      "Root rotation",
      "rad",
      **({"wrap": True} if wrap else {}),
    )
    for a in ("roll", "pitch", "yaw")
  ]
  return out


def _joint_channel(
  name: str, limits: dict, group: str | None = None, prefix: str = ""
) -> dict:
  lo, hi = limits.get(name, (None, None))
  return _ch(
    f"{prefix}{name}", name, group or joint_group(name), "rad", lower=lo, upper=hi
  )


def hinge_limits(model) -> dict[str, tuple[float, float]]:
  """Joint name → (lower, upper) for the model's limited hinge / slide joints."""
  from wbc_mjlab.studio.clips import _hinges

  return {n: (lo, hi) for n, _, _, limited, lo, hi in _hinges(model) if limited}


def effort_limits(session) -> dict[str, tuple[float, float, str]]:
  """Joint name → (lower, upper, source) torque limits of a Play session's robot: the
  tightest of the compiled model's joint ``actuatorfrcrange`` / joint-actuator ``forcerange``
  (× gear) — from the asset's MJCF —, the actuators' constant ``effort_limit`` and the peak of
  a torque-speed envelope (Unitree actuators: the real cap is lower at speed). Joints with
  none of them are left out."""
  import mujoco

  m = session.env.sim.mj_model
  out: dict[str, tuple[float, float, str]] = {}

  def offer(name: str, lo: float, hi: float, source: str) -> None:
    name = name.rsplit("/", 1)[-1]  # the entity prefix ("robot/")
    lo, hi = sorted((float(lo), float(hi)))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo <= 0:
      return
    old = out.get(name)
    if old is None or hi - lo < old[1] - old[0]:
      out[name] = (lo, hi, source)

  for j in range(m.njnt):
    if m.jnt_actfrclimited[j]:
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
      offer(name, *m.jnt_actfrcrange[j], "model")
  for a in range(m.nu):
    if (
      m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_forcelimited[a]
    ):
      j = int(m.actuator_trnid[a, 0])
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
      gear = float(m.actuator_gear[a, 0])
      offer(name, *(m.actuator_forcerange[a] * gear), "model")
  for act in session.robot.actuators:
    cfg = act.cfg
    cap = getattr(cfg, "effort_limit", None)
    y1, y2 = getattr(cfg, "y1", None), getattr(cfg, "y2", None)
    for name in act.target_names:
      if cap is not None and np.isfinite(cap) and cap > 0:
        offer(name, -cap, cap, "actuator")
      if y1 is not None and y2 is not None:
        peak = max(float(y1), float(y2))
        offer(name, -peak, peak, "actuator peak (lower at speed)")
  return out


# ---------------------------------------------------------------------- clip (Data)
def _ground_speed(q: np.ndarray, fps: float) -> np.ndarray:
  """Horizontal speed of the root (central differences of x, y)."""
  if len(q) < 2:
    return np.zeros(len(q), np.float32)
  return np.linalg.norm(np.gradient(q[:, 0:2], 1.0 / fps, axis=0), axis=1)


def _clip_columns(clip, model, feet: list[str], sole_z: float) -> list[np.ndarray]:
  from wbc_mjlab.studio.clips import _hinges, foot_heights

  q = clip.qpos
  cols = [q[:, i] for i in range(3)]
  cols += list(_rpy(q[:, 3:7]).T)
  cols.append(_ground_speed(q, clip.fps))
  cols += [q[:, qadr] for _, qadr, *_ in _hinges(model)]
  cols += list(foot_heights(clip, model, feet, sole_z).values())
  return cols


def clip_channels(clip, model, *, feet: list[str], sole_z: float, compare=None) -> dict:
  """Channels of a clip (Data step); ``compare`` (the raw source of a converted clip) is
  resampled to the clip's frames and sent as ``ref``."""
  from wbc_mjlab.studio.clips import _hinges, foot_heights

  limits = hinge_limits(model)
  channels = _root_channels()
  channels.append(_ch("root_speed", "root speed (xy)", "Root speed", "m/s"))
  channels += [_joint_channel(n, limits) for n, *_ in _hinges(model)]
  channels += [
    _ch(f"foot_{n}", n.removesuffix("_link"), "Feet (sole height)", "m")
    for n in foot_heights(clip, model, feet, sole_z)
  ]
  data = np.stack(_clip_columns(clip, model, feet, sole_z), axis=1)
  out = {
    "source": "clip",
    "title": clip.name,
    "fps": float(clip.fps),
    "frames": int(clip.frames),
    "start": 0,
    "channels": channels,
    "data": _b64(data),
    "ref": None,
    "ref_label": "",
  }
  if compare is not None:
    ref = np.stack(_clip_columns(compare, model, feet, sole_z), axis=1)
    if ref.shape[1] == data.shape[1]:
      t, tc = np.arange(clip.frames) / clip.fps, np.arange(compare.frames) / compare.fps
      ref = np.stack([np.interp(t, tc, ref[:, j]) for j in range(ref.shape[1])], 1)
      ref[t > tc[-1] + 1e-6] = np.nan  # beyond the source: no comparison
      out["ref"], out["ref_label"] = _b64(ref), "raw source"
  return out


# ---------------------------------------------------------------------- Play
MAX_TRACKS = (
  32  # envs recorded at once (all of a small session; else the recently focused)
)
TRACK_BUDGET = 128 * 2**20  # bytes of GPU memory the tracks may take


class PlayRecorder:
  """Rolling record of a Play session, several envs at once (each a *track*): the robot
  (``data``) next to the reference it tracks (``ref``, the ``ref_cols`` channels only), plus
  per-row meta columns (clip, frame in the clip, episode step) the client uses to mark episode
  and clip changes.

  A session of up to :data:`MAX_TRACKS` envs (memory permitting) records every env from the
  start, so switching the focused env shows its history at once; a larger one records the
  most recently focused envs (an env starts recording when first focused or fetched). One
  vectorized write per step for all tracks; nothing leaves the GPU until fetched."""

  META = ("clip", "clip_frame", "episode_step")

  def __init__(self, session, model, *, fps: float, horizon_s: float = HISTORY_S):
    self.s = session
    self.fps = float(fps)
    self.horizon = max(2, int(round(horizon_s * fps)))
    self.count = (
      0  # rows recorded (all tracks share the clock; row k sits at k % horizon)
    )
    self._lock = threading.Lock()
    rm = session.env.reward_manager
    self.reward_names = list(rm.active_terms)
    self._weights = torch.tensor(
      [rm.get_term_cfg(n).weight for n in self.reward_names],
      device=session.device,
      dtype=torch.float,
    )
    self._build_channels(model)
    c, r, m = len(self.channels), len(self.ref_cols), len(self.META)
    per_env = self.horizon * (c + r + m) * 4
    k = max(1, min(session.num_envs, MAX_TRACKS, TRACK_BUDGET // per_env))
    dev = session.device
    self._data = torch.zeros(self.horizon, k, c, device=dev)
    self._ref = torch.full((self.horizon, k, r), float("nan"), device=dev)
    self._meta = torch.zeros(self.horizon, k, m, device=dev)
    self._envs = torch.arange(k, device=dev)  # env of each track
    self._track_env = list(range(k))
    self._begin = [0] * k  # first row of each track's record
    self._epoch = list(range(1, k + 1))  # changes when a track starts over
    self._next_epoch = k + 1
    self._used = [0.0] * k  # last focus / fetch (which track a new env takes over)
    self._has_pending = torch.zeros(k, dtype=torch.bool, device=dev)
    self._pending = (  # the reference each track's next row is compared to
      torch.zeros(k, r, device=dev),
      torch.zeros(k, len(self.slices_body), 3, device=dev),
    )

  @property
  def num_tracks(self) -> int:
    return len(self._track_env)

  # -------------------------------------------------------------- layout
  def _build_channels(self, model) -> None:
    s = self.s
    joints = list(s.robot.joint_names)
    limits = hinge_limits(model)
    am = s.env.action_manager
    action_names: list[str] = []
    for name in am.active_terms:
      term = am.get_term(name)
      names = list(getattr(term, "target_names", None) or [])
      action_names += (
        names
        if len(names) == term.action_dim
        else [f"{name}[{i}]" for i in range(term.action_dim)]
      )
    bodies = [b.removesuffix("_link") for b in s.cmd.cfg.body_names]
    chans: list[dict] = []
    slices: dict[str, slice] = {}

    def add(key: str, items: list[dict]) -> None:
      slices[key] = slice(len(chans), len(chans) + len(items))
      chans.extend(items)

    add("root", _root_channels(wrap=True))
    add("joint_pos", [_joint_channel(n, limits) for n in joints])
    add(
      "joint_target",
      [_joint_channel(n, limits, "PD targets", "target:") for n in joints],
    )
    efforts = effort_limits(s)

    def torque(n: str) -> dict:
      lo, hi, src = efforts.get(n, (None, None, ""))
      return _ch(
        f"tau:{n}", n, "Joint torques", "Nm", lower=lo, upper=hi, limit_note=src
      )

    # the motor torque (what the effort limits cap); actuators with a friction model (Unitree)
    # take friction off it: the joint gets motor - friction (qfrc_actuator)
    self._friction = [
      (act.target_ids, act)
      for act in s.robot.actuators
      if getattr(act, "friction", False) is not False
    ]
    add("torque", [torque(n) for n in joints])
    if self._friction:
      add(
        "friction",
        [_ch(f"fric:{n}", n, "Actuator friction", "Nm") for n in joints],
      )
    add(
      "action",
      [_ch(f"act:{n}", n, "Actions (policy output)", "") for n in action_names],
    )
    add(
      "body_err",
      [_ch(f"err:{b}", b, "Tracking error (keybody position)", "m") for b in bodies],
    )
    add(
      "reward",
      [_ch(f"rew:{n}", n, "Rewards (weighted)", "reward") for n in self.reward_names]
      + [_ch("rew:total", "total", "Rewards (weighted)", "reward")],
    )
    add(
      "reward_raw",
      [
        _ch(f"raw:{n}", n, "Rewards (raw, before weight)", "raw")
        for n in self.reward_names
      ],
    )
    self.channels = chans
    self.slices = slices
    self.slices_body = range(len(bodies))
    # channels with a reference: root and joints, the first columns (the ref is stored for
    # those only)
    self.ref_cols = list(range(slices["joint_pos"].stop))
    self._nan_raw = self._weights == 0.0

  # -------------------------------------------------------------- tracks
  def _track(self, env: int) -> int:
    """The track recording ``env``; a new env takes over the least recently used track
    (under the lock)."""
    env = int(env)
    if env in self._track_env:
      k = self._track_env.index(env)
    else:
      k = min(range(self.num_tracks), key=self._used.__getitem__)
      self._track_env[k] = env
      self._envs[k] = env
      self._restart(k)
    self._used[k] = time.monotonic()
    return k

  def _restart(self, k: int) -> None:
    self._begin[k] = self.count
    self._epoch[k] = self._next_epoch
    self._next_epoch += 1
    self._has_pending[k] = False

  def follow(self, env: int) -> None:
    """Keep ``env`` (the focused one) recorded."""
    if 0 <= int(env) < self.s.num_envs:
      with self._lock:
        self._track(env)

  def clear(self) -> None:
    """Drop every record (restart): the client starts over."""
    with self._lock:
      for k in range(self.num_tracks):
        self._restart(k)

  # -------------------------------------------------------------- recording
  @torch.no_grad()
  def record(self) -> None:
    """One row for every track (call after each env step)."""
    with self._lock:
      self._record()

  def _record(self) -> None:
    s, e = self.s, self._envs
    d, cmd = s.robot.data, s.cmd
    origin = s.env.scene.env_origins[e]
    sl = self.slices
    row = self._data[self.count % self.horizon]
    ref = self._ref[self.count % self.horizon]
    row[:, sl["root"]] = torch.cat(
      (d.root_link_pos_w[e] - origin, _rpy_t(d.root_link_quat_w[e])), dim=-1
    )
    row[:, sl["joint_pos"]] = d.joint_pos[e]
    row[:, sl["joint_target"]] = d.joint_pos_target[e]
    if self._friction:
      fric = torch.zeros_like(d.qfrc_actuator[e])
      for ids, act in self._friction:
        if act.friction is not None:
          fric[:, ids] = act.friction[e]
      row[:, sl["torque"]] = d.qfrc_actuator[e] + fric
      row[:, sl["friction"]] = fric
    else:
      row[:, sl["torque"]] = d.qfrc_actuator[e]
    row[:, sl["action"]] = s.env.action_manager.action[e]
    # the reference this step's rewards compared to: set at the end of the previous step
    # (none on a track's first row)
    has = self._has_pending[:, None]
    ref_prev, bodies_prev = self._pending
    ref.copy_(torch.where(has, ref_prev, float("nan")))
    err = torch.norm(bodies_prev - cmd.robot_body_pos_w[e], dim=-1)
    row[:, sl["body_err"]] = torch.where(has, err, float("nan"))
    # the frame the next step's rewards compare against (the command just advanced)
    self._pending = (
      torch.cat(
        (
          cmd.body_pos_w[e, 0] - origin,
          _rpy_t(cmd.body_quat_w[e, 0]),
          cmd.joint_pos[e],
        ),
        dim=-1,
      ),
      cmd.body_pos_relative_w[e],
    )
    self._has_pending.fill_(True)
    step = s.env.reward_manager._step_reward[e]  # noqa: SLF001 — weighted, per step
    row[:, sl["reward"]] = torch.cat((step, step.sum(-1, keepdim=True)), dim=-1)
    raw = step / torch.where(self._nan_raw, 1.0, self._weights)
    row[:, sl["reward_raw"]] = torch.where(self._nan_raw, float("nan"), raw)
    tid = cmd.trajectory_ids[e]
    self._meta[self.count % self.horizon] = torch.stack(
      (
        tid.float(),
        (cmd.time_steps[e] - cmd.motion.segment_start_idx[tid]).float(),
        s.env.episode_length_buf[e].float(),
      ),
      dim=-1,
    )
    self.count += 1

  # -------------------------------------------------------------- fetch
  def fetch(self, env: int = 0, since: int = 0, epoch: int | None = None) -> dict:
    """Rows ``since``… of ``env``'s record as (rows, columns) float32 arrays (``data``,
    ``ref`` for the ``ref_cols`` channels, ``meta``); all of it, with the channel list, when
    ``epoch`` is not the record's (another env, a restart: the client starts over). An env
    not recorded yet starts recording now."""
    env = min(max(0, int(env)), self.s.num_envs - 1)
    with self._lock:
      k = self._track(env)
      fresh = epoch != self._epoch[k]
      hi = self.count
      lo = max(self._begin[k], hi - self.horizon, 0 if fresh else min(int(since), hi))
      idx = torch.arange(lo, hi, device=self.s.device) % self.horizon
      parts = (self._data[idx, k], self._ref[idx, k], self._meta[idx, k])
      out = {
        "source": "play",
        "epoch": self._epoch[k],
        "env": env,
        "fps": self.fps,
        "start": lo,
        "end": hi,
        "fresh": fresh,
      }
    out["data"], out["ref"], out["meta"] = (p.cpu().numpy() for p in parts)
    if fresh:
      out["channels"] = self.channels
      out["ref_cols"] = self.ref_cols
      out["clips"] = self.s.clip_names()
      out["meta_columns"] = list(self.META)
      out["ref_label"] = "reference"
      out["history_s"] = self.horizon / self.fps
    return out


def encode_rows(out: dict) -> bytes:
  """A :meth:`PlayRecorder.fetch` result as ``[u32 header length][header JSON][pad to 4]``
  then the float32 ``data``, ``ref`` and ``meta`` rows (little-endian); the header lists
  their column counts (``cols``)."""
  arrays = [np.ascontiguousarray(out[k], dtype="<f4") for k in ("data", "ref", "meta")]
  head = {k: v for k, v in out.items() if k not in ("data", "ref", "meta")}
  head["cols"] = [int(a.shape[1]) for a in arrays]
  text = json.dumps(head).encode()
  pad = (-(4 + len(text))) % 4
  return b"".join(
    [
      struct.pack("<I", len(text) + pad),
      text,
      b" " * pad,
      *(a.tobytes() for a in arrays),
    ]
  )


__all__ = [
  "PlayRecorder",
  "clip_channels",
  "effort_limits",
  "encode_rows",
  "hinge_limits",
  "joint_group",
]
