"""The env's observation and reward terms, evaluated on imagined states.

Observation and reward terms read the robot through ``env.scene[...]``, ``robot.data`` and
:class:`~wbc_mjlab.env.mdp.commands.TrackingView`, and the policy's actions through
``env.action_manager``. :class:`ImaginedEnv` provides exactly those objects, built from a
:class:`RobotState`, its forward kinematics and a reference frame, so the env's own term
functions run unchanged (and differentiably) on world-model states. Any attribute it does
not provide raises :class:`NotInImagination`: such a reward term is learned instead
(:class:`Imagination`), such an observation term is an error.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import mujoco
import torch
from mjlab.envs.mdp import builtin_sensor, dr
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.env.mdp.actions import ReferenceJointPositionAction
from wbc_mjlab.env.mdp.commands import MotionCommand, TrackingView
from wbc_mjlab.motion.library import MotionFrame
from wbc_mjlab.rl.wm.kinematics import BodyKinematics, RobotKinematics, RobotState

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_term_config import ManagerTermBaseCfg


class NotInImagination(Exception):
  """A term needs something an imagined state does not have (contacts, actuator forces, …)."""


class _Strict:
  """Raises :class:`NotInImagination` for any attribute it does not define."""

  _name = "env"

  def __getattr__(self, attr: str) -> Any:
    raise NotInImagination(f"{self._name}.{attr}")


class ImaginedRobotData(_Strict):
  """``robot.data`` of imagined states: joint and body kinematics, the env's constants."""

  _name = "robot.data"

  def __init__(
    self, s: RobotState, bodies: BodyKinematics, const: dict[str, torch.Tensor]
  ):
    self.joint_pos, self.joint_vel = s.joint_pos, s.joint_vel
    self.root_link_pos_w, self.root_link_quat_w = s.root_pos, s.root_quat
    self.root_link_lin_vel_w, self.root_link_ang_vel_w = s.root_lin_vel, s.root_ang_vel
    self.body_link_pos_w, self.body_link_quat_w = bodies.pos, bodies.quat
    self.body_link_lin_vel_w, self.body_link_ang_vel_w = bodies.lin_vel, bodies.ang_vel
    for name, value in const.items():
      setattr(self, name, value)

  @cached_property
  def joint_pos_biased(self) -> torch.Tensor:
    return self.joint_pos + self.encoder_bias

  @cached_property
  def projected_gravity_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.root_link_quat_w, self.gravity_vec_w)

  @cached_property
  def root_link_lin_vel_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

  @cached_property
  def root_link_ang_vel_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)


class _Robot(_Strict):
  _name = "robot"

  def __init__(self, data: ImaginedRobotData, body_names, joint_names):
    self.data, self.body_names, self.joint_names = data, body_names, joint_names


class _Scene(_Strict):
  _name = "env.scene"

  def __init__(self, entity_name: str, robot: _Robot, num_envs: int, device):
    self._entity_name, self._robot = entity_name, robot
    self.env_origins = torch.zeros(num_envs, 3, device=device)

  def __getitem__(self, name: str) -> _Robot:
    if name != self._entity_name:
      raise NotInImagination(f"env.scene[{name!r}]")
    return self._robot


@dataclass
class Goal:
  """Where the reference is, for a batch of envs."""

  frame: MotionFrame  # at the current step
  window: (
    MotionFrame | None
  )  # the current frame and the next ones, fields (n, frames, …)
  time_step: torch.Tensor
  trajectory: torch.Tensor
  reference_offset: torch.Tensor


class ImaginedCommand(TrackingView, _Strict):
  """The motion command's tracking quantities for an imagined state and its goal."""

  _name = "motion command"

  def __init__(self, real: MotionCommand, robot: _Robot, env: ImaginedEnv, goal: Goal):
    self.cfg, self.robot, self._env, self._goal = real.cfg, robot, env, goal
    self.motion = real.motion
    self.robot_anchor_body_index = real.robot_anchor_body_index
    self.motion_anchor_body_index = real.motion_anchor_body_index
    self.body_indexes = real.body_indexes
    self._tracked_joint_ids = real._tracked_joint_ids
    self.reference_offset_w = goal.reference_offset
    self.time_steps, self.trajectory_ids = goal.time_step, goal.trajectory
    self.update_relative_body_poses()

  def reference(self) -> MotionFrame:
    return self._goal.frame

  def reference_window(self, num_frames: int) -> MotionFrame:
    w = self._goal.window
    if w is None or w.joint_pos.shape[1] < num_frames:
      raise NotInImagination(f"reference_window({num_frames})")
    return w if w.joint_pos.shape[1] == num_frames else w.head(num_frames)


class _Actions(_Strict):
  _name = "env.action_manager"

  def __init__(self, term_name: str, action: torch.Tensor, prev_action: torch.Tensor):
    self._term_name, self.action, self.prev_action = term_name, action, prev_action

  def get_term(self, name: str) -> Any:
    if name != self._term_name:
      raise NotInImagination(f"env.action_manager.get_term({name!r})")
    return _RawAction(self.action)


@dataclass
class _RawAction:
  raw_action: torch.Tensor


class _Commands(_Strict):
  _name = "env.command_manager"

  def __init__(self, name: str, command: ImaginedCommand):
    self._command_name, self._command = name, command

  def get_term(self, name: str) -> ImaginedCommand:
    if name != self._command_name:
      raise NotInImagination(f"env.command_manager.get_term({name!r})")
    return self._command


class _Terminations(_Strict):
  _name = "env.termination_manager"

  def __init__(self, num_envs: int, device):
    # Imagined states are alive; terminations are the world model's soft survival weight.
    self.terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.time_outs = self.terminated


class ImaginedEnv(_Strict):
  """What observation and reward terms read from ``env``, for one imagined step."""

  def __init__(
    self,
    imagination: Imagination,
    s: RobotState,
    bodies: BodyKinematics,
    env_ids: torch.Tensor,
    goal: Goal,
    action: torch.Tensor,
    prev_action: torch.Tensor,
  ):
    im = imagination
    n = s.root_pos.shape[0]
    self.num_envs, self.device, self.step_dt = n, s.root_pos.device, im.step_dt
    self._imagination, self._bodies = im, bodies
    const = {k: v[env_ids] for k, v in im.robot_constants.items()}
    robot = _Robot(ImaginedRobotData(s, bodies, const), im.body_names, im.joint_names)
    self.scene = _Scene(im.entity_name, robot, n, self.device)
    self.action_manager = _Actions(im.action_term_name, action, prev_action)
    self.termination_manager = _Terminations(n, self.device)
    command = ImaginedCommand(im.command, robot, self, goal)
    self.command_manager = _Commands(im.command_name, command)

  def builtin_sensor(self, sensor_name: str) -> torch.Tensor:
    """Gyro / velocimeter on a robot site, in the site frame (MuJoCo's definition)."""
    gyro, body, site_pos, site_quat = self._imagination.site_sensor(sensor_name)
    quat_b, ang = self._bodies.quat[:, body], self._bodies.ang_vel[:, body]
    quat_s = quat_mul(quat_b, site_quat.expand_as(quat_b))
    if gyro:
      return quat_apply_inverse(quat_s, ang)
    offset = quat_apply(quat_b, site_pos.expand(quat_b.shape[0], 3))
    lin = self._bodies.lin_vel[:, body] + torch.cross(ang, offset, dim=-1)
    return quat_apply_inverse(quat_s, lin)


def _builtin_sensor(env: ImaginedEnv, sensor_name: str) -> torch.Tensor:
  return env.builtin_sensor(sensor_name)


# mjlab terms that check a concrete type, with their imagined counterpart.
_TERM_OVERRIDES = {builtin_sensor: _builtin_sensor}


def _frames_ahead(cfg: ManagerTermBaseCfg) -> int:
  """Reference frames an observation term reads, the current one included."""
  params = cfg.params or {}
  return (int(params.get("frames", 1)) - 1) * int(params.get("stride", 1)) + 1


def call_term(cfg: ManagerTermBaseCfg, env: ImaginedEnv) -> torch.Tensor:
  func = _TERM_OVERRIDES.get(cfg.func, cfg.func)
  if not inspect.isfunction(func):  # class terms hold state of the real env
    raise NotInImagination(f"class-based term {type(cfg.func).__name__}")
  return func(env, **cfg.params)


# Domain randomization the world model is conditioned on (sampled once per env).
_CONTEXT_FUNCS = (dr.geom_friction, dr.body_com_offset, dr.encoder_bias)

# Goal-relative features of the critic and of the step heads: anchor error, keybodies of the
# reference and of the robot (positions and linear velocities) in the robot anchor frame.
_TRACKING_FEATURES = (
  mdp.motion_anchor_pos_b,
  mdp.motion_anchor_ori_b,
  mdp.ref_body_pos_b,
  mdp.robot_body_pos_b,
  mdp.ref_body_lin_vel,
  mdp.motion_body_lin_vel,
)


class Imagination:
  """Everything the world-model algorithm needs from the env, read once at construction.

  Raises ``ValueError`` when the env has something imagination cannot follow: an actor
  observation that is not a function of the robot state and the reference, observation
  delay, an unsupported action term, or domain
  randomization outside the context (:meth:`context`).
  """

  def __init__(self, env: ManagerBasedRlEnv, *, actor_group: str = "actor"):
    self.env = env
    self.step_dt = env.step_dt
    command = next(
      (
        (name, term)
        for name in env.command_manager.active_terms
        if isinstance(term := env.command_manager.get_term(name), MotionCommand)
      ),
      None,
    )
    if command is None:
      raise ValueError("world-model training needs a MotionCommand")
    self.command_name, self.command = command
    robot = self.command.robot
    self.robot = robot
    self.entity_name = self.command.cfg.entity_name
    self.body_names, self.joint_names = robot.body_names, robot.joint_names
    self.kinematics = RobotKinematics(
      env.sim.mj_model,
      robot.indexing.body_ids.tolist(),
      robot.indexing.joint_q_adr.tolist(),
      env.device,
    )
    d = robot.data
    self.robot_constants = {
      "default_joint_pos": d.default_joint_pos,
      "default_joint_vel": d.default_joint_vel,
      "soft_joint_pos_limits": d.soft_joint_pos_limits,
      "encoder_bias": d.encoder_bias,
      "gravity_vec_w": d.gravity_vec_w,
    }
    self._sensors: dict[str, tuple] = {}

    self.action_term_name, self._action_term = self._joint_action()
    self.num_actions = env.action_manager.total_action_dim

    om = env.observation_manager
    # Own copies of the noise configs: theirs cache tensors made during env steps
    # (inference mode), which autograd cannot save.
    self._obs_terms = [
      (name, cfg, cfg.noise and dataclasses.replace(cfg.noise))
      for name, cfg in zip(
        om._group_obs_term_names[actor_group],
        om._group_obs_term_cfgs[actor_group],
        strict=True,
      )
    ]
    if any(cfg.delay_max_lag > 0 for _, cfg, _ in self._obs_terms):
      raise ValueError("actor observations: no observation delay")
    # The history kept is the longest; a term with a shorter one reads its last steps.
    self._history_terms = {
      name: cfg.history_length
      for name, cfg, _ in self._obs_terms
      if cfg.history_length > 1
    }
    self.history_length = max(self._history_terms.values(), default=1)
    # Reference frames the actor looks ahead (``frames`` / ``stride`` of the reference
    # terms), the current one included.
    self.window_frames = max(
      [_frames_ahead(cfg) for _, cfg, _ in self._obs_terms] or [1]
    )

    rm = env.reward_manager
    self.reward_scale = env.step_dt if rm._scale_by_dt else 1.0
    active = [
      (i, n, c)
      for i, (n, c) in enumerate(zip(rm._term_names, rm._term_cfgs, strict=True))
      if c.weight != 0.0
    ]
    probe = self.view(*self._probe())
    for name, cfg, _ in self._obs_terms:
      try:
        call_term(cfg, probe)
      except NotInImagination as e:
        raise ValueError(
          f"actor observation {name!r} needs {e}, not in imagined states"
        ) from e
    self.analytic_rewards: list[tuple[str, ManagerTermBaseCfg]] = []
    # (reward manager index, name, what it needs that imagined states lack)
    self.learned_rewards: list[tuple[int, str, str]] = []
    for i, name, cfg in active:
      try:
        call_term(cfg, probe)
        self.analytic_rewards.append((name, cfg))
      except NotInImagination as e:
        self.learned_rewards.append((i, name, str(e)))
    self.context_names, self._context = self._read_context()
    self.history_dim = self.env_history().shape[-1]

  def summary(self) -> str:
    learned = ", ".join(f"{n} (needs {why})" for _, n, why in self.learned_rewards)
    return (
      f"exact rewards: {', '.join(n for n, _ in self.analytic_rewards)}\n"
      f"learned rewards: {learned or 'none'}\n"
      f"context: {', '.join(self.context_names) or 'none'}"
    )

  # --- construction helpers ---

  def _joint_action(
    self,
  ) -> tuple[str, JointPositionAction | ReferenceJointPositionAction]:
    am = self.env.action_manager
    if len(am.active_terms) != 1:
      raise ValueError("world-model training supports one joint-position action term")
    name = am.active_terms[0]
    term = am.get_term(name)
    if not isinstance(term, (JointPositionAction, ReferenceJointPositionAction)):
      raise ValueError(f"action term {type(term).__name__} is not supported")
    if list(term._target_ids.tolist()) != list(range(len(self.joint_names))):
      raise ValueError("the action must drive every robot joint, in order")
    return name, term

  def _read_context(self) -> tuple[list[str], torch.Tensor]:
    """Per-env domain-randomization values (sampled once, at startup)."""
    env, idx = self.env, self.robot.indexing
    names, cols = [], []
    for mode in env.event_manager.available_modes:
      for name, cfg in zip(
        env.event_manager._mode_term_names[mode],
        env.event_manager._mode_term_cfgs[mode],
        strict=True,
      ):
        func = cfg.func
        if mode in ("interval", "step") and cfg.params.get("enabled", True):
          raise ValueError(
            f"event {name!r} ({mode}) acts on the robot during episodes: the world model "
            "cannot predict it (see wbc_mjlab.rl.wm.config.wm_env_cfg)"
          )
        if not getattr(func, "__module__", "").startswith(dr.__name__):
          continue  # not domain randomization
        if func not in _CONTEXT_FUNCS or mode != "startup":
          raise ValueError(
            f"domain randomization {name!r} ({mode}): the world model's context holds "
            "startup geom_friction, body_com_offset and encoder_bias only"
          )
        if func is dr.geom_friction:
          geoms = idx.geom_ids[cfg.params["asset_cfg"].geom_ids]
          axes = cfg.params.get("axes") or [0]
          value = env.sim.model.geom_friction[:, geoms][..., axes].reshape(
            env.num_envs, -1
          )
          if cfg.params.get("shared_random"):
            value = value[:, : len(axes)]
        elif func is dr.body_com_offset:
          bodies = idx.body_ids[cfg.params["asset_cfg"].body_ids]
          value = env.sim.model.body_ipos[:, bodies].reshape(env.num_envs, -1)
        else:
          value = self.robot.data.encoder_bias
        names.append(name)
        cols.append(value.float())
    if not cols:
      return [], torch.zeros(env.num_envs, 0, device=env.device)
    return names, torch.cat(cols, dim=-1).clone()

  def goal(self) -> Goal:
    """The env's current goal (window included when the actor looks ahead)."""
    c = self.command
    window = c.reference_window(self.window_frames) if self.window_frames > 1 else None
    return Goal(
      c.reference(), window, c.time_steps, c.trajectory_ids, c.reference_offset_w
    )

  def _probe(self):
    """Arguments of :meth:`view` for the env's current state (term classification, tests)."""
    s = self.robot_state()
    ids = torch.arange(self.env.num_envs, device=self.env.device)
    am = self.env.action_manager
    return (s, self.kinematics.bodies(s), ids, self.goal(), am.action, am.prev_action)

  def env_history(self) -> torch.Tensor:
    """The env's observation history of the history terms, ``(N, H, history_dim)``,
    oldest first (``(N, 1, 0)`` without history)."""
    buffers = self.env.observation_manager._group_obs_term_history_buffer["actor"]
    parts = []
    for name in self._history_terms:
      part = buffers[name].buffer.flatten(2)
      pad = (
        self.history_length - part.shape[1]
      )  # a shorter history: unread oldest steps
      parts.append(torch.cat([part[:, :1].expand(-1, pad, -1), part], dim=1))
    if not parts:
      return torch.zeros(self.env.num_envs, 1, 0, device=self.env.device)
    return torch.cat(parts, dim=-1)

  # --- per step ---

  def context(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self._context[env_ids]

  def robot_state(self) -> RobotState:
    """The simulator's robot state in the env frame (env origin subtracted)."""
    d = self.robot.data
    return RobotState(
      d.root_link_pos_w - self.env.scene.env_origins,
      d.root_link_quat_w,
      d.root_link_lin_vel_w,
      d.root_link_ang_vel_w,
      d.joint_pos,
      d.joint_vel,
    )

  def view(self, s, bodies, env_ids, goal: Goal, action, prev_action) -> ImaginedEnv:
    return ImaginedEnv(self, s, bodies, env_ids, goal, action, prev_action)

  def site_sensor(
    self, sensor_name: str
  ) -> tuple[bool, int, torch.Tensor, torch.Tensor]:
    """(is gyro, else velocimeter; robot body; site pos and quat in the body)."""
    if sensor_name not in self._sensors:
      m = self.env.sim.mj_model
      sensor = m.sensor(sensor_name)
      kind = int(sensor.type[0])
      if kind not in (
        mujoco.mjtSensor.mjSENS_GYRO,
        mujoco.mjtSensor.mjSENS_VELOCIMETER,
      ):
        raise NotInImagination(f"sensor {sensor_name!r} (type {kind})")
      body, pos, quat = self.kinematics.site_frame(int(sensor.objid[0]))
      self._sensors[sensor_name] = (
        kind == mujoco.mjtSensor.mjSENS_GYRO,
        body,
        pos,
        quat,
      )
    return self._sensors[sensor_name]

  def actor_obs(
    self, env: ImaginedEnv, past: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """The actor observation group (noise as configured, clip, scale) and the history.

    ``past``: the history terms' previous ``H - 1`` values, ``(n, H - 1, history_dim)``
    oldest first. Returns the observation and the history including this step's values.
    """
    values: dict[str, torch.Tensor] = {}
    for name, cfg, noise in self._obs_terms:
      obs = call_term(cfg, env)
      if noise is not None:
        obs = noise.apply(obs)
      if cfg.clip:
        obs = obs.clamp(min=cfg.clip[0], max=cfg.clip[1])
      if cfg.scale is not None:
        obs = obs * cfg.scale
      values[name] = obs.reshape(env.num_envs, -1)
    newest = [values[name] for name in self._history_terms]
    if not newest:
      return torch.cat(list(values.values()), dim=-1), torch.zeros(
        env.num_envs, 1, 0, device=env.device
      )
    assert past is not None, "the history terms need their past values"
    history = torch.cat([past, torch.cat(newest, dim=-1).unsqueeze(1)], dim=1)
    # mjlab's layout: per term, its history flattened oldest first.
    start = 0
    for name, length in self._history_terms.items():
      width = values[name].shape[-1]
      values[name] = history[:, -length:, start : start + width].flatten(1)
      start += width
    return torch.cat(list(values.values()), dim=-1), history

  def analytic_reward(self, env: ImaginedEnv) -> torch.Tensor:
    """Sum of the reward terms that are functions of the imagined state (× dt)."""
    total = torch.zeros(env.num_envs, device=env.device)
    for _, cfg in self.analytic_rewards:
      total = total + torch.nan_to_num(call_term(cfg, env)) * cfg.weight
    return total * self.reward_scale

  def learned_reward_targets(self) -> torch.Tensor:
    """The env's last step values of the learned reward terms (weighted, before × dt)."""
    idx = [i for i, _, _ in self.learned_rewards]
    return self.env.reward_manager._step_reward[:, idx]

  def tracking_features(self, env: ImaginedEnv) -> torch.Tensor:
    return torch.cat([f(env, self.command_name) for f in _TRACKING_FEATURES], dim=-1)

  def joint_setpoint(
    self, action: torch.Tensor, ref: MotionFrame, env_ids: torch.Tensor
  ):
    """PD position targets of a raw action (the action term's processing, encoder bias)."""
    from wbc_mjlab.env.mdp.actions import joint_position_target

    reference = ref.joint_pos
    if isinstance(self._action_term, ReferenceJointPositionAction):
      tracked = self._action_term._command._tracked_joint_ids
      if tracked is not None:
        reference = reference[:, tracked]
    return joint_position_target(self._action_term, action, reference, env_ids)
