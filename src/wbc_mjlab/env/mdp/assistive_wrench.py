"""WBC Section S6: model-based assistive wrench curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_box_minus

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass
class AssistiveWrenchNominal:
  """Whole-body nominal dynamics used by the virtual wrench (Eq. 13)."""

  mass: float
  inertia: torch.Tensor
  gravity: torch.Tensor


class AssistiveWrenchEvent:
  """Apply a bin-coupled virtual spatial wrench at the anchor body each step."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._env = env
    self._nominal: AssistiveWrenchNominal | None = None
    self._anchor_body_index: int | None = None

  def _ensure_nominal(
    self,
    asset: Entity,
    body_name: str,
    device: torch.device,
  ) -> None:
    if self._nominal is not None:
      return

    body_index = asset.body_names.index(body_name)
    self._anchor_body_index = body_index
    model = self._env.sim.mj_model
    body_id = int(asset.indexing.body_ids[body_index].item())

    mass = float(model.body_subtreemass[body_id])
    inertia_diag = torch.tensor(
      model.body_inertia[body_id, :3], dtype=torch.float32, device=device
    )
    inertia = torch.diag(inertia_diag)
    gravity = torch.tensor(model.opt.gravity, dtype=torch.float32, device=device)
    self._nominal = AssistiveWrenchNominal(mass=mass, inertia=inertia, gravity=gravity)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    body_name: str = "torso_link",
    kvp: float = 0.0,
    kvd: float = 10.0,
    kwp: float = 200.0,
    kwd: float = 10.0,
    enabled: bool = True,
  ) -> None:
    del env_ids
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset: Entity = env.scene[asset_cfg.name]
    self._ensure_nominal(asset, body_name, env.device)
    assert self._nominal is not None
    assert self._anchor_body_index is not None

    num_envs = env.num_envs
    beta = command.episode_assist_gain
    forces = torch.zeros(num_envs, 1, 3, device=env.device)
    torques = torch.zeros(num_envs, 1, 3, device=env.device)

    if not enabled or beta.max() <= 0.0:
      command.assist_force_w.zero_()
      command.assist_torque_w.zero_()
      asset.write_external_wrench_to_sim(
        forces,
        torques,
        body_ids=[self._anchor_body_index],
      )
      return

    ref_pos = command.anchor_pos_w
    ref_quat = command.anchor_quat_w
    ref_lin_vel = command.anchor_lin_vel_w
    ref_ang_vel = command.anchor_ang_vel_w
    ref_lin_acc = command.anchor_lin_acc_w
    ref_ang_acc = command.anchor_ang_acc_w

    pos = asset.data.body_link_pos_w[:, self._anchor_body_index]
    quat = asset.data.body_link_quat_w[:, self._anchor_body_index]
    lin_vel = asset.data.body_link_lin_vel_w[:, self._anchor_body_index]
    ang_vel = asset.data.body_link_ang_vel_w[:, self._anchor_body_index]

    pos_err = ref_pos - pos
    lin_vel_err = ref_lin_vel - lin_vel
    ori_err = quat_box_minus(ref_quat, quat)
    ang_vel_err = ref_ang_vel - ang_vel

    nominal = self._nominal
    M = nominal.mass
    I = nominal.inertia
    g = nominal.gravity

    # Eq. 13a: F_b = M v̇̂ + kvp(p̂−p) + kvd(v̂−v) − g
    f_nominal = M * ref_lin_acc + kvp * pos_err + kvd * lin_vel_err - g

    # Eq. 13b: M_b = I ω̇̂ + kp Φ̂⊟Φ + kd(ω̂−ω) + ω×(Iω) − r×Mg
    root_pos = asset.data.root_link_pos_w
    subtree_com = asset.data.data.subtree_com[:, asset.indexing.root_body_id]
    rb_com = subtree_com - root_pos
    mg = M * g
    rb_cross_mg = torch.cross(rb_com, mg.expand(num_envs, 3), dim=-1)

    i_omega = torch.einsum("ij,nj->ni", I, ang_vel)
    omega_cross_i_omega = torch.cross(ang_vel, i_omega, dim=-1)
    i_ang_acc = torch.einsum("ij,nj->ni", I, ref_ang_acc)
    i_ori_err = torch.einsum("ij,nj->ni", I, ori_err)
    i_ang_vel_err = torch.einsum("ij,nj->ni", I, ang_vel_err)

    m_nominal = (
      i_ang_acc
      + kwp * i_ori_err
      + kwd * i_ang_vel_err
      + omega_cross_i_omega
      - rb_cross_mg
    )

    forces[:, 0] = beta.unsqueeze(-1) * f_nominal
    torques[:, 0] = beta.unsqueeze(-1) * m_nominal
    command.assist_force_w[:] = forces[:, 0]
    command.assist_torque_w[:] = torques[:, 0]

    asset.write_external_wrench_to_sim(
      forces,
      torques,
      body_ids=[self._anchor_body_index],
    )


def assistive_wrench_force(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.assist_force_w


def assistive_wrench_torque(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.assist_torque_w


def assistive_wrench_gain(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.episode_assist_gain.unsqueeze(-1)
