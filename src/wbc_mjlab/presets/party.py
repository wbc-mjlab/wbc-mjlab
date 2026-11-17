"""PartyMimic task choices on the BlinkMimic task.

PartyMimic (Roboparty, 2026), MIT; see THIRD_PARTY_NOTICES:
https://github.com/Roboparty/PartyMimic (train.py ``--std_scale``, ``--div_k``,
``--interface sparse5``).
Training method choices live in the wm-party recipe, independently of this task.
"""

from wbc_mjlab.presets.blink import apply_blink

POSE_KERNELS = (
  "motion_body_pos",
  "motion_body_ori",
  "motion_global_root_pos",
  "motion_global_root_ori",
)
"""Kernels PartyMimic's ``--std_scale`` widens or narrows; velocity kernels keep theirs."""


def apply_party(
  cfg,
  *,
  foot_bodies,
  root_body,
  kernel_scale: float = 0.6,
  diversity_clusters: int = 24,
):
  """Blink with narrower pose kernels and diversity-balanced uniform resets.

  ``kernel_scale`` multiplies the pose kernel widths (0.6: PartyMimic's default; 0.35
  tracks tighter but slips more under soft contact). ``diversity_clusters`` > 0 spreads
  the uniform share of episode starts evenly over k-means clusters of 1 s reference bins
  instead of by duration (0: by duration).
  """
  if kernel_scale <= 0:
    raise ValueError("kernel_scale must be positive")
  apply_blink(cfg, foot_bodies=foot_bodies, root_body=root_body)
  for name in POSE_KERNELS:
    params = cfg.rewards[name].params
    params["std"] = params["std"] * kernel_scale
  rsi = cfg.commands["motion"].rsi
  rsi.diversity_clusters = diversity_clusters


def apply_sparse_keypoints(cfg, *, keypoints, error_bodies=None):
  """The policy follows keypoint poses instead of joint references (PartyMimic
  ``--interface sparse5``), as a teleoperation or video-capture source provides them.

  The actor's joint reference and anchor shift give way to ``ref_keypoints`` (the
  ``keypoints`` over the lookahead frames, in the reference's own heading frame) and
  ``keypoint_error`` (robot against reference on ``error_bodies``, default every keypoint
  but the first). The reference anchor's orientation against the robot stays. The critic
  keeps its privileged joint references; rewards are unchanged.
  """
  from mjlab.managers.observation_manager import ObservationTermCfg
  from mjlab.utils.noise import UniformNoiseCfg as Unoise

  from wbc_mjlab.env.mdp.observations import keypoint_error, ref_keypoints
  from wbc_mjlab.presets.lookahead import FUTURE_FRAMES

  keypoints = tuple(keypoints)
  error_bodies = tuple(error_bodies) if error_bodies is not None else keypoints[1:]
  actor = cfg.observations["actor"]
  for name in ("ref_joint_pos", "ref_joint_vel", "ref_anchor_shift_b"):
    actor.terms.pop(name, None)
  actor.terms = {
    "ref_keypoints": ObservationTermCfg(
      func=ref_keypoints,
      params={
        "command_name": "motion",
        "body_names": keypoints,
        "frames": FUTURE_FRAMES,
      },
    ),
    **actor.terms,
    "keypoint_error": ObservationTermCfg(
      func=keypoint_error,
      params={"command_name": "motion", "body_names": error_bodies},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
  }
