"""Success-guided sampling (SGS) preset: adaptive RSI aimed at a target success rate.

Zhang et al., "A Balanced Data Diet" (arXiv:2610.12465). Instead of starting episodes
on the hardest clip bins, SGS favours bins the policy succeeds on about
``target_success`` of the time: mastered bins and currently impossible ones (bad
retargets, flips it can't do yet) both get few starts.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.env.mdp.commands import MotionCommandCfg

# RsiCfg fields SGS sets (shared with the Studio's built-in ``sgs`` module).
# alpha 0.01 ≈ the paper's success window of the last 100 episodes per bin.
SGS_RSI_FIELDS: dict[str, object] = {
  "strategy": "success_target",
  "alpha": 0.01,
  "target_success": 0.66,
  "target_concentration": 5.0,
  "target_temperature": 2.0,
}


def apply_sgs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Switch adaptive RSI to success-guided sampling; everything else is unchanged."""
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.rsi.sampling_mode = "adaptive"
  for name, value in SGS_RSI_FIELDS.items():
    setattr(motion_cmd.rsi, name, value)
