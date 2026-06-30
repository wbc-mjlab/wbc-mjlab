"""State-estimation actor observation preset."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.env.se_actor_obs import configure_state_estimation_actor_obs


def apply_se_actor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Drop height/gravity proxies; add anchor pose error + base lin vel."""
  configure_state_estimation_actor_obs(cfg)
