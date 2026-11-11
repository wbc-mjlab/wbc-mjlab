"""BlinkMimic task choices on the shared Lookahead task.

SuperMimic: https://arxiv.org/abs/2610.28480.
BlinkMimic (Chen, 2026), d264658, mdp.py/motion.py (MIT; THIRD_PARTY_NOTICES):
https://github.com/fuyichen0725/blinkmimic/tree/d264658d6c9ce61fb52f9f0f8985a369f3ccc409
Training method choices live in the wm-blink recipe, independently of this task.
"""

from mjlab.managers.reward_manager import RewardTermCfg

from wbc_mjlab.env.mdp.rewards import reference_contact_reward
from wbc_mjlab.presets.lookahead import apply_lookahead


def apply_blink(cfg, *, foot_bodies, root_body):
  apply_lookahead(cfg)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 512)
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch or 0, 128)
  cfg.commands["motion"].anchor_body_name = root_body
  cfg.events["base_com"].params["asset_cfg"].body_names = (root_body,)
  cfg.rewards["reference_contact"] = RewardTermCfg(
    func=reference_contact_reward,
    weight=0.25,
    params={"command_name": "motion", "body_names": tuple(foot_bodies)},
  )
