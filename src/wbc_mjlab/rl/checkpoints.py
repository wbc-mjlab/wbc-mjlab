"""One checkpoint reader for inference and training.

Compatibility conventions follow mjlab.rl.runner.MjlabOnPolicyRunner.load:
https://github.com/mujocolab/mjlab/blob/main/src/mjlab/rl/runner.py
Legacy rsl_rl actor/critic and distribution names are normalized before loading.
"""

import torch


def read_checkpoint(path, device=None):
  checkpoint = torch.load(path, map_location=device, weights_only=False)
  if "model_state_dict" in checkpoint:
    old = checkpoint.pop("model_state_dict")
    for role in ("actor", "critic"):
      state = {}
      for prefix, replacement in (
        (role + ".", "mlp."),
        (role + "_obs_normalizer.", "obs_normalizer."),
      ):
        state.update(
          {
            replacement + k[len(prefix) :]: v
            for k, v in old.items()
            if k.startswith(prefix)
          }
        )
      if role == "actor":
        state.update({k: old[k] for k in ("std", "log_std") if k in old})
      checkpoint[role + "_state_dict"] = state
  actor = checkpoint.get("actor_state_dict", {})
  for old, new in (
    ("std", "distribution.std_param"),
    ("log_std", "distribution.log_std_param"),
  ):
    if old in actor:
      actor[new] = actor.pop(old)
  return checkpoint
