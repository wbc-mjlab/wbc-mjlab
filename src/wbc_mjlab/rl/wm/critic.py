"""Optional structured value network; the default actor-input critic is unchanged.

Inspired by official SuperMimic's privileged critic and LayerNorm MLP,
models.py/world_model_env.py at abc124c1bd5976341034668af3d12c087f2f1f55
(Apache-2.0); Liao et al., https://arxiv.org/abs/2610.28480.
Unlike the upstream compact critic, retain the policy's full proprioceptive
history and last action, compressed by a learned encoder rather than discarded.
"""

import torch
from torch import nn


def value_mlp(input_dim, hidden, activation, *, layer_norm=False):
  act = {"elu": nn.ELU, "gelu": nn.GELU, "silu": nn.SiLU, "swish": nn.SiLU}
  if activation not in act:
    raise ValueError(f"structured/LayerNorm critic does not support {activation}")
  layers = []
  for width in hidden:
    layers.append(nn.Linear(input_dim, width))
    if layer_norm:
      layers.append(nn.LayerNorm(width))
    layers.append(act[activation]())
    input_dim = width
  return nn.Sequential(*layers, nn.Linear(input_dim, 1))


class StructuredCritic(nn.Module):
  """[physical state, tracking, context, clean goal, policy history] → value."""

  def __init__(self, privileged_dim, history_dim, hidden, activation, layer_norm):
    super().__init__()
    self.privileged_dim = privileged_dim
    # Fixed small bottleneck; architecture changes are checkpoint-versioned.
    self.history_encoder = nn.Sequential(nn.Linear(history_dim, 64), nn.ELU())
    self.value = value_mlp(
      privileged_dim + 64, hidden, activation, layer_norm=layer_norm
    )

  def forward(self, x):
    return self.value(
      torch.cat(
        [
          x[..., : self.privileged_dim],
          self.history_encoder(x[..., self.privileged_dim :]),
        ],
        dim=-1,
      )
    )
