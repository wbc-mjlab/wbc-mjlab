"""bfloat16 autocast of selected networks, everything around them in float32.

PartyMimic (Roboparty, 2026) networks.py ``run_mlp`` / ``--amp bf16``, MIT; see
THIRD_PARTY_NOTICES: the MLP matmuls run in bf16 while state integration, kinematics,
rewards and normalization stay in float32.
"""

from __future__ import annotations

import torch
from torch import nn

_CLASSES: dict[type, type] = {}


class _Bf16:
  def forward(self, *args, **kwargs):
    if not any(isinstance(a, torch.Tensor) and a.is_cuda for a in args):
      return super().forward(*args, **kwargs)  # CPU (e.g. ONNX export): float32
    with torch.autocast("cuda", dtype=torch.bfloat16):
      out = super().forward(*args, **kwargs)
    return out.float()


def autocast_bf16(module: nn.Module) -> nn.Module:
  """Run ``module`` under bf16 autocast on CUDA inputs, returning float32. The module
  keeps its parameters, state dict and copies (a subclass, not a wrapper)."""
  cls = type(module)
  if issubclass(cls, _Bf16):
    return module
  if cls not in _CLASSES:
    _CLASSES[cls] = type(f"Bf16{cls.__name__}", (_Bf16, cls), {})
  module.__class__ = _CLASSES[cls]
  return module


def check_bf16(device) -> None:
  if torch.device(device).type != "cuda" or not torch.cuda.is_bf16_supported():
    raise ValueError("bf16 needs a CUDA device with bfloat16 support")
