"""Optimizer: Muon (hidden matrices) + Adam.

Adapted from BlinkMimic (Fuyi Chen, 2026), commit d264658:
https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/optim.py
MIT; see THIRD_PARTY_NOTICES. SuperMimic: https://arxiv.org/abs/2610.28480.
Private PyTorch Muon helpers are isolated here and checked at construction.

- ``MuonAdam``: Muon (Jordan et al. 2024; torch.optim.Muon) on the hidden weight matrices of the given MLPs, Adam on
  their first and last Linear and on all vectors. Muon's step is rescaled to the update RMS of Adam
  (adjust_lr_fn='match_rms_adamw', Liu et al. 2025), so the Adam learning rate is reused unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _linears(m: nn.Module) -> list[nn.Linear]:
  return [x for x in m.modules() if isinstance(x, nn.Linear)]


class MuonAdam:
  """Muon on hidden matrices of ``mlps`` (all Linear layers except each MLP's first and last), Adam elsewhere.

  ``capturable`` (--cuda_graph): the learning rate is a 0-dim device tensor (set it with ``fill_``), Adam is the
  capturable fused kernel, and the Muon step is done here with the same arithmetic as torch.optim.Muon but with the
  tensor learning rate (torch.optim.Muon reads a tensor lr on the host, so a captured step would freeze it)."""

  def __init__(self, model: nn.Module, mlps: list[nn.Module], lr: float, betas=(0.9, 0.95), fused: bool = True,
                 capturable: bool = False):  # fmt: skip
    hidden = {id(layer.weight) for m in mlps for layer in _linears(m)[1:-1]}
    muon = [p for p in model.parameters() if id(p) in hidden]
    adam = [p for p in model.parameters() if id(p) not in hidden and p.requires_grad]
    self.capturable = capturable
    if capturable:
      dev = muon[0].device
      self.muon = torch.optim.Muon(
        muon,
        lr=torch.tensor(float(lr), device=dev),
        weight_decay=0.0,
        adjust_lr_fn="match_rms_adamw",
      )
      self.adam = torch.optim.Adam(
        adam,
        lr=torch.tensor(float(lr), device=dev),
        betas=betas,
        fused=True,
        capturable=True,
      )
    else:
      self.muon = torch.optim.Muon(
        muon, lr=lr, weight_decay=0.0, adjust_lr_fn="match_rms_adamw"
      )
      self.adam = torch.optim.Adam(adam, lr=lr, betas=betas, fused=fused)

  @property
  def param_groups(self):
    # a property: Optimizer.load_state_dict replaces the group dicts, so a list built in __init__ goes stale on --resume
    return self.muon.param_groups + self.adam.param_groups

  def zero_grad(self, set_to_none: bool = True):
    (
      self.muon.zero_grad(set_to_none=set_to_none),
      self.adam.zero_grad(set_to_none=set_to_none),
    )

  @torch.no_grad()
  def _muon_step_tensor_lr(self, ok: torch.Tensor | None = None):
    from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

    for g in self.muon.param_groups:
      assert g["weight_decay"] == 0.0
      for p in g["params"]:
        if p.grad is None:
          continue
        st = self.muon.state[p]
        if "momentum_buffer" not in st:
          st["momentum_buffer"] = torch.zeros_like(
            p.grad, memory_format=torch.preserve_format
          )
        buf = st["momentum_buffer"]
        if ok is None:
          buf.lerp_(p.grad, 1 - g["momentum"])
        else:
          buf.copy_(torch.where(ok, buf.lerp(p.grad, 1 - g["momentum"]), buf))
        update = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
        update = _zeropower_via_newtonschulz(
          update, g["ns_coefficients"], g["ns_steps"], g["eps"]
        )
        ratio = _adjust_lr(1.0, g["adjust_lr_fn"], p.shape)
        d = update.to(p.dtype) * (
          g["lr"] * -ratio
        )  # fp32 product (a bf16 update x 0-dim tensor stays bf16)
        p.add_(d if ok is None else torch.where(ok, d, 0.0))

  @torch.no_grad()
  def step(self, ok: torch.Tensor | None = None):
    """``ok``: optional 0-dim bool device tensor; where it is false the whole step is skipped without a host read
    (parameters, momenta, Adam moments and step counts unchanged). The gradients must be finite (zero them first)."""
    if ok is not None:
      if self.capturable:
        self._muon_step_tensor_lr(ok)
      else:  # torch.optim.Muon's own step (unchanged arithmetic), undone where ok is false
        ps = [
          p for g in self.muon.param_groups for p in g["params"] if p.grad is not None
        ]
        bufs = lambda: [self.muon.state[p]["momentum_buffer"] for p in ps]  # noqa: E731
        for p in ps:
          self.muon.state[p].setdefault(
            "momentum_buffer",
            torch.zeros_like(p.grad, memory_format=torch.preserve_format),
          )
        keep = [t.clone() for t in ps + bufs()]
        self.muon.step()
        for t, k in zip(ps + bufs(), keep, strict=True):
          t.copy_(torch.where(ok, t, k))
      assert self.adam.defaults["fused"], (
        "a gated step needs the fused Adam (--fused_adam)"
      )
      self.adam.found_inf = (
        ~ok
      ).float()  # the fused kernel skips the step and undoes the step count
      try:
        self.adam.step()
      finally:
        self.adam.found_inf = None
      return
    if self.capturable:
      self._muon_step_tensor_lr()
    else:
      self.muon.step()
    self.adam.step()

  def state_dict(self):
    return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

  def load_state_dict(self, sd: dict):
    lrs = [g["lr"] for g in self.param_groups]
    self.muon.load_state_dict(sd["muon"]), self.adam.load_state_dict(sd["adam"])
    if (
      self.capturable
    ):  # keep the device learning-rate tensors (the training loop fills them)
      for g, lr in zip(self.param_groups, lrs, strict=True):
        lr.fill_(float(g["lr"]))
        g["lr"] = lr


def make_optimizer(
  model, mlps, *, kind, lr, betas=(0.9, 0.999), fused=False, capturable=False
):
  """One optimizer boundary for the shared WM trainer; CPU Adam remains supported."""
  if kind == "muon":
    if not hasattr(torch.optim, "Muon"):
      raise ValueError("Muon requires a PyTorch version with torch.optim.Muon")
    try:
      from torch.optim._muon import (  # noqa: F401
        _adjust_lr,
        _zeropower_via_newtonschulz,
      )
    except ImportError as exc:
      raise ValueError(
        "This PyTorch version lacks the capturable Muon helpers"
      ) from exc
    if not any(len(_linears(m)) > 2 for m in mlps):
      raise ValueError("Muon needs an MLP with at least two hidden layers")
    return MuonAdam(model, mlps, float(lr), betas, fused, capturable)
  if kind != "adam":
    raise ValueError(f"unknown optimizer: {kind}")
  return torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=lr.clone(),
    betas=betas,
    fused=fused,
    capturable=capturable,
  )


def step_if_finite(optimizer, parameters, *, enabled=False, valid=None):
  """Blink's device-side bad-gradient gate; skip parameters, moments and counters.

  Fused Adam and the capturable Muon path avoid a host read. The eager fallback is
  available for CPU debugging. Sanitization prevents NaNs entering Muon's matrix
  iteration even when the resulting update is gated off. ``valid`` also gates on
  the loss and pre-clipping norm: an overflowed norm can turn finite gradients
  into zeros, which must not advance optimizer momentum or counters.
  """
  if not enabled:
    optimizer.step()
    return
  params = [p for p in parameters if p.grad is not None]
  ok = torch.stack([torch.isfinite(p.grad).all() for p in params]).all()
  if valid is not None:
    ok = ok & valid
  for p in params:
    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
  if isinstance(optimizer, MuonAdam) and optimizer.adam.defaults["fused"]:
    optimizer.step(ok=ok)
  elif not isinstance(optimizer, MuonAdam) and optimizer.defaults.get("fused"):
    optimizer.found_inf = (~ok).float()
    try:
      optimizer.step()
    finally:
      optimizer.found_inf = None
  elif bool(ok):
    optimizer.step()
