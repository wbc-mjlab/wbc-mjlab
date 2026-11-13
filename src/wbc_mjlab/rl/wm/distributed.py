"""Data parallelism within the existing mjlab/rsl_rl launcher.

SuperMimic (Liao et al., 2026), Appendix D, https://arxiv.org/abs/2610.28480.
Explicit gradient averaging follows the approach in CuboiLeo/SuperMimic-Reproduced
at ba61053, src/supermimic/training/distributed.py:
https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2
The implementation here also pools running normalization statistics across ranks.
"""

import torch
import torch.distributed as dist


def world_size() -> int:
  return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def sum_(x: torch.Tensor) -> torch.Tensor:
  if world_size() > 1:
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
  return x


@torch.no_grad()
def average_gradients(module: torch.nn.Module) -> None:
  """Average before clipping; every rank takes the same number of optimizer steps."""
  if world_size() == 1:
    return
  grads = [p.grad for p in module.parameters() if p.grad is not None]
  if not grads:
    return
  flat = torch.cat([g.reshape(-1) for g in grads])
  sum_(flat).div_(world_size())
  offset = 0
  for grad in grads:
    grad.copy_(flat[offset : offset + grad.numel()].view_as(grad))
    offset += grad.numel()


@torch.no_grad()
def broadcast(module: torch.nn.Module) -> None:
  if world_size() > 1:
    for tensor in (*module.parameters(), *module.buffers()):
      dist.broadcast(tensor, src=0)
