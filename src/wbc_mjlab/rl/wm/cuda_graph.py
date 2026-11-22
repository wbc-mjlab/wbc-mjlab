"""CUDA graph capture, replay and validation of referenced tensor storage."""

import dataclasses
from typing import Any

import torch


def graph_tensors(modules, optimizers=()):
  """Tensor storage read by a captured update, including nested Muon/Adam state.

  Address validation inspired by official SuperMimic learner_utils.py (Apache-2.0),
  abc124c1bd5976341034668af3d12c087f2f1f55. No device synchronization is required.
  """
  tensors = [t for module in modules for t in (*module.parameters(), *module.buffers())]

  def visit(value):
    if isinstance(value, torch.Tensor):
      tensors.append(value)
    elif isinstance(value, dict):
      for child in value.values():
        visit(child)
    elif isinstance(value, (list, tuple)):
      for child in value:
        visit(child)

  for optimizer in optimizers:
    visit(optimizer.state_dict())
  return tensors


class GraphStorageGuard:
  """Fail before replay if a referenced tensor is replaced, resized or re-strided."""

  def __init__(self, state):
    self.state = state
    self.signature = None

  def current(self):
    return [
      (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
      for t in self.state()
    ]

  def capture(self):
    self.signature = self.current()

  def check(self):
    if self.signature != self.current():
      raise RuntimeError(
        "CUDA graph storage changed; reset and recapture the graph before replay"
      )


class CudaGraphed:
  """``fn`` as one CUDA graph: after ``warmup`` eager calls (on a side stream, as capture
  wants), it is captured once; each call then copies its tensor arguments into the
  captured ones and replays. Arguments must keep their shapes; the returned tensors are
  overwritten by the next call."""

  def __init__(self, fn, warmup: int = 2, state=None):
    self._fn, self._warmup, self._calls = fn, warmup, 0
    self._graph: torch.cuda.CUDAGraph | None = None
    self._inputs: Any = None
    self._outputs: Any = None
    self._guard = GraphStorageGuard(state) if state is not None else None

  def reset(self):
    self._graph = self._inputs = self._outputs = None
    self._calls = 0

  def __call__(self, *args):
    # Contiguous in every call: compiled callees specialize on input strides.
    if self._graph is None and self._calls < self._warmup:
      self._calls += 1
      stream = torch.cuda.Stream()
      stream.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(stream):
        out = self._fn(*map_tensors(args, torch.Tensor.contiguous))
      torch.cuda.current_stream().wait_stream(stream)
      return out
    if self._graph is None:
      self._inputs = map_tensors(
        args, lambda x: x.clone(memory_format=torch.contiguous_format)
      )
      self._graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(self._graph):
        self._outputs = self._fn(*self._inputs)
      if self._guard is not None:
        self._guard.capture()
    else:
      if self._guard is not None:
        self._guard.check()
      _copy_tensors(self._inputs, args)
    self._graph.replay()
    return self._outputs


def map_tensors(x, fn):
  if isinstance(x, torch.Tensor):
    return fn(x)
  if dataclasses.is_dataclass(x):
    return type(x)(
      *(map_tensors(getattr(x, f.name), fn) for f in dataclasses.fields(x))
    )
  if isinstance(x, (list, tuple)):
    return type(x)(map_tensors(v, fn) for v in x)
  return x


def _copy_tensors(dst, src) -> None:
  if isinstance(dst, torch.Tensor):
    dst.copy_(src)
  elif dataclasses.is_dataclass(dst):
    for f in dataclasses.fields(dst):
      _copy_tensors(getattr(dst, f.name), getattr(src, f.name))
  elif isinstance(dst, (list, tuple)):
    for d, s in zip(dst, src, strict=True):
      _copy_tensors(d, s)
