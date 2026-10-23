"""Replay of simulator steps on the GPU: ``[step, env]`` rings, sampled as k-step windows."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Batch:
  """Rows ``[k, B]`` (or ``[B]``) of the replay, and the env of each column."""

  state: torch.Tensor  # RobotState vector, env frame
  action: torch.Tensor  # raw action taken (with exploration noise)
  setpoint: torch.Tensor  # PD targets it produced
  last_action: torch.Tensor  # action of the step before (the actor's ``actions`` obs)
  time_step: torch.Tensor  # reference frame index (global)
  trajectory: torch.Tensor  # reference segment
  reference_offset: torch.Tensor
  rewards: torch.Tensor  # learned reward terms of the step (weighted)
  terminated: torch.Tensor  # the step ended the episode (not a time-out)
  done: torch.Tensor  # the episode ended (terminated or timed out): the env reset
  continues: torch.Tensor  # the next row is this env's next state
  observed: torch.Tensor  # this step's values of the history observation terms
  env_ids: torch.Tensor
  past: torch.Tensor | None = None  # earlier ``observed`` values, oldest first


class Replay:
  """Last ``capacity`` steps of every env. Row ``t`` holds the state before step ``t``, the
  action taken and what the step produced; ``continues[t]`` says whether row ``t + 1`` is
  the state that action led to (no reset, no reference resample in between)."""

  def __init__(
    self,
    capacity: int,
    num_envs: int,
    state_dim: int,
    num_actions: int,
    num_rewards: int,
    observed_dim: int,
    device: torch.device | str,
  ):
    self.capacity, self.num_envs, self.device = capacity, num_envs, device

    def buf(*shape, dtype=torch.float32):
      return torch.zeros(capacity, num_envs, *shape, dtype=dtype, device=device)

    self._fields = {
      "state": buf(state_dim),
      "action": buf(num_actions),
      "setpoint": buf(num_actions),
      "last_action": buf(num_actions),
      "time_step": buf(dtype=torch.long),
      "trajectory": buf(dtype=torch.long),
      "reference_offset": buf(3),
      "rewards": buf(num_rewards),
      "terminated": buf(dtype=torch.bool),
      "done": buf(dtype=torch.bool),
      "continues": buf(dtype=torch.bool),
      "observed": buf(observed_dim),
    }
    self._next = 0  # row the next step is written to
    self.size = 0

  def begin(self, **row: torch.Tensor) -> None:
    """Write the state and action of the step about to run."""
    for name, value in row.items():
      self._fields[name][self._next] = value

  def end(self, **row: torch.Tensor) -> None:
    """Write what the step produced, completing the row."""
    for name, value in row.items():
      self._fields[name][self._next] = value
    self._next = (self._next + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def _rows(self, age: torch.Tensor) -> torch.Tensor:
    """Ring index of rows ``age`` steps after the oldest one."""
    return (self._next - self.size + age) % self.capacity

  def sample_windows(self, batch: int, length: int) -> Batch:
    """``batch`` windows of ``length`` consecutive rows (fields shaped ``[length, batch]``)."""
    if self.size < length:
      raise ValueError(f"replay holds {self.size} steps, windows need {length}")
    start = torch.randint(0, self.size - length + 1, (batch,), device=self.device)
    env = torch.randint(0, self.num_envs, (batch,), device=self.device)
    rows = self._rows(
      start.unsqueeze(0) + torch.arange(length, device=self.device).unsqueeze(1)
    )
    cols = env.unsqueeze(0).expand(length, batch)
    return Batch(**{k: v[rows, cols] for k, v in self._fields.items()}, env_ids=env)

  def sample_states(self, batch: int, past_steps: int = 0) -> Batch:
    """``batch`` single rows (fields shaped ``[batch]``), with the ``past_steps`` earlier
    ``observed`` values of each (``past``), as the env's observation history held them:
    across an episode start, the episode's first value repeats."""
    if self.size <= past_steps:
      raise ValueError(
        f"replay holds {self.size} steps, the history needs {past_steps}"
      )
    age = torch.randint(past_steps, self.size, (batch,), device=self.device)
    env = torch.randint(0, self.num_envs, (batch,), device=self.device)
    rows = self._rows(age)
    out = Batch(**{k: v[rows, env] for k, v in self._fields.items()}, env_ids=env)
    if past_steps:
      observed, done = self._fields["observed"], self._fields["done"]
      value, alive, past = out.observed, torch.ones_like(out.done), []
      for k in range(1, past_steps + 1):
        r = self._rows(age - k)
        alive = alive & ~done[r, env]
        value = torch.where(alive.unsqueeze(-1), observed[r, env], value)
        past.append(value)
      out.past = torch.stack(past[::-1], dim=1)
    return out
