"""Bounded training-library probes for failure-directed RSI (BlinkMimic d264658).

https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/train.py
MIT; THIRD_PARTY_NOTICES. SuperMimic: https://arxiv.org/abs/2610.28480.
Uses the shared task, deterministic evaluation config and reset lifecycle. A sweep
visits tiles across the training library; it is not a held-out benchmark. Batches
bound memory independently of dataset duration. In-flight batches restart on resume.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from tensordict import TensorDict

from wbc_mjlab.env.mdp.sampling import failure_counts
from wbc_mjlab.rl.evaluation import deterministic_cfg


class CurriculumProbes:
  def __init__(
    self,
    algorithm,
    *,
    interval,
    start,
    num_envs,
    tile_s,
    lead_s,
    steps_per_iteration,
    asynchronous=False,
    cuda_graphs=False,
  ):
    if (
      interval < 1
      or start < 0
      or num_envs < 1
      or tile_s <= 0
      or not 0 <= lead_s[0] <= lead_s[1]
      or steps_per_iteration < 0
    ):
      raise ValueError("invalid curriculum probe configuration")
    if asynchronous and torch.device(algorithm.device).type != "cuda":
      raise ValueError("asynchronous probes require CUDA")
    from wbc_mjlab.rl.wm.distributed import world_size

    if world_size() > 1:
      raise ValueError(
        "curriculum probes currently require single-rank training; set probe_interval=0"
      )
    self.alg, self.interval, self.next_update = algorithm, interval, start
    self.num_envs, self.steps_per_iteration = num_envs, steps_per_iteration
    self.asynchronous, self.cuda_graphs = asynchronous, cuda_graphs
    self.command = algorithm.imagination.command
    self.motion = self.command.motion
    dt = algorithm.env.step_dt
    self.tile_steps = max(1, round(tile_s / dt))
    self.lead = (round(lead_s[0] / dt), round(lead_s[1] / dt))
    # Metadata only: never transfer live simulator state to construct the queue.
    tiles, clips = [], []
    for c, (a, b) in enumerate(
      zip(
        self.motion.segment_start_idx.tolist(),
        self.motion.segment_end_idx.tolist(),
        strict=True,
      )
    ):
      starts = list(range(a, max(a + 1, b - 1), self.tile_steps))
      tiles.extend(starts)
      clips.extend([c] * len(starts))
    self.tiles = torch.tensor(tiles, device=algorithm.device)
    self.clips = torch.tensor(clips, device=algorithm.device)
    self.cursor, self.sweeps = 0, 0
    self.counts = torch.zeros_like(self.command._probe_failures)
    self.env = self.policy = self.stream = self.event = None
    self.active = False
    self.batch_active = False
    self.generator = torch.Generator().manual_seed((algorithm.env.cfg.seed or 0) + 7919)
    order = torch.randperm(len(self.tiles), generator=self.generator).to(
      algorithm.device
    )
    self.tiles, self.clips = self.tiles[order], self.clips[order]

  def _ensure_env(self):
    if self.env is not None:
      return
    from mjlab.envs import ManagerBasedRlEnv

    cfg = deterministic_cfg(copy.deepcopy(self.alg.env.cfg))
    cfg.scene.num_envs = self.num_envs
    cfg.commands["motion"].rsi.recent_failure_share = 0.0
    cfg.commands["motion"].rsi.probe_share = 0.0
    # Construction uses a separate seed without perturbing the training RNG.
    devices = (
      [torch.device(self.alg.device)]
      if torch.device(self.alg.device).type == "cuda"
      else []
    )
    if self.asynchronous:
      self.stream = torch.cuda.Stream(device=self.alg.device)
      self.stream.wait_stream(torch.cuda.current_stream(self.alg.device))
    # TorchArray remembers the Warp stream at construction for indexed writes.
    # Construct the entire independent environment on its owning stream; changing
    # only the current stream later leaves those writes on the training stream.
    from wbc_mjlab.motion.library import reuse_online_motion

    with (
      torch.random.fork_rng(devices=devices),
      self._scope(),
      reuse_online_motion(self.motion),
    ):
      self.env = ManagerBasedRlEnv(cfg, device=self.alg.device)
    if self.stream is not None:
      self.stream.synchronize()  # one-time initialization / graph capture boundary
    self.policy = copy.deepcopy(self.alg.get_policy()).requires_grad_(False).eval()
    self.policy_forward = self._forward
    if self.cuda_graphs:
      if torch.device(self.alg.device).type != "cuda":
        raise ValueError("probe policy graphs require CUDA")
      from wbc_mjlab.rl.wm.algorithm import CudaGraphed

      self.policy_forward = CudaGraphed(torch.compile(self._forward, dynamic=False))

  @contextmanager
  def _scope(self):
    if self.stream is None:
      yield
    else:
      import warp as wp

      with (
        torch.cuda.stream(self.stream),
        wp.ScopedStream(wp.stream_from_torch(self.stream), sync_enter=False),
      ):
        yield

  def _forward(self, obs):
    return self.policy(TensorDict({"actor": obs}, batch_size=[self.num_envs]))

  def _start_batch(self):
    ids = torch.arange(self.num_envs, device=self.alg.device) + self.cursor
    self.valid = ids < len(self.tiles)
    ids = ids.clamp_max(len(self.tiles) - 1)
    self.batch_clips, tile = self.clips[ids], self.tiles[ids]
    lead = int(
      torch.randint(self.lead[0], self.lead[1] + 1, (), generator=self.generator)
    )
    starts = torch.maximum(tile - lead, self.motion.segment_start_idx[self.batch_clips])
    self.stop = torch.minimum(
      tile + self.tile_steps, self.motion.segment_end_idx[self.batch_clips] - 1
    )
    command = self.env.command_manager.get_term("motion")
    command.set_start_frames(self.batch_clips, starts)
    self.obs, _ = self.env.reset()
    self.alive = self.valid.clone()
    self.batch_counts = torch.zeros_like(self.counts)
    self.left = lead + self.tile_steps
    self.batch_active = True

  def _steps(self, count):
    command = self.env.command_manager.get_term("motion")
    for _ in range(count):
      frame = command.time_steps.clone()
      actions = self.policy_forward(self.obs["actor"])
      finite = torch.isfinite(actions).all(-1)
      actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
      if self.alg.clip_actions is not None:
        actions = actions.clamp(-self.alg.clip_actions, self.alg.clip_actions)
      self.obs, _, terminated, truncated, _ = self.env.step(actions)
      failed = self.alive & (terminated | ~finite)
      bins = command._bin_index_for_frame(self.batch_clips, frame)
      self.batch_counts.add_(
        failure_counts(self.counts.shape, self.batch_clips, bins, failed.float())
      )
      self.alive &= ~(terminated | truncated | ~finite | (frame + 1 >= self.stop))
    self.left -= count

  def _join(self):
    if self.event is not None:
      torch.cuda.current_stream(self.alg.device).wait_event(self.event)
      self.event = None

  @torch.no_grad()
  def tick(self):
    self._tick()
    if self.steps_per_iteration == 0:
      while self.active:
        self._tick()

  def _tick(self):
    self._join()
    if not self.active:
      if self.alg.num_updates < self.next_update:
        return
      self._ensure_env()
      self.policy.load_state_dict(self.alg.get_policy().state_dict())
      self.active = True
    if self.batch_active and self.left == 0:
      # Publish completed batches immediately, not after an entire large archive.
      # Decay once per sweep; each tile still contributes exactly once.
      if self.cursor == 0:
        self.command._probe_failures.mul_(self.command.cfg.rsi.probe_decay)
      self.command._probe_failures.add_(self.batch_counts)
      self.counts.add_(self.batch_counts)
      self.cursor += self.num_envs
      self.batch_active = False
      if self.cursor < len(self.tiles) and self.steps_per_iteration:
        self.active = False
        self.next_update = self.alg.num_updates + self.interval
        return
    if self.cursor >= len(self.tiles):
      self.cursor = 0
      self.counts.zero_()
      self.active = False
      self.sweeps += 1
      self.next_update = self.alg.num_updates + self.interval
      return
    if not self.batch_active:
      if self.stream is not None:
        self.stream.wait_stream(torch.cuda.current_stream(self.alg.device))
      with self._scope():
        self._start_batch()
    count = min(self.left, self.steps_per_iteration or self.left)
    if self.stream is None:
      self._steps(count)
    else:
      self.stream.wait_stream(torch.cuda.current_stream(self.alg.device))
      with self._scope():
        self._steps(count)
        self.event = torch.cuda.Event()
        self.event.record(self.stream)

  def state_dict(self):
    self._join()
    return {
      "version": 2,
      "cursor": self.cursor,
      "sweeps": self.sweeps,
      "next_update": self.next_update,
      "counts": self.counts.clone(),
      "generator": self.generator.get_state(),
      "tiles": self.tiles.clone(),
      "clips": self.clips.clone(),
    }

  def load_state_dict(self, state):
    self._join()
    tiles, clips = state["tiles"].to(self.tiles), state["clips"].to(self.clips)
    order, saved_order = self.tiles.argsort(), tiles.argsort()
    if not torch.equal(tiles[saved_order], self.tiles[order]) or not torch.equal(
      clips[saved_order], self.clips[order]
    ):
      raise ValueError("probe curriculum motion tiles changed")
    # Keep an older checkpoint's queue order (before deterministic shuffling).
    self.tiles, self.clips = tiles, clips
    self.cursor, self.sweeps = state["cursor"], state["sweeps"]
    self.next_update = state["next_update"]
    self.counts.copy_(state["counts"])
    if state.get("version", 1) < 2:
      # Older sweeps withheld all feedback until completion. Revisit their tiles
      # instead of publishing counts before the runner restores sampler state.
      self.cursor = 0
      self.counts.zero_()
    self.generator.set_state(state["generator"].cpu())
    self.active = self.batch_active = False

  def close(self):
    self._join()
    if self.stream is not None:
      self.stream.synchronize()
    if self.env is not None:
      self.env.close()
      self.env = None
    self.active = self.batch_active = False
