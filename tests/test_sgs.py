"""Success-guided sampling (``RsiCfg.strategy = "success_target"``)."""

from __future__ import annotations

import pytest
import torch

from wbc_mjlab.env.mdp.sampling import (
  RsiCfg,
  SuccessTargetKernel,
  bin_sampling_probs,
  compute_sampling_prob_matrix,
  update_failure_ema,
)


def test_kernel_prefers_bins_near_target_success() -> None:
  kernel = SuccessTargetKernel(target=0.66, concentration=5.0, temperature=2.0)
  # success rates: impossible, at target, mastered
  failure = 1.0 - torch.tensor([0.02, 0.66, 0.98])
  probs = bin_sampling_probs(
    failure, temperature_base=1.0, uniform_ratio=0.0, success_target=kernel
  )
  assert abs(probs.sum().item() - 1.0) < 1e-5
  assert probs.argmax().item() == 1
  # Hardest-first would pick the impossible bin instead.
  hardest = bin_sampling_probs(failure, temperature_base=1.0, uniform_ratio=0.0)
  assert hardest.argmax().item() == 0


def test_uniform_ratio_keeps_every_bin_sampleable() -> None:
  kernel = SuccessTargetKernel(epsilon=1.0e-8)
  failure = torch.tensor([0.0, 0.34, 1.0])
  probs = bin_sampling_probs(
    failure, temperature_base=1.0, uniform_ratio=0.15, success_target=kernel
  )
  assert probs.min().item() >= 0.05 - 1e-6


def test_success_kernel_only_for_success_target() -> None:
  assert RsiCfg().success_kernel() is None
  cfg = RsiCfg(strategy="success_target", target_success=0.5, target_concentration=1.0)
  kernel = cfg.success_kernel()
  assert kernel is not None and kernel.target == 0.5 and kernel.concentration == 1.0


def test_prob_matrix_uses_kernel() -> None:
  failure = torch.tensor([[0.34, 1.0], [0.0, 0.5]])
  valid = torch.tensor([[True, True], [True, False]])
  probs = compute_sampling_prob_matrix(
    failure,
    valid,
    temperature_base=1.0,
    uniform_ratio=0.0,
    success_target=SuccessTargetKernel(),
  )
  assert abs(probs.sum().item() - 1.0) < 1e-5
  assert probs[1, 1].item() == 0.0
  assert probs[0, 0].item() == probs.max().item()


@pytest.mark.parametrize("strategy", ["success_target", "binary_failure"])
def test_termination_strategies_count_successes(strategy) -> None:
  levels = torch.full((2, 2), 0.34)
  update_failure_ema(
    levels,
    strategy=strategy,
    bins_per_trajectory=2,
    alpha=0.5,
    traj_ids=torch.tensor([0, 1]),
    start_bins=torch.tensor([0, 1]),
    episode_terminated=torch.tensor([True, False]),
    episode_similarity_sum=None,
    episode_step_count=None,
  )
  assert abs(levels[0, 0].item() - 0.67) < 1e-5  # failed: up
  assert (
    abs(levels[1, 1].item() - 0.17) < 1e-5
  )  # succeeded: down (binary_failure skips it)
  assert torch.allclose(levels[[0, 1], [1, 0]], torch.tensor(0.34))  # not visited


def test_sgs_task_and_studio_module() -> None:
  from wbc_mjlab.robots.env import make_wbc_env_cfg
  from wbc_mjlab.studio import presets

  rsi = make_wbc_env_cfg("g1", task_id="Wbc-G1-SGS").commands["motion"].rsi
  assert rsi.strategy == "success_target" and rsi.alpha == 0.01

  module = presets.load_module("sgs", "g1")
  assert module["builtin"]
  assert module["overrides"]["command"]["rsi.strategy"] == "success_target"
  env, _ = presets.build("Wbc-G1-Zest", {}, modules=["sgs"])
  assert env.commands["motion"].rsi.strategy == "success_target"
  with pytest.raises(ValueError):
    presets.save_module("sgs", {})


def test_play_export_keeps_training_rsi_stats(tmp_path, monkeypatch) -> None:
  from wbc_mjlab.export import policy_bundle

  calls = []
  monkeypatch.setattr(
    policy_bundle, "export_tracking_params_yaml", lambda d, *a, **k: d
  )
  monkeypatch.setattr(policy_bundle, "export_rsi_bin_stats", lambda *a: calls.append(a))
  policy_bundle.export_deploy_params(
    tmp_path, None, None, robot_id="g1", write_rsi_stats=False
  )
  assert calls == []
  policy_bundle.export_deploy_params(tmp_path, None, None, robot_id="g1")
  assert len(calls) == 1


def test_tasks_register_whatever_is_imported_first() -> None:
  import subprocess
  import sys

  code = (
    "import wbc_mjlab.env.mdp.sampling\n"
    "from mjlab.tasks.registry import list_tasks\n"
    "assert 'Wbc-G1-SGS' in list_tasks(), list_tasks()\n"
  )
  subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
