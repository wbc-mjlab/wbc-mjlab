"""Saved behavior and checkpoint compatibility shared by Play, resume and evaluation."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import mjlab  # noqa: F401
import pytest
import torch

from wbc_mjlab.rl.algos import rl_cfg
from wbc_mjlab.run_params import load_run_config
from wbc_mjlab.tasks import register_all_wbc_tasks


@pytest.mark.parametrize("algo", ["ppo", "wm"])
def test_saved_run_restores_behavior_and_old_actor_keys(tmp_path, monkeypatch, algo):
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.rl.algos import env_cfg
  from wbc_mjlab.scripts.train import _safe_dump_yaml

  register_all_wbc_tasks()
  cfg = env_cfg(load_env_cfg("Wbc-G1-Lookahead"), algo)
  name = next(iter(cfg.rewards))
  cfg.rewards[name].weight = 1.234
  removed = next(reversed(cfg.rewards))
  del cfg.rewards[removed]
  cfg.commands["motion"].motion_file = "/tmp/saved-motion.npz"
  cfg.commands["motion"].pose_range = {"x": (-0.12, 0.12)}
  cfg.scene.env_spacing = 3.456
  agent = asdict(rl_cfg("Wbc-G1-Lookahead", algo))
  agent["algorithm"]["policy_architecture"] = "motion_encoder"
  agent["algorithm"]["encoder_hidden_dims"] = (64, 32)
  agent["actor"].pop("architecture")
  agent["actor"].pop("encoder_hidden_dims")
  for kind, value in (
    ("run", {"task_id": "Wbc-G1-Lookahead"}),
    ("env", asdict(cfg)),
    ("agent", agent),
  ):
    _safe_dump_yaml(tmp_path / "params" / kind, value)
  restored = load_run_config(tmp_path, play=False)
  assert restored.env.rewards[name].weight == 1.234
  assert removed not in restored.env.rewards
  assert list(restored.env.rewards) == list(cfg.rewards)
  assert restored.env.commands["motion"].motion_file == "/tmp/saved-motion.npz"
  assert tuple(restored.env.commands["motion"].pose_range["x"]) == (-0.12, 0.12)
  assert restored.agent.actor.architecture == "motion_encoder"
  assert restored.agent.actor.encoder_hidden_dims == (64, 32)
  assert restored.env.scene.env_spacing == 3.456
  assert (
    restored.env.actions["joint_pos"].transmission_type
    == cfg.actions["joint_pos"].transmission_type
  )
  played = load_run_config(tmp_path)
  assert played.env.rewards[name].weight == 1.234
  assert played.env.commands["motion"].pose_range == {}

  # Resume hands the same defaults to tyro, which can then apply explicit CLI flags.
  import sys
  from dataclasses import replace

  from mjlab.scripts import train as upstream
  from mjlab.tasks import registry
  from mjlab.utils import os as upstream_os

  from wbc_mjlab.scripts import train

  entry = registry._REGISTRY["Wbc-G1-Lookahead"]
  monkeypatch.setitem(registry._REGISTRY, "Wbc-G1-Lookahead", replace(entry))
  checkpoint = tmp_path / "model_0.pt"
  monkeypatch.setattr(upstream_os, "get_checkpoint_path", lambda *a: checkpoint)
  monkeypatch.setattr(
    sys, "argv", ["train", "--task", "Wbc-G1-Lookahead", "--agent.resume", "True"]
  )

  def parse():
    import tyro

    selected = registry._REGISTRY["Wbc-G1-Lookahead"]
    assert selected.env_cfg.rewards[name].weight == 1.234
    assert selected.rl_cfg.actor.architecture == "motion_encoder"
    assert type(selected.rl_cfg.algorithm) is type(restored.agent.algorithm)
    parsed = tyro.cli(
      upstream.TrainConfig,
      default=upstream.TrainConfig.from_task("Wbc-G1-Lookahead"),
      args=["--env.scene.num-envs", "4", "--agent.actor.latent-dim", "8"],
      config=mjlab.TYRO_FLAGS,
    )
    assert parsed.env.scene.num_envs == 4 and parsed.agent.actor.latent_dim == 8
    assert parsed.env.commands["motion"].pose_range["x"] == (-0.12, 0.12)

  monkeypatch.setattr(upstream, "main", parse)
  train.main()


def test_legacy_actor_checkpoint_matches_current_model(tmp_path):
  from rsl_rl.models import MLPModel
  from tensordict import TensorDict

  from wbc_mjlab.rl.checkpoints import read_checkpoint

  obs = TensorDict({"actor": torch.randn(4, 6)}, batch_size=[4])
  cfg = dict(
    hidden_dims=(8,),
    distribution_cfg={"class_name": "GaussianDistribution", "std_type": "scalar"},
  )
  actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 3, **deepcopy(cfg))
  old = {}
  for key, value in actor.state_dict().items():
    if key.startswith("mlp."):
      key = "actor." + key.removeprefix("mlp.")
    elif key.startswith("obs_normalizer."):
      key = "actor_obs_normalizer." + key.removeprefix("obs_normalizer.")
    elif key == "distribution.std_param":
      key = "std"
    old[key] = value
  path = tmp_path / "legacy.pt"
  torch.save({"model_state_dict": old}, path)
  loaded = MLPModel(obs, {"actor": ["actor"]}, "actor", 3, **cfg)
  loaded.load_state_dict(read_checkpoint(path)["actor_state_dict"])
  assert torch.equal(actor(obs), loaded(obs))


def test_sampler_rejects_same_shape_different_library_without_mutation():
  from wbc_mjlab.env.mdp.sampling import restore_sampling_state, sampling_state

  def command(lengths):
    return SimpleNamespace(
      motion=SimpleNamespace(
        segment_names=["a", "b"], segment_length=torch.tensor(lengths)
      ),
      cfg=SimpleNamespace(rsi=SimpleNamespace(bin_width_s=1.0)),
      bin_failure_levels=torch.ones(2, 3),
      _bin_memory=torch.ones(2, 3),
      bin_valid_mask=torch.ones(2, 3, dtype=torch.bool),
      bin_visit_counts=torch.zeros(2, 3),
    )

  original, target = command([100, 100]), command([100, 90])
  target.bin_failure_levels.zero_()
  with pytest.raises(ValueError):
    restore_sampling_state(target, sampling_state(original), strict=True)
  assert not target.bin_failure_levels.any()
