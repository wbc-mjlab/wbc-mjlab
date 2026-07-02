"""Tests for project/data path resolution across extension repos."""

from __future__ import annotations

from pathlib import Path

import pytest

import wbc_mjlab.data_paths as data_paths


@pytest.fixture(autouse=True)
def _reset_registered_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  monkeypatch.setattr(data_paths, "_REGISTERED_PROJECT_ROOTS", [])
  monkeypatch.chdir(tmp_path)
  yield


def test_resolve_dataset_root_prefers_cwd_project(tmp_path: Path) -> None:
  (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
  dataset = tmp_path / "data" / "g1" / "samples"
  dataset.mkdir(parents=True)
  (dataset / "clip.pkl").write_bytes(b"x")

  root = data_paths.resolve_dataset_root("g1", "samples")
  assert root == dataset


def test_resolve_dataset_root_uses_registered_extension_root(tmp_path: Path) -> None:
  ext = tmp_path / "ext-repo"
  ext.mkdir()
  (ext / "pyproject.toml").write_text("[project]\nname='ext'\n")
  dataset = ext / "data" / "g1" / "samples"
  dataset.mkdir(parents=True)

  data_paths.register_project_root(ext)
  root = data_paths.resolve_dataset_root("g1", "samples")
  assert root == dataset


def test_register_wbc_extension_registers_project_root() -> None:
  from wbc_mjlab.extension import WbcRobotSpec, register_wbc_extension
  from wbc_mjlab.robots.env import make_wbc_env_cfg, make_wbc_rl_cfg
  from wbc_mjlab.tasks.config import WbcTaskConfig

  ext = Path(__file__).resolve().parents[1] / "_tmp_ext_root"
  ext.mkdir(exist_ok=True)
  (ext / "pyproject.toml").write_text("[project]\nname='ext'\n")
  dataset = ext / "data" / "testbot" / "clips"
  dataset.mkdir(parents=True)

  try:
    register_wbc_extension(
      WbcRobotSpec(
        robot_id="testbot",
        project_root=ext,
        make_env_cfg=lambda **_: make_wbc_env_cfg("g1", task_id="Wbc-G1"),
        make_rl_cfg=make_wbc_rl_cfg,
      ),
      WbcTaskConfig(
        task_id="Wbc-Testbot",
        robot_id="testbot",
        description="smoke",
        experiment_name="wbc_testbot",
        build_env_cfg=lambda: make_wbc_env_cfg("g1", task_id="Wbc-G1"),
      ),
    )
    assert ext in data_paths.iter_project_roots()
    assert data_paths.resolve_dataset_root("testbot", "clips") == dataset
  finally:
    import shutil

    shutil.rmtree(ext, ignore_errors=True)
