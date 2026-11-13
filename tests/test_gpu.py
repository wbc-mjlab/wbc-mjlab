"""GPU telemetry, physical-to-CUDA mapping and non-destructive Docker selection."""

import json
from types import SimpleNamespace

import pytest

from wbc_mjlab import gpu


@pytest.fixture
def core():
  import mjlab  # noqa: F401

  from wbc_mjlab.studio.core import StudioCore

  instance = StudioCore(robot="g1", device="cpu")
  yield instance
  instance.close()


def test_gpu_metrics_colors_unknown_and_csv_names():
  rows = gpu.parse_gpus(
    '0, GPU-a, "Test, GPU", 24000, 20000, 4000, 10\n'
    "3, GPU-b, GPU B, 24000, 10000, 14000, 40\n"
    "7, GPU-c, GPU C, 24000, 1000, 23000, 5\n"
    "8, GPU-d, GPU D, 24000, 20000, 4000, 90\n"
    "9, GPU-e, GPU E, 24000, [N/A], [N/A], [Not Supported]\n"
  )
  assert [r["status"] for r in rows] == [
    "available",
    "moderate",
    "busy",
    "busy",
    "unknown",
  ]
  assert rows[0]["name"] == "Test, GPU"
  assert rows[-1]["free_mib"] is None
  assert "GiB free" in gpu.label(rows[0])
  assert "?" in gpu.label(rows[-1])


def test_selection_uses_cuda_ordinals_not_host_indices(monkeypatch):
  import torch

  monkeypatch.setattr(
    gpu,
    "query_gpus",
    lambda: [
      {"index": 0, "uuid": "GPU-a"},
      {"index": 3, "uuid": "GPU-b"},
      {"index": 7, "uuid": "GPU-c"},
    ],
  )
  monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
  monkeypatch.setattr(
    torch.cuda, "get_device_properties", lambda i: SimpleNamespace(uuid=["c", "b"][i])
  )
  rows = gpu.visible_gpus()
  assert gpu.training_gpu_indices(["GPU-b", "GPU-c"], rows) == [1, 0]
  for chosen in ([], ["GPU-a"], ["GPU-b", "GPU-b"]):
    with pytest.raises(ValueError):
      gpu.training_gpu_indices(chosen, rows)


def test_single_device_resolution_keeps_default_and_checks_visibility(monkeypatch):
  monkeypatch.setattr(gpu, "visible_gpus", lambda: [{"uuid": "GPU-b", "cuda_index": 2}])
  assert gpu.resolve_device(None, "cpu") == "cpu"
  assert gpu.resolve_device(None, "cuda:1") == "cuda:1"
  assert gpu.resolve_device("GPU-b", "cpu") == "cuda:2"
  for invalid in ("GPU-gone", ""):
    with pytest.raises(ValueError):
      gpu.resolve_device(invalid, "cpu")


def test_play_gpu_switch_rebuilds_without_changing_core_device(core, monkeypatch):
  from wbc_mjlab.studio import jobs, play

  monkeypatch.setattr(
    gpu,
    "visible_gpus",
    lambda: [
      {"uuid": "GPU-a", "cuda_index": 0},
      {"uuid": "GPU-b", "cuda_index": 2},
    ],
  )
  monkeypatch.setattr(
    jobs, "find_run", lambda run: SimpleNamespace(motion_source=lambda: "clips")
  )
  s = core.play
  s.session = SimpleNamespace(
    device="cuda:0",
    num_envs=16,
    task="task",
    clip_names=lambda: ["clip"],
    cmd=SimpleNamespace(trajectory_ids=[0]),
  )
  s.run_label, s.motion, s.checkpoint = "run", "clips", play.ZERO
  monkeypatch.setattr(s, "_render", lambda: None)
  rebuilt = []
  monkeypatch.setattr(s, "_rebuild", lambda *args: rebuilt.append(args) or True)
  try:
    original = s.session
    assert s.configure("run", "", motion="clips", gpu_uuid="GPU-a") == "unchanged"
    assert not rebuilt and s.gpu_uuid == "GPU-a"
    assert s.configure("run", "", motion="clips", gpu_uuid="GPU-b") == "rebuilding"
    assert rebuilt[-1][-2:] == ("cuda:2", "GPU-b")
    assert core.device == "cpu" and s.session is original
    with pytest.raises(ValueError, match="no longer visible"):
      s.configure("run", "", motion="clips", gpu_uuid="GPU-gone")
    assert s.session is original
  finally:
    s.session = None  # the session above is a lightweight mock


def test_play_api_forwards_gpu_choice(core, monkeypatch):
  from fastapi.testclient import TestClient

  from wbc_mjlab.studio.web.server import create_app

  captured = {}
  monkeypatch.setattr(
    core.play,
    "configure",
    lambda *args, **kwargs: captured.update(kwargs) or "unchanged",
  )
  response = TestClient(create_app(core), client=("127.0.0.1", 1234)).post(
    "/api/play/configure",
    json={"run": "run", "gpu_uuid": "GPU-b"},
  )
  assert response.status_code == 200
  assert captured["gpu_uuid"] == "GPU-b"


def test_docker_selection_preserves_env_and_refuses_running_workspace(
  tmp_path, monkeypatch
):
  (tmp_path / ".devcontainer").mkdir()
  env = tmp_path / ".env"
  env.write_text("# keep this\nWBC_STUDIO_PORT=18765\n")
  env.chmod(0o600)
  running = True

  def run(*args, **kwargs):
    return SimpleNamespace(
      returncode=0, stdout="container-id" if running else "", stderr=""
    )

  monkeypatch.setattr(gpu.subprocess, "run", run)
  with pytest.raises(ValueError, match="docker-stop"):
    gpu.save_selection(tmp_path, ["GPU-a"], "docker compose")
  assert not (tmp_path / ".devcontainer/gpus.local.yaml").exists()
  running = False
  gpu.save_selection(tmp_path, ["GPU-a"], "docker compose")
  selection = json.loads((tmp_path / ".devcontainer/gpus.local.yaml").read_text())
  assert selection["services"]["gpu"]["gpus"][0]["device_ids"] == ["GPU-a"]
  assert "WBC_STUDIO_PORT=18765" in env.read_text()
  assert "# keep this" in env.read_text()
  assert env.stat().st_mode & 0o777 == 0o600
  running = True
  gpu.save_selection(
    tmp_path, ["GPU-a"], "docker compose"
  )  # unchanged selection is harmless
  with pytest.raises(ValueError, match="docker-stop"):
    gpu.save_selection(tmp_path, ["GPU-b"], "docker compose")


def test_api_gpu_status(core, monkeypatch):
  from fastapi.testclient import TestClient

  from wbc_mjlab.studio.web.server import create_app

  monkeypatch.setattr(gpu, "visible_gpus", lambda: [{"uuid": "GPU-a", "cuda_index": 0}])
  client = TestClient(create_app(core), client=("127.0.0.1", 1234))
  assert client.get("/api/gpus").json()["gpus"][0]["uuid"] == "GPU-a"


def test_training_launch_uses_selected_cuda_ordinal(core, monkeypatch):
  from wbc_mjlab.studio import jobs

  monkeypatch.setattr(gpu, "visible_gpus", lambda: [{"uuid": "GPU-b", "cuda_index": 1}])
  captured = {}

  def launch(kind, args, **kwargs):
    captured.update(args=args, **kwargs)
    return SimpleNamespace(name="gpu-test")

  monkeypatch.setattr(jobs, "launch", launch)
  core.training.launch("Wbc-G1", "samples", gpu_uuids=["GPU-b"])
  assert captured["args"][-2:] == ["--gpu-ids", "[1]"]
  # Exercise the upstream parser, not just our command string expectations.
  import mjlab
  import tyro
  from mjlab.scripts.train import TrainConfig

  parsed = tyro.cli(
    TrainConfig,
    default=TrainConfig.from_task("Wbc-G1"),
    args=captured["args"][-2:],
    config=mjlab.TYRO_FLAGS,
  )
  assert parsed.gpu_ids == [1]
  assert captured["meta"]["gpu_uuids"] == ["GPU-b"]
  with pytest.raises(ValueError, match="instead of overriding"):
    core.training.launch("Wbc-G1", "samples", gpu_uuids=["GPU-b"], extra="--gpu-ids 0")
  monkeypatch.setattr(
    gpu,
    "visible_gpus",
    lambda: [
      {"uuid": "GPU-a", "cuda_index": 0},
      {"uuid": "GPU-b", "cuda_index": 1},
    ],
  )
  core.training.launch("Wbc-G1", "samples", gpu_uuids=["GPU-a", "GPU-b"])
  parsed = tyro.cli(
    TrainConfig,
    default=TrainConfig.from_task("Wbc-G1"),
    args=captured["args"][-2:],
    config=mjlab.TYRO_FLAGS,
  )
  assert parsed.gpu_ids == [0, 1]
  with pytest.raises(ValueError, match="probe_interval=0"):
    core.training.launch(
      "Wbc-G1", "samples", algo="wm", recipe="wm-blink", gpu_uuids=["GPU-a", "GPU-b"]
    )


@pytest.mark.parametrize(
  "argument,expected",
  [("[2]", [2]), ("[0, 2]", [0, 2]), ("all", "all"), ("None", None)],
)
def test_world_model_cli_parses_gpu_selection(monkeypatch, argument, expected):
  """Parse the reported command end to end without starting a simulator or workers."""
  import sys

  import mjlab.scripts.train as upstream

  from wbc_mjlab.scripts.train import main

  captured = {}
  monkeypatch.setattr(
    upstream, "launch_training", lambda **kwargs: captured.update(kwargs)
  )
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "wbc-mjlab-train",
      "--task",
      "Wbc-G1-Lookahead",
      "--dataset",
      "samples",
      "--env.scene.num-envs",
      "16384",
      "--agent.max-iterations",
      "30000",
      "--algo",
      "wm",
      "--recipe",
      "wm-supermimic",
      "--env.commands.motion.mirror",
      "True",
      "--gpu-ids",
      argument,
    ],
  )
  main()
  assert captured["task_id"] == "Wbc-G1-Lookahead"
  assert captured["args"].gpu_ids == expected
  assert captured["args"].env.scene.num_envs == 16384
  assert captured["args"].agent.algorithm.dynamics_mode == "velocity"
