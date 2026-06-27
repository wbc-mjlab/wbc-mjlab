"""Emit a wbc-demo live-policy folder (issue wbc-mjlab-ow5, live-engine pivot).

``wbc-mjlab-export-web-reference`` assembles a ready-to-drop ``policies/<id>/``
folder for the wbc-demo live engine from a trained run + a motion dataset::

  <out>/<policy-id>/
    policy.onnx            # reused from the run's params/ (policy-only export)
    config.yaml            # deploy obs/action config (wbc_tracking_params_v1)
    motion_library.yaml    # clip-library provenance (wbc_motion_library_v1)
    reference/index.json   # wbc_reference_stream_v1 clip listing + term layout
    reference/<clipId>.bin # per-clip reference command stream (frames x 39)
    thumb.png              # gallery thumbnail (placeholder)
    policy.yaml            # manifest stub (conforms to policy.schema.json)

Unlike the shelved render-pose exporter, this ships ONLY the compact per-frame
**reference command** (39 floats: ref_base_height + ref_base_lin_vel_b +
ref_base_ang_vel_b + ref_gravity_b + ref_joint_pos). The browser runs the policy
ONNX live and concatenates this reference stream with proprioception from its own
sim to form the full actor observation. See :mod:`wbc_mjlab.export.web_reference`
for the wire format and the math (matched to mjlab's MotionCommand and the deploy
C++ WbcMotionLoader).

Run via the console script or as a module::

  uv run wbc-mjlab-export-web-reference --task Wbc-G1 --dataset samples --out OUT
  python -m wbc_mjlab.export.web_reference_bundle --task Wbc-G1 --dataset samples --out OUT
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wbc_mjlab.deploy_paths import (
  PLAY_CONFIG_NAME,
  PLAY_MOTION_LIBRARY_MANIFEST,
  PLAY_PARAMS_SUBDIR,
  PLAY_POLICY_ONNX_NAME,
)
from wbc_mjlab.export.web_reference import (
  REFERENCE_SUBDIR,
  ReferenceClip,
  command_dim,
  expected_bin_bytes,
  reference_command_from_npz,
  write_reference_bin,
  write_reference_index,
)
from wbc_mjlab.motion.manifest import (
  MANIFEST_SCHEMA,
  build_motion_library_manifest,
  write_motion_library_manifest,
)
from wbc_mjlab.motion.stack_bundle import list_clip_npz_files
from wbc_mjlab.robots.ids import RobotId
from wbc_mjlab.tasks import (
  get_task_config,
  prepare_wbc_run,
  resolve_task_id,
  robot_id_for_run,
)

THUMB_NAME = "thumb.png"
MANIFEST_NAME = "policy.yaml"
MANIFEST_SCHEMA_VERSION = "1"
_DEFAULT_TAGS = ("locomotion",)
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Heuristic clip tags from clip id, for gallery filtering.
_TAG_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
  ("walk", ("walk", "locomotion")),
  ("run", ("run", "locomotion")),
  ("sprint", ("sprint", "run", "locomotion")),
  ("jump", ("jump",)),
  ("flip", ("flip",)),
  ("fight", ("fight",)),
  ("sport", ("sport",)),
  ("dance", ("dance",)),
  ("getup", ("getup",)),
  ("fallandgetup", ("getup",)),
  ("crawl", ("crawl",)),
)


def slugify(value: str) -> str:
  """Lowercase kebab-case slug matching the policy.schema.json id pattern."""
  slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
  return slug or "policy"


def clip_tags_from_id(clip_id: str) -> tuple[str, ...]:
  """Infer skill tags from a clip id (e.g. ``walk1_subject1`` -> walk, locomotion)."""
  lowered = clip_id.lower()
  tags: list[str] = []
  for keyword, mapped in _TAG_KEYWORDS:
    if keyword in lowered:
      for tag in mapped:
        if tag not in tags:
          tags.append(tag)
  return tuple(tags)


def clip_display_name(clip_id: str) -> str:
  """Human label from a clip id (``walk1_subject1`` -> ``Walk1 Subject1``)."""
  return clip_id.replace("_", " ").replace("-", " ").strip().title() or clip_id


def _robot_xml_path(robot_id: RobotId) -> Path:
  if robot_id == "g1":
    from wbc_mjlab.robots.g1.constants import G1_XML

    return G1_XML
  raise ValueError(f"No default MuJoCo XML for robot {robot_id!r}")


def robot_body_names_from_xml(robot_id: RobotId) -> list[str]:
  """Full robot body order matching the NPZ ``body_*_w`` columns.

  This is the MJCF body order excluding the MuJoCo ``world`` body, i.e. model
  body ids 1..nbody-1, which equals mjlab ``robot.body_names`` (the order the
  conversion logged ``robot.data.body_link_pos_w``).
  """
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(_robot_xml_path(robot_id)))
  names: list[str] = []
  for body_id in range(1, model.nbody):  # skip world (id 0)
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    names.append(name if name is not None else f"body_{body_id}")
  return names


def _resolve_run_dir(
  *,
  run: str | None,
  checkpoint_file: str | None,
) -> Path | None:
  """Locate the run folder that holds ``params/`` (ONNX + config), if any."""
  candidates: list[Path] = []
  if checkpoint_file is not None:
    candidates.append(Path(checkpoint_file).expanduser().resolve().parent)
  if run is not None:
    run_path = Path(run).expanduser().resolve()
    if run_path.is_file():
      candidates.append(run_path.parent)
    else:
      candidates.append(run_path)
  for cand in candidates:
    if (cand / PLAY_PARAMS_SUBDIR).is_dir() or (cand / PLAY_POLICY_ONNX_NAME).is_file():
      return cand
  return candidates[0] if candidates else None


def _find_artifact(run_dir: Path | None, name: str) -> Path | None:
  """Find ``name`` under the run dir (``params/`` first, then the dir itself)."""
  if run_dir is None:
    return None
  for cand in (run_dir / PLAY_PARAMS_SUBDIR / name, run_dir / name):
    if cand.is_file():
      return cand
  return None


def _resolve_dataset_clips(
  *,
  robot_id: RobotId,
  dataset: str | None,
  dataset_path: str | None,
  motion_file: str | None,
) -> tuple[list[Path], Path, str]:
  """Resolve per-clip NPZs, the dataset root, and the dataset name."""
  from wbc_mjlab.data_paths import resolve_dataset_root

  root: Path
  if motion_file is not None:
    src = Path(motion_file).expanduser().resolve()
    root = src.parent.parent if src.parent.name == "npz" else src.parent
  elif dataset_path is not None:
    p = Path(dataset_path).expanduser().resolve()
    root = p.parent if p.is_file() else p
  elif dataset is not None:
    root = resolve_dataset_root(robot_id, dataset)
  else:
    raise ValueError("Provide --dataset, --dataset-path, or --motion-file")

  clips = list_clip_npz_files(root)
  if not clips:
    raise FileNotFoundError(
      f"No per-clip NPZs under {root / 'npz'}. Convert first: "
      f"wbc-mjlab-data-to-npz --robot {robot_id} --dataset <name>"
    )
  return clips, root, root.name


def _export_config_yaml(out_path: Path, *, task_id: str, robot_id: RobotId) -> None:
  """Generate ``config.yaml`` from the task env cfg (no checkpoint needed)."""
  from wbc_mjlab.export.policy_bundle import actor_has_state_estimation
  from wbc_mjlab.export.tracking_params_yaml import write_wbc_tracking_params_yaml
  from wbc_mjlab.robots.env import make_wbc_env_cfg

  cfg = make_wbc_env_cfg(robot_id, play=True, task_id=task_id)
  write_wbc_tracking_params_yaml(
    out_path,
    cfg,
    robot_id=robot_id,
    has_state_estimation=actor_has_state_estimation(cfg),
  )


def _read_config_reference_layout(config_path: Path) -> tuple[list[str], str]:
  """Read (joint_names, anchor_body_name) from a ``config.yaml`` (authoritative)."""
  doc = yaml.safe_load(config_path.read_text(encoding="utf-8"))
  joint_names = [str(x) for x in doc["joint_names"]]
  anchor = str(doc["tracking"]["anchor_body_name"])
  return joint_names, anchor


def _write_placeholder_png(path: Path, *, size: int = 256) -> None:
  """Write a small solid-color PNG placeholder using only the stdlib."""
  width = height = size
  r, g, b = 24, 28, 36  # wbc-demo dark card background
  row = b"\x00" + bytes((r, g, b)) * width
  raw = row * height
  compressed = zlib.compress(raw, 9)

  def chunk(tag: bytes, data: bytes) -> bytes:
    return (
      struct.pack(">I", len(data))
      + tag
      + data
      + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )

  ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
  png = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", ihdr)
    + chunk(b"IDAT", compressed)
    + chunk(b"IEND", b"")
  )
  path.write_bytes(png)


def _build_manifest(
  *,
  robot_id: RobotId,
  task_id: str,
  default_clip: str,
  has_onnx: bool,
) -> dict[str, Any]:
  """Manifest stub conforming to policy.schema.json (kind: manifest clips)."""
  task = get_task_config(task_id)
  manifest: dict[str, Any] = {
    "schemaVersion": MANIFEST_SCHEMA_VERSION,
    "name": f"{robot_id.upper()} {task_id}",
    "description": task.description,
    "robot": robot_id,
    "tags": list(_DEFAULT_TAGS),
    "author": "wbc-mjlab",
    "links": {"code": "https://github.com/wbc-mjlab/wbc-mjlab"},
    "thumbnail": THUMB_NAME,
    "defaultClip": default_clip,
    "camera": "default",
  }
  # The live engine needs the policy graph + obs config + the reference stream.
  # Clips point at the motion library manifest (which carries the clip ids that
  # also key reference/index.json), matching the schema's "manifest" clip form.
  artifacts: dict[str, Any] = {
    "config": PLAY_CONFIG_NAME,
    "clips": {
      "kind": "manifest",
      "file": PLAY_MOTION_LIBRARY_MANIFEST,
      "default": default_clip,
    },
  }
  # ``artifacts.onnx`` is required when the block is present; only emit the block
  # if we actually have the policy graph (the live engine needs it).
  if has_onnx:
    manifest["artifacts"] = {"onnx": PLAY_POLICY_ONNX_NAME, **artifacts}
  return manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
  header = (
    "# wbc-demo policy manifest (schemaVersion 1) — edit metadata before commit\n"
  )
  path.write_text(
    header + yaml.safe_dump(manifest, sort_keys=False, default_flow_style=None),
    encoding="utf-8",
  )


def export_web_reference(
  *,
  out: Path,
  task_id: str,
  robot_id: RobotId,
  dataset: str | None = None,
  dataset_path: str | None = None,
  motion_file: str | None = None,
  run: str | None = None,
  checkpoint_file: str | None = None,
  policy_id: str | None = None,
) -> Path:
  """Assemble a wbc-demo live-policy folder; returns the policy folder path."""
  clip_paths, dataset_root, dataset_name = _resolve_dataset_clips(
    robot_id=robot_id,
    dataset=dataset,
    dataset_path=dataset_path,
    motion_file=motion_file,
  )

  pid = slugify(policy_id) if policy_id else slugify(f"{robot_id}-{dataset_name}")
  policy_dir = out.expanduser().resolve() / pid
  policy_dir.mkdir(parents=True, exist_ok=True)
  reference_dir = policy_dir / REFERENCE_SUBDIR
  reference_dir.mkdir(parents=True, exist_ok=True)

  # ---- policy.onnx + config.yaml first (config defines the reference layout) ----
  run_dir = _resolve_run_dir(run=run, checkpoint_file=checkpoint_file)
  onnx_src = _find_artifact(run_dir, PLAY_POLICY_ONNX_NAME)
  has_onnx = onnx_src is not None
  if has_onnx:
    assert onnx_src is not None
    shutil.copyfile(onnx_src, policy_dir / PLAY_POLICY_ONNX_NAME)
    print(f"[INFO] policy.onnx <- {onnx_src}")
  else:
    print(
      "[WARN] No policy.onnx found in the run dir; emitting a metadata-only card "
      "(no `artifacts` block). The live engine needs the ONNX — pass --run."
    )

  config_dst = policy_dir / PLAY_CONFIG_NAME
  config_src = _find_artifact(run_dir, PLAY_CONFIG_NAME)
  if config_src is not None:
    shutil.copyfile(config_src, config_dst)
    print(f"[INFO] config.yaml <- {config_src}")
  else:
    _export_config_yaml(config_dst, task_id=task_id, robot_id=robot_id)
    print("[INFO] config.yaml generated from task env cfg")

  config_joint_names, anchor_body_name = _read_config_reference_layout(config_dst)
  robot_body_names = robot_body_names_from_xml(robot_id)
  dim = command_dim(config_joint_names)
  print(
    f"[INFO] robot={robot_id} anchor={anchor_body_name!r} "
    f"joints={len(config_joint_names)} commandDim={dim}; clips={len(clip_paths)}"
  )

  # ---- reference/<clip>.bin + reference/index.json (wbc_reference_stream_v1) ----
  ref_clips: list[ReferenceClip] = []
  fps_value = 50.0
  total_bytes = 0
  for clip_path in clip_paths:
    clip_id = clip_path.stem
    with np.load(clip_path, allow_pickle=True) as data:
      fps_value = float(np.asarray(data["fps"], dtype=np.float64).reshape(-1)[0])
      command = reference_command_from_npz(
        data,
        robot_body_names=robot_body_names,
        anchor_body_name=anchor_body_name,
        config_joint_names=config_joint_names,
      )
    frames = int(command.shape[0])
    bin_name = f"{clip_id}.bin"
    written = write_reference_bin(reference_dir / bin_name, command)
    expected = expected_bin_bytes(frames, dim)
    if written != expected:
      raise RuntimeError(
        f"{bin_name}: wrote {written} bytes, expected {expected} "
        f"(frames={frames}, dim={dim})"
      )
    total_bytes += written
    ref_clips.append(
      ReferenceClip(
        id=clip_id,
        name=clip_display_name(clip_id),
        file=bin_name,
        frames=frames,
        duration_sec=frames / fps_value if fps_value else 0.0,
        tags=clip_tags_from_id(clip_id) or _DEFAULT_TAGS,
      )
    )
    print(f"[INFO]   {bin_name}: {frames} frames, {written} bytes")

  write_reference_index(
    reference_dir,
    robot_id=robot_id,
    fps=fps_value,
    joint_names=config_joint_names,
    clips=ref_clips,
  )

  # ---- motion_library.yaml (provenance; clip form B) ----
  manifest_doc = build_motion_library_manifest(dataset_root)
  write_motion_library_manifest(policy_dir / PLAY_MOTION_LIBRARY_MANIFEST, manifest_doc)
  assert manifest_doc["schema"] == MANIFEST_SCHEMA

  # ---- thumbnail ----
  _write_placeholder_png(policy_dir / THUMB_NAME)

  # ---- policy.yaml manifest stub ----
  default_clip = ref_clips[0].id if ref_clips else dataset_name
  manifest = _build_manifest(
    robot_id=robot_id,
    task_id=task_id,
    default_clip=default_clip,
    has_onnx=has_onnx,
  )
  _write_manifest(policy_dir / MANIFEST_NAME, manifest)

  print(
    f"[INFO] Wrote {len(ref_clips)} reference clip(s), {total_bytes} bytes total "
    f"-> {policy_dir}"
  )
  return policy_dir


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--out", type=Path, required=True, help="parent dir for <policy-id>/"
  )
  parser.add_argument(
    "--task",
    default=None,
    help="WBC task id (e.g. Wbc-G1). Overrides --no-state-estimation.",
  )
  parser.add_argument("--robot", default=None, help="optional; inferred from --task")
  parser.add_argument("--no-state-estimation", action="store_true")
  parser.add_argument(
    "--dataset", default=None, help="dataset name under data/<robot>/"
  )
  parser.add_argument("--dataset-path", default=None, help="dataset dir or .npz path")
  parser.add_argument("--motion-file", default=None, help="explicit clip NPZ or dir")
  parser.add_argument(
    "--run",
    default=None,
    help="run dir (or checkpoint) holding params/ (onnx + config)",
  )
  parser.add_argument(
    "--checkpoint-file", default=None, help="checkpoint .pt; its dir holds params/"
  )
  parser.add_argument(
    "--policy-id", default=None, help="folder slug (default: <robot>-<dataset>)"
  )
  args = parser.parse_args()

  task_id = resolve_task_id(
    args.robot, task=args.task, no_state_estimation=args.no_state_estimation
  )
  prepare_wbc_run(task_id=task_id)
  robot_id = robot_id_for_run(
    task_id=task_id, robot_id=args.robot, robot_explicit=args.robot is not None
  )

  policy_dir = export_web_reference(
    out=args.out,
    task_id=task_id,
    robot_id=robot_id,
    dataset=args.dataset,
    dataset_path=args.dataset_path,
    motion_file=args.motion_file,
    run=args.run,
    checkpoint_file=args.checkpoint_file,
    policy_id=args.policy_id,
  )
  print(f"Exported web reference policy -> {policy_dir}")


if __name__ == "__main__":
  sys.exit(main())
