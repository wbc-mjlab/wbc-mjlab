"""Data service: the motion datasets of the selected robot, from source files to training NPZ.

* **Datasets** — every ``data/<robot>/<name>/`` folder across project roots (this repo and
  registered extensions): source clips (``*.csv`` / ``*.pkl`` at the root or in ``raw/``) and
  converted clips (``npz/*.npz``), matched by name.
* **Browser** — paged, ``*``-wildcard search over thousands of clips; per-row frames / fps /
  conversion state read from file headers only.
* **Preview** — one clip on the robot (converted NPZ, or the raw source *before* conversion),
  with QA (joint limits, speed spikes, foot penetration …), curves, the root path, and the raw
  source as a ghost over the converted clip to see what conversion changed. **Grid** plays the
  first N clips of the current search side by side.
* **Conversion / import** — ``wbc-mjlab-data-to-npz`` as a detached job on all, missing or
  selected clips (via a folder of symlinks), and source clips imported from a folder (copied
  or linked) or uploaded from the browser.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from wbc_mjlab.studio import clips as clipmod
from wbc_mjlab.studio import jobs as jobmod
from wbc_mjlab.studio.core import CONTROL_FPS
from wbc_mjlab.studio.scene import Overlay, grid_offsets

OWNER = "data"
PATH_COLOR = np.array([54, 207, 224], np.uint8)
FOOT_DOWN = np.array([79, 199, 138], np.uint8)
FOOT_UP = np.array([143, 160, 179], np.uint8)
MAX_GRID = 49
NAME_RE = re.compile(r"^[\w][\w.-]*$")


def _dir_sig(*dirs: Path) -> tuple:
  sig = []
  for d in dirs:
    try:
      sig.append(d.stat().st_mtime_ns)
    except OSError:
      sig.append(0)
  return tuple(sig)


class DataService:
  def __init__(self, core):
    self.c = core
    self.dataset = ""
    self._index_cache: dict[str, tuple[tuple, dict]] = {}
    self._header_cache: dict[tuple[str, int], dict] = {}
    self._lock = threading.RLock()
    # preview
    self.clips: list[clipmod.Clip] = []  # 1 (single) or N (grid)
    self.compare: clipmod.Clip | None = (
      None  # raw source drawn as a ghost over a converted clip
    )
    self.mode = "single"
    self.t = 0.0
    self.playing = False
    self.speed = 1.0
    self.loop = True
    self.show_compare = True
    self.show_path = True
    self.show_feet = True
    self.qa: dict = {}

  # ------------------------------------------------------------------ datasets
  def on_robot(self) -> None:
    with self._lock:
      self.clips, self.compare, self.qa = [], None, {}
      names = [d["name"] for d in self.datasets()]
      self.dataset = "samples" if "samples" in names else (names[0] if names else "")

  def _root(self, dataset: str) -> Path:
    from wbc_mjlab.data_paths import resolve_dataset_root

    return resolve_dataset_root(self.c.robot_id, dataset)

  def _sources_dir(self, root: Path) -> Path:
    raw = root / "raw"
    if raw.is_dir() and any(
      p.suffix.lower() in clipmod.RAW_EXTENSIONS for p in raw.iterdir()
    ):
      return raw
    return root

  def index(self, dataset: str) -> dict[str, dict]:
    """``clip name → {"raw": path | None, "npz": path | None}`` (cached on folder mtimes)."""
    root = self._root(dataset)
    src, npz = self._sources_dir(root), root / "npz"
    sig = (str(root), *_dir_sig(root, src, npz))
    cached = self._index_cache.get(dataset)
    if cached and cached[0] == sig:
      return cached[1]
    idx: dict[str, dict] = {}
    if src.is_dir():
      with os.scandir(src) as it:
        for e in it:
          stem, ext = os.path.splitext(e.name)
          if ext.lower() in clipmod.RAW_EXTENSIONS and e.is_file():
            idx.setdefault(stem, {"raw": None, "npz": None})["raw"] = Path(e.path)
    if npz.is_dir():
      with os.scandir(npz) as it:
        for e in it:
          stem, ext = os.path.splitext(e.name)
          if ext == ".npz" and e.is_file():
            idx.setdefault(stem, {"raw": None, "npz": None})["npz"] = Path(e.path)
    idx = dict(sorted(idx.items()))
    self._index_cache[dataset] = (sig, idx)
    return idx

  def datasets(self) -> list[dict]:
    from wbc_mjlab.data_paths import list_datasets

    out = []
    for name in list_datasets(self.c.robot_id):
      try:
        idx = self.index(name)
      except OSError:
        continue
      raw = sum(1 for v in idx.values() if v["raw"])
      npz = sum(1 for v in idx.values() if v["npz"])
      root = self._root(name)
      out.append(
        {
          "name": name,
          "path": str(root),
          "clips": len(idx),
          "raw": raw,
          "npz": npz,
          "missing": sum(1 for v in idx.values() if v["raw"] and not v["npz"]),
          "linked": root.is_symlink(),
          "stats": self._load_stats(name),
        }
      )
    return out

  # ------------------------------------------------------------------ browsing
  def _header(self, path: Path) -> dict:
    try:
      mtime = path.stat().st_mtime_ns
    except OSError:
      return {}
    key = (str(path), mtime)
    if key not in self._header_cache:
      try:
        h = (
          clipmod.npz_header(path)
          if path.suffix == ".npz"
          else clipmod.raw_header(path)
        )
      except Exception as exc:  # noqa: BLE001
        h = {"error": str(exc)}
      if len(self._header_cache) > 50000:
        self._header_cache.clear()
      self._header_cache[key] = h
    return self._header_cache[key]

  @staticmethod
  def _match(names: list[str], q: str) -> list[str]:
    q = q.strip()
    if not q:
      return names
    if any(ch in q for ch in "*?["):
      return [n for n in names if fnmatch.fnmatch(n.lower(), q.lower())]
    terms = q.lower().split()
    return [n for n in names if all(t in n.lower() for t in terms)]

  def filtered(self, dataset: str, q: str = "", status: str = "all") -> list[str]:
    idx = self.index(dataset)
    names = self._match(list(idx), q)
    if status == "converted":
      names = [n for n in names if idx[n]["npz"]]
    elif status == "missing":
      names = [n for n in names if idx[n]["raw"] and not idx[n]["npz"]]
    elif status == "raw":
      names = [n for n in names if idx[n]["raw"]]
    return names

  def listing(
    self, dataset: str, q: str = "", status: str = "all", page: int = 0, size: int = 100
  ) -> dict:
    if dataset:
      self.dataset = dataset
    if not self.dataset:
      return {"dataset": "", "total": 0, "rows": [], "page": 0, "pages": 0}
    idx = self.index(self.dataset)
    names = self.filtered(self.dataset, q, status)
    pages = max(1, -(-len(names) // size))
    page = max(0, min(page, pages - 1))
    rows = []
    for n in names[page * size : (page + 1) * size]:
      e = idx[n]
      head = self._header(e["npz"] or e["raw"])
      rows.append(
        {
          "name": n,
          "raw": e["raw"].name if e["raw"] else None,
          "npz": bool(e["npz"]),
          "frames": head.get("frames"),
          "fps": head.get("fps"),
          "duration": head["frames"] / head["fps"]
          if head.get("frames") and head.get("fps")
          else None,
          "error": head.get("error"),
        }
      )
    return {
      "dataset": self.dataset,
      "path": str(self._root(self.dataset)),
      "total": len(names),
      "all": len(idx),
      "page": page,
      "pages": pages,
      "size": size,
      "rows": rows,
    }

  # ------------------------------------------------------------------ dataset stats (background)
  def _stats_path(self, dataset: str) -> Path:
    return jobmod.logs_dir() / "studio" / "index" / f"{self.c.robot_id}__{dataset}.json"

  def _load_stats(self, dataset: str) -> dict | None:
    p = self._stats_path(dataset)
    try:
      return json.loads(p.read_text()) if p.is_file() else None
    except Exception:  # noqa: BLE001
      return None

  def scan(self, dataset: str) -> bool:
    """Frames / duration of every clip of ``dataset`` (headers only) → a summary + histogram."""

    def job():
      idx = self.index(dataset)
      durations, fps_counts, errors = [], {}, 0
      names = list(idx)
      for i, n in enumerate(names):
        e = idx[n]
        h = self._header(e["npz"] or e["raw"])
        if h.get("frames") and h.get("fps"):
          durations.append(h["frames"] / h["fps"])
          key = f"{h['fps']:g}"
          fps_counts[key] = fps_counts.get(key, 0) + 1
        else:
          errors += 1
        if i % 200 == 0:
          self.c.progress(f"{i}/{len(names)} clips", i, len(names))
      d = np.asarray(durations or [0.0])
      edges = [0, 2, 5, 10, 20, 40, 80, 160, 1e9]
      hist = np.histogram(d, bins=edges)[0].tolist()
      stats = {
        "clips": len(names),
        "hours": float(d.sum() / 3600.0),
        "mean_s": float(d.mean()),
        "max_s": float(d.max()),
        "min_s": float(d.min()),
        "fps": fps_counts,
        "hist": {"edges": edges[:-1], "counts": hist},
        "errors": errors,
        "scanned": time.time(),
      }
      p = self._stats_path(dataset)
      p.parent.mkdir(parents=True, exist_ok=True)
      p.write_text(json.dumps(stats))
      self.c.log(
        f"{dataset}: {len(names)} clips, {stats['hours']:.2f} h, mean {stats['mean_s']:.1f} s",
        "ok",
      )

    return self.c.run_task(
      f"scan {dataset}", job, done=lambda: self.c.emit("data", dataset=dataset)
    )

  # ------------------------------------------------------------------ preview
  def _robot_feet(self) -> tuple[list[str], float]:
    try:
      from wbc_mjlab.motion.robot_assets import get_robot_motion_spec

      _, spec = get_robot_motion_spec(self.c.robot_id)
      return list(spec.foot_body_names or ()), float(spec.foot_sole_z or 0.0)
    except Exception:  # noqa: BLE001
      from wbc_mjlab.studio.robots import robot_wiring

      return list(robot_wiring(self.c.robot_id).get("feet", [])), 0.0

  def load(self, dataset: str, clip: str, source: str = "auto") -> None:
    """Preview ``clip``: the converted NPZ (``source="npz"``), the raw source (``"raw"``) or
    the NPZ when there is one (``"auto"``); a converted clip gets its raw source as ghost."""
    idx = self.index(dataset)
    if clip not in idx:
      raise ValueError(f"no clip {clip!r} in {dataset}")
    e = idx[clip]
    use = (
      source
      if source in ("npz", "raw") and e[source]
      else ("npz" if e["npz"] else "raw")
    )
    loaded = clipmod.load_clip(e[use], self.c)
    compare = None
    if use == "npz" and e["raw"]:
      try:
        compare = clipmod.load_clip(e["raw"], self.c)
      except Exception as exc:  # noqa: BLE001
        self.c.log(f"raw source of {clip}: {exc}", "warn")
    feet, sole = self._robot_feet()
    qa = clipmod.analyze(loaded, self.c.model, feet=feet, sole_z=sole)
    if compare is not None:
      n = min(compare.frames, int(compare.duration * loaded.fps))
      tq = np.minimum(
        (np.arange(n) / loaded.fps * compare.fps).round().astype(int),
        compare.frames - 1,
      )
      k = min(n, loaded.frames)
      dz = loaded.qpos[:k, 2] - compare.qpos[tq[:k], 2]
      qa["vs_raw"] = {
        "dz_mean": float(dz.mean()),
        "frames_raw": compare.frames,
        "fps_raw": compare.fps,
        "xy_offset": (loaded.qpos[0, 0:2] - compare.qpos[0, 0:2]).tolist(),
      }
    with self._lock:
      self.dataset = dataset
      self.clips, self.compare, self.qa = [loaded], compare, qa
      self.mode, self.t = "single", 0.0
      self.playing = True
      self.c.take_scene(OWNER)
      self._static_overlays()
      self._render()
    self.c.emit("data", preview=self.state())
    self.c.log(f"preview {clip} ({use}): {loaded.frames} frames @ {loaded.fps:g} Hz")

  def load_grid(
    self, dataset: str, q: str = "", status: str = "all", count: int = 16
  ) -> bool:
    count = max(1, min(int(count), MAX_GRID))
    names = self.filtered(dataset, q, status)[:count]
    if not names:
      raise ValueError("no clips match")

    def job():
      idx = self.index(dataset)
      loaded = []
      for i, n in enumerate(names):
        e = idx[n]
        try:
          loaded.append(clipmod.load_clip(e["npz"] or e["raw"], self.c))
        except Exception as exc:  # noqa: BLE001
          self.c.log(f"{n}: {exc}", "warn")
        self.c.progress(f"{i + 1}/{len(names)} clips", i + 1, len(names))
      if not loaded:
        raise RuntimeError("no clip could be loaded")
      with self._lock:
        self.dataset = dataset
        self.clips, self.compare, self.qa = loaded, None, {}
        self.mode, self.t, self.playing = "grid", 0.0, True
        self.c.take_scene(OWNER)
        self._static_overlays()
        self._render()
      self.c.emit("data", preview=self.state())

    return self.c.run_task(f"grid of {len(names)} clips", job)

  def close(self) -> None:
    with self._lock:
      self.clips, self.compare, self.qa = [], None, {}
      self.playing = False
    self.c.scene.remove_overlays(prefix=f"{OWNER}/")
    self.c.robots.show_pose()
    self.c.emit("data", preview=self.state())

  def state(self) -> dict:
    c0 = self.clips[0] if self.clips else None
    return {
      "active": bool(self.clips),
      "mode": self.mode,
      "dataset": self.dataset,
      "clip": c0.name if c0 and self.mode == "single" else "",
      "kind": c0.kind if c0 else "",
      "path": str(c0.path) if c0 and self.mode == "single" else "",
      "count": len(self.clips),
      "names": [c.name for c in self.clips] if self.mode == "grid" else [],
      "frames": c0.frames if c0 else 0,
      "fps": c0.fps if c0 else 0.0,
      "duration": self._duration(),
      "playing": self.playing,
      "speed": self.speed,
      "loop": self.loop,
      "has_compare": self.compare is not None,
      "show_compare": self.show_compare,
      "show_path": self.show_path,
      "show_feet": self.show_feet,
      "qa": self.qa,
    }

  def clip_series(self, keys: list[str]) -> dict:
    if not self.clips or self.mode != "single":
      return {"t": [], "series": {}}
    feet, sole = self._robot_feet()
    return clipmod.series(self.clips[0], self.c.model, keys, feet=feet, sole_z=sole)

  def joint_names(self) -> list[str]:
    from wbc_mjlab.studio.clips import _hinges

    return [h[0] for h in _hinges(self.c.model)] if self.c.model is not None else []

  # ------------------------------------------------------------------ transport
  def _duration(self) -> float:
    return max((c.duration for c in self.clips), default=0.0)

  def set_frame(self, frame: int) -> None:
    with self._lock:
      if not self.clips:
        return
      self.t = max(0, int(frame)) / self.clips[0].fps
      self.c.take_scene(OWNER)
      self._render()

  def set_time(self, t: float) -> None:
    with self._lock:
      self.t = float(np.clip(t, 0.0, self._duration()))
      self.c.take_scene(OWNER)
      self._render()

  def set_playing(self, playing: bool) -> None:
    with self._lock:
      self.playing = bool(playing) and bool(self.clips)
      if self.playing:
        if self.t >= self._duration() - 1e-6 and self.mode == "single":
          self.t = 0.0
        self.c.take_scene(OWNER)
        self._static_overlays()
    self.c.emit("data", preview=self.state())

  def set_options(self, **opts) -> None:
    with self._lock:
      for k in ("speed", "loop", "show_compare", "show_path", "show_feet"):
        if opts.get(k) is not None:
          setattr(self, k, type(getattr(self, k))(opts[k]))
      if self.clips:
        self._static_overlays()
        self._render()
    self.c.emit("data", preview=self.state())

  def tick(self, dt: float) -> None:
    if not self.playing or not self.clips or self.c.owner != OWNER:
      return
    with self._lock:
      if not self.clips:
        return
      self.t += dt * self.speed
      dur = self._duration()
      if self.mode == "single" and self.t >= dur:
        if self.loop:
          self.t %= max(dur, 1e-6)
        else:
          self.t, self.playing = dur, False
          self.c.emit("data", preview=self.state())
      self._render()

  # ------------------------------------------------------------------ rendering
  def _grid_offsets(self) -> np.ndarray:
    """Per clip: move its first-frame root onto its grid cell."""
    offs = grid_offsets(len(self.clips), spacing=1.8)
    starts = np.array(
      [[c.qpos[0, 0], c.qpos[0, 1], 0.0] for c in self.clips], np.float32
    )
    return offs - starts

  @staticmethod
  def _recenter(clip: clipmod.Clip) -> np.ndarray:
    """Offset putting the clip's first-frame root on the world origin (xy). Converted NPZs
    carry the conversion pool's env origin in ``body_pos_w``; the raw source does not."""
    return np.array([-clip.qpos[0, 0], -clip.qpos[0, 1], 0.0], np.float32)

  def _static_overlays(self) -> None:
    """Root path of the clip (drawn once per load, not per frame)."""
    if self.mode == "single" and self.clips and self.show_path:
      root = self.clips[0].qpos[:, 0:3].astype(np.float32) + self._recenter(
        self.clips[0]
      )
      root[:, 2] = 0.004
      step = max(1, len(root) // 400)
      pts = root[::step]
      if len(pts) > 1:
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        self.c.overlay(
          OWNER, "path", Overlay("segments", segs, PATH_COLOR, 0.012, opacity=0.65)
        )
        return
    self.c.overlay(OWNER, "path", None)

  def _render(self) -> None:
    if not self.clips or self.c.owner != OWNER:
      return
    if self.mode == "single":
      clip = self.clips[0]
      f = clip.frame_at(self.t)
      pos, quat = clip.poses(f, self._recenter(clip))
      ghost = None
      if self.compare is not None and self.show_compare:
        gf = self.compare.frame_at(self.t)
        gp, gq = self.compare.poses(gf, self._recenter(self.compare))
        ghost = (gp[None], gq[None])
      self.c.scene.set_poses(pos[None], quat[None], ghost=ghost, labels=[])
      self._feet_overlay(clip, f)
      hud = {
        "owner": OWNER,
        "title": clip.name,
        "frame": f,
        "frames": clip.frames,
        "time": f / clip.fps,
        "duration": clip.duration,
        "playing": self.playing,
        "fps": clip.fps,
        "kind": clip.kind,
        "ghost": ghost is not None,
      }
    else:
      offs = self._grid_offsets()
      pos, quat, labels = [], [], []
      for clip, off in zip(self.clips, offs, strict=True):
        f = clip.frame_at(self.t % max(clip.duration, 1e-6))
        p, q = clip.poses(f, off)
        pos.append(p)
        quat.append(q)
        labels.append(clip.name)
      self.c.scene.set_poses(np.stack(pos), np.stack(quat), labels=labels)
      self.c.overlay(OWNER, "feet", None)
      dur = self._duration()
      hud = {
        "owner": OWNER,
        "title": f"{len(self.clips)} clips · {self.dataset}",
        "frame": int(self.t * CONTROL_FPS),
        "frames": int(dur * CONTROL_FPS) + 1,
        "time": self.t,
        "duration": dur,
        "playing": self.playing,
        "grid": len(self.clips),
      }
    self.c.scene.set_hud(hud)

  def _feet_overlay(self, clip: clipmod.Clip, frame: int) -> None:
    if not self.show_feet:
      self.c.overlay(OWNER, "feet", None)
      return
    feet, sole = self._robot_feet()
    ids = [i for i in (self._body_id(n) for n in feet) if i >= 0]
    if not ids:
      return
    p = clip.body_pos[frame, ids] + self._recenter(clip)
    down = (p[:, 2] - sole) < 0.03
    colors = np.where(down[:, None], FOOT_DOWN, FOOT_UP).astype(np.uint8)
    p[:, 2] = 0.006
    self.c.overlay(OWNER, "feet", Overlay("points", p, colors, 0.07, opacity=0.9))

  def _body_id(self, name: str) -> int:
    import mujoco

    return mujoco.mj_name2id(self.c.model, mujoco.mjtObj.mjOBJ_BODY, name)

  # ------------------------------------------------------------------ conversion
  def convert(
    self,
    dataset: str,
    which: str = "missing",
    clips: list[str] | None = None,
    *,
    batch_size: int = 8,
    output_fps: float = 50.0,
    mirror: bool = False,
    debias_z: bool = False,
    input_fps: float | None = None,
  ) -> jobmod.Job:
    """Run ``wbc-mjlab-data-to-npz`` on ``all`` / ``missing`` / ``selected`` source clips of
    ``dataset`` as a detached job (a subset goes through a folder of symlinks)."""
    if mirror:
      from wbc_mjlab.robots.symmetry import get_robot_symmetry_config

      if self.c.robot_id != "g1" and get_robot_symmetry_config(self.c.robot_id) is None:
        raise ValueError(
          f"mirroring needs a left/right symmetry config for {self.c.robot_id} "
          "(RobotSymmetryConfig in an extension): convert without mirror"
        )
    idx = self.index(dataset)
    root = self._root(dataset)
    if which == "all":
      chosen = [n for n, e in idx.items() if e["raw"]]
    elif which == "missing":
      chosen = [n for n, e in idx.items() if e["raw"] and not e["npz"]]
    else:
      chosen = [n for n in (clips or []) if n in idx and idx[n]["raw"]]
    if not chosen:
      raise ValueError("nothing to convert (no source clips selected / missing)")
    running = [
      j
      for j in jobmod.list_jobs("convert")
      if j.alive and j.meta.get("dataset") == dataset
    ]
    if running:
      raise RuntimeError(f"{dataset} is already being converted ({running[0].name})")
    if which == "all" and len(chosen) == sum(1 for e in idx.values() if e["raw"]):
      src = self._sources_dir(root)
    else:
      src = (
        jobmod.logs_dir()
        / "studio"
        / "convert"
        / f"{time.strftime('%Y%m%d-%H%M%S')}_{dataset}"
      )
      src.mkdir(parents=True, exist_ok=True)
      for n in chosen:
        target = idx[n]["raw"]
        link = src / target.name
        if not link.exists():
          link.symlink_to(target.resolve())
    args = [
      "wbc_mjlab.motion.data_to_npz",
      "--robot",
      self.c.robot_id,
      "--input-path",
      str(src),
      "--output-dir",
      str(root),
      "--batch-size",
      str(max(1, int(batch_size))),
      "--output-fps",
      f"{float(output_fps):g}",
      "--device",
      self.c.device,
    ]
    if input_fps:
      args += ["--input-fps", f"{float(input_fps):g}"]
    if mirror:
      args += ["--mirror", "True"]
    if debias_z:
      args += ["--debias-z", "True"]
    job = jobmod.launch(
      "convert",
      args,
      label=f"{self.c.robot_id}-{dataset}",
      meta={
        "robot": self.c.robot_id,
        "dataset": dataset,
        "clips": len(chosen),
        "which": which,
        "mirror": mirror,
        "debias_z": debias_z,
      },
    )
    self.c.log(f"converting {len(chosen)} clip(s) of {dataset} → {root / 'npz'}", "ok")
    self.c.emit("jobs")
    return job

  # ------------------------------------------------------------------ import / upload
  def _robot_data_dir(self) -> Path:
    from wbc_mjlab.data_paths import robot_data_dir

    return robot_data_dir(self.c.robot_id)

  def new_dataset(self, name: str) -> Path:
    if not NAME_RE.match(name or ""):
      raise ValueError("dataset names use letters, digits, '_', '-', '.'")
    root = self._robot_data_dir() / name
    root.mkdir(parents=True, exist_ok=True)
    self.c.emit("data", dataset=name)
    return root

  def import_folder(self, source: str, name: str, link: bool = False) -> bool:
    """Copy (or link, ``link=True``) the ``*.csv`` / ``*.pkl`` clips of ``source`` into
    ``data/<robot>/<name>/``. Linking the whole folder keeps it where it is."""
    src = Path(source).expanduser().resolve()
    if not src.is_dir():
      raise ValueError(f"not a folder: {src}")
    if not NAME_RE.match(name or ""):
      raise ValueError("dataset names use letters, digits, '_', '-', '.'")
    dest = self._robot_data_dir() / name
    if link:
      if dest.exists():
        raise ValueError(f"{dest} exists; pick another name to link a folder")
      dest.parent.mkdir(parents=True, exist_ok=True)
      dest.symlink_to(src, target_is_directory=True)
      self.c.log(f"linked {src} as dataset {name}", "ok")
      self.c.emit("data", dataset=name)
      return True

    files = sorted(
      p
      for p in src.rglob("*")
      if p.is_file() and p.suffix.lower() in clipmod.RAW_EXTENSIONS
    )
    if not files:
      raise ValueError(f"no .csv / .pkl clips under {src}")

    def job():
      dest.mkdir(parents=True, exist_ok=True)
      for i, p in enumerate(files):
        shutil.copy2(p, dest / p.name)
        if i % 50 == 0:
          self.c.progress(f"{i}/{len(files)} files", i, len(files))
      self.c.log(f"imported {len(files)} clip(s) into {dest}", "ok")

    return self.c.run_task(
      f"import {name}", job, done=lambda: self.c.emit("data", dataset=name)
    )

  def upload(self, dataset: str, filename: str, content: bytes) -> Path:
    name = Path(filename).name
    if Path(name).suffix.lower() not in clipmod.RAW_EXTENSIONS:
      raise ValueError(f"{name}: only .csv / .pkl source clips")
    root = (
      self.new_dataset(dataset)
      if dataset not in self._datasets_names()
      else self._root(dataset)
    )
    dest = root / name
    dest.write_bytes(content)
    return dest

  def _datasets_names(self) -> list[str]:
    from wbc_mjlab.data_paths import list_datasets

    return list_datasets(self.c.robot_id)


__all__ = ["DataService"]
