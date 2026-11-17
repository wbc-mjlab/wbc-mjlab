"""Motion dataset QA, diversity analysis and diversity-based subset selection.

PartyMimic (Roboparty, 2026) tools/data/bones_subset.py and tools/diversity, MIT; see
THIRD_PARTY_NOTICES. A clip is cut into 1 s windows (10 Hz, 0.5 s stride) of joint angles,
root height, gravity in the root frame, root velocity in the heading frame and yaw rate, so
lying, kneeling and standing with the same joint angles are apart. k-means over every
window of the pool gives motion clusters:

* **QA** flags clips with a body below the floor, a joint faster than a limit or a base
  jump between frames (bad retargeting).
* **Analysis** reports how evenly the windows fill the clusters (effective cluster count),
  the rarest and most redundant clips and, for a held-out set, the share of its windows a
  pool covers.
* **Selection** picks clips filling the clusters evenly: cluster ``k`` gets a target
  ``t_k ∝ n_k^tau`` (``tau < 1`` flattens the common motions) and the utility
  ``Σ_k t_k log(1 + m_k / t_k)`` of the selected windows ``m_k`` (submodular) is maximized
  greedily by gain per cost (clip seconds + ``lam``); a duration budget bisects ``lam``.

Selected or passing clips are written as a dataset folder of links to the source clips,
which training takes as ``--motion-file``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FEATURE_HZ, WINDOW, STRIDE = 10.0, 10, 5
W_JOINT, W_HEIGHT, W_GRAVITY, W_VELOCITY, W_YAW = 1.0, 2.0, 1.0, 0.5, 0.3


def _rotation(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return np.stack(
    [
      1 - 2 * (y * y + z * z),
      2 * (x * y - w * z),
      2 * (x * z + w * y),
      2 * (x * y + w * z),
      1 - 2 * (x * x + z * z),
      2 * (y * z - w * x),
      2 * (x * z - w * y),
      2 * (y * z + w * x),
      1 - 2 * (x * x + y * y),
    ],
    -1,
  ).reshape(-1, 3, 3)


def frame_features(qpos: np.ndarray, fps: float) -> np.ndarray:
  """Weighted per-frame features of robot coordinates ``(T, 7 + J)`` (wxyz base)."""
  qpos = np.asarray(qpos, np.float64)
  rot = _rotation(qpos[:, 3:7])
  yaw = np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))
  dt = 1.0 / fps
  edge = 2 if len(qpos) > 2 else 1
  vel = (
    np.gradient(qpos[:, :3], dt, axis=0, edge_order=edge)
    if len(qpos) > 1
    else 0 * qpos[:, :3]
  )
  yaw_rate = np.gradient(yaw, dt, edge_order=edge) if len(qpos) > 1 else 0 * yaw
  c, s = np.cos(yaw), np.sin(yaw)
  heading_vel = np.stack(
    [c * vel[:, 0] + s * vel[:, 1], -s * vel[:, 0] + c * vel[:, 1], vel[:, 2]], -1
  )
  gravity = -rot[:, 2, :]  # world -z in the root frame
  return np.concatenate(
    [
      W_JOINT * qpos[:, 7:],
      W_HEIGHT * qpos[:, 2:3],
      W_GRAVITY * gravity,
      W_VELOCITY * heading_vel,
      W_YAW * yaw_rate[:, None],
    ],
    -1,
  )


def clip_windows(qpos: np.ndarray, fps: float) -> np.ndarray:
  """``(n_windows, WINDOW · D)`` float32 windows of one clip (none when it is shorter)."""
  f = frame_features(qpos, fps)
  f = f[:: max(1, round(fps / FEATURE_HZ))]
  if f.shape[0] < WINDOW:
    return np.zeros((0, WINDOW * f.shape[1]), np.float32)
  idx = np.arange(0, f.shape[0] - WINDOW + 1, STRIDE)[:, None] + np.arange(WINDOW)
  return f[idx].reshape(len(idx), -1).astype(np.float32)


# ------------------------------------------------------------------------------- k-means
def nearest(x: torch.Tensor, centers: torch.Tensor, chunk: int = 65536):
  """Index of and squared distance to the nearest center of each row."""
  c2 = (centers * centers).sum(1)
  labels, dists = [], []
  for i in range(0, len(x), chunk):
    part = x[i : i + chunk]
    d = (part * part).sum(1, keepdim=True) - 2 * part @ centers.T + c2
    best = d.min(1)
    labels.append(best.indices)
    dists.append(best.values.clamp_min(0))
  return torch.cat(labels), torch.cat(dists)


def kmeans(x: torch.Tensor, k: int, iters: int = 30, seed: int = 0):
  """Centers and labels of k-means (seeded random initialization)."""
  k = min(k, len(x))
  g = torch.Generator(device="cpu").manual_seed(seed)
  centers = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
  for _ in range(iters):
    labels, _ = nearest(x, centers)
    sums = torch.zeros_like(centers).index_add_(0, labels, x)
    counts = torch.bincount(labels, minlength=k).to(x.dtype)
    centers = torch.where(
      counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centers
    )
  return centers, nearest(x, centers)[0]


# ------------------------------------------------------------------------------- scanning
@dataclass
class DatasetScan:
  """Per-clip QA numbers and the windows of every clip (``windows[start[i]:][:count[i]]``)."""

  names: list[str]
  sources: list[str]
  seconds: np.ndarray
  min_body_z: np.ndarray
  max_joint_speed: np.ndarray
  max_base_jump: np.ndarray
  windows: np.ndarray
  count: np.ndarray

  @property
  def start(self) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(self.count)[:-1]]).astype(np.int64)

  def window_clip(self) -> np.ndarray:
    return np.repeat(np.arange(len(self.names)), self.count)

  def passing(self, floor: float, joint_speed: float, base_jump: float) -> np.ndarray:
    return (
      (self.min_body_z >= floor)
      & (self.max_joint_speed <= joint_speed)
      & (self.max_base_jump <= base_jump)
      & (self.count > 0)
    )


@torch.no_grad()
def scan_dataset(
  source: Path | str,
  *,
  robot_id: str,
  fps: float = 50.0,
  device: str = "cuda:0",
  format_name: str | None = None,
  input_fps: float | None = None,
) -> DatasetScan:
  """QA numbers and windows of every source clip of a dataset (not mirrored, no debias)."""
  from wbc_mjlab.motion.library import Baker

  baker = Baker(
    source,
    robot_id=robot_id,
    fps=fps,
    device=device,
    format_name=format_name,
    input_fps=input_fps,
  )
  lib = baker.library
  names, sources, seconds, low, speed, jump, windows = [], [], [], [], [], [], []
  for segment in baker.segments():
    n = int(baker.lengths[segment])
    seg = torch.full((n,), segment, dtype=torch.long, device=lib.device)
    t = torch.arange(n, device=lib.device, dtype=torch.float64) / fps
    q, qv = lib.state_at(seg, t, 1.0 / fps)
    pos = baker.kin.fk(q, qv)[0]
    names.append(baker.segment_names[segment])
    sources.append(str(baker.source_of(segment)))
    seconds.append(n / fps)
    low.append(float(pos[..., 2].min()))
    speed.append(float(qv[:, 6:].abs().max()) if n else 0.0)
    clip = int(lib.segment_clip[segment])
    start, length = int(lib.packed.start[clip]), int(lib.packed.length[clip])
    raw = lib.packed.frames[start : start + length, :3]
    jump.append(
      float(np.linalg.norm(np.diff(raw, axis=0), axis=1).max()) if length > 1 else 0.0
    )
    windows.append(clip_windows(q.cpu().numpy(), fps))
  return DatasetScan(
    names=names,
    sources=sources,
    seconds=np.array(seconds),
    min_body_z=np.array(low),
    max_joint_speed=np.array(speed),
    max_base_jump=np.array(jump),
    windows=np.concatenate(windows) if windows else np.zeros((0, 0), np.float32),
    count=np.array([len(w) for w in windows], np.int64),
  )


# ------------------------------------------------------------------------------- selection
def _targets(counts: torch.Tensor, tau: float, budget: float) -> torch.Tensor:
  t = counts.clamp_min(1) ** tau * (counts > 0)
  return t / t.sum() * budget


def select_clips(
  windows: np.ndarray,
  count: np.ndarray,
  seconds: np.ndarray,
  *,
  num_clips: int | None = None,
  hours: float | None = None,
  clusters: int = 4096,
  tau: float = 0.5,
  bisect: int = 12,
  seed: int = 0,
  device: str = "cuda:0",
) -> list[int]:
  """Clip indices in selection order (see the module docstring).

  ``num_clips`` alone ranks by gain per clip (``lam`` large); ``hours`` alone takes clips
  by gain per second until the duration is reached; both bisect ``lam`` so that
  ``num_clips`` clips last about ``hours``."""
  if num_clips is None and hours is None:
    raise ValueError("give num_clips, hours or both")
  if device.startswith("cuda") and not torch.cuda.is_available():
    device = "cpu"
  clips = np.flatnonzero(count > 0)
  if num_clips is not None:
    num_clips = min(num_clips, len(clips))
  x = torch.as_tensor(windows, dtype=torch.float32, device=device)
  owner = torch.as_tensor(
    np.searchsorted(clips, np.repeat(np.arange(len(count)), count)), device=device
  )
  _, labels = kmeans(x, clusters, seed=seed)
  k = int(labels.max()) + 1
  counts = torch.bincount(labels, minlength=k).float()
  per_clip = torch.as_tensor(count[clips], dtype=torch.float32, device=device)
  dur = torch.as_tensor(seconds[clips], dtype=torch.float32, device=device)

  def greedy(lam: float, n: int | None, budget_s: float | None) -> list[int]:
    expected = n if n is not None else budget_s / float(dur.mean())
    t = _targets(counts, tau, float(per_clip.mean()) * expected)
    m = torch.zeros(k, device=device)
    chosen = torch.zeros(len(clips), dtype=torch.bool, device=device)
    order, total = [], 0.0
    while len(order) < len(clips):
      if n is not None and len(order) >= n:
        break
      if n is None and total >= budget_s:
        break
      tl = t[labels]
      gain = tl * torch.log1p(1.0 / (tl + m[labels]).clamp_min(1e-9))
      score = torch.zeros(len(clips), device=device).index_add_(0, owner, gain)
      score = score / (dur + lam)
      score[chosen] = -1.0
      i = int(score.argmax())
      chosen[i] = True
      m.index_add_(0, labels[owner == i], torch.ones(int(per_clip[i]), device=device))
      order.append(i)
      total += float(dur[i])
    return order

  if hours is None:
    order = greedy(1e6, num_clips, None)
  elif num_clips is None:
    order = greedy(0.0, None, hours * 3600.0)
  else:
    lo, hi = -0.95 * float(dur.min()), 4.0 * float(dur.mean())
    order, best = [], math.inf
    for _ in range(bisect):
      lam = 0.5 * (lo + hi)
      trial = greedy(lam, num_clips, None)
      h = float(dur[trial].sum()) / 3600.0
      if abs(h - hours) < best:
        order, best = trial, abs(h - hours)
      if h > hours:
        hi = lam  # smaller lam favours short clips
      else:
        lo = lam
  return [int(clips[i]) for i in order]


# ------------------------------------------------------------------------------- analysis
def analyze_windows(
  windows: np.ndarray,
  count: np.ndarray,
  *,
  clusters: int = 1024,
  heldout: np.ndarray | None = None,
  quantile: float = 0.9,
  seed: int = 0,
  device: str = "cuda:0",
) -> dict:
  """Cluster occupancy of a pool, per-clip rarity and the coverage of a held-out set.

  ``rarity`` of a clip is the mean over its windows of ``mean cluster size / its cluster's
  size`` (1 is typical, above is rare, below redundant). A held-out window is covered when
  its distance to the nearest pool cluster center is within the ``quantile`` of that
  cluster's members' distances (:func:`cluster_radii`)."""
  if device.startswith("cuda") and not torch.cuda.is_available():
    device = "cpu"
  x = torch.as_tensor(windows, dtype=torch.float32, device=device)
  centers, labels = kmeans(x, clusters, seed=seed)
  k = len(centers)
  sizes = torch.bincount(labels, minlength=k).float()
  p = sizes[sizes > 0] / sizes.sum()
  entropy = float(-(p * p.log()).sum())
  top = torch.sort(sizes, descending=True).values
  owner = torch.as_tensor(np.repeat(np.arange(len(count)), count), device=device)
  inv = sizes.mean() / sizes[labels]
  rarity = torch.zeros(len(count), device=device).index_add_(0, owner, inv)
  rarity = rarity / torch.as_tensor(count, device=device).clamp_min(1)
  out = {
    "windows": int(len(x)),
    "clusters": int(k),
    "occupied_clusters": int((sizes > 0).sum()),
    "effective_clusters": math.exp(entropy),
    "evenness": entropy / math.log(max(int((sizes > 0).sum()), 2)),
    "top10pct_cluster_share": float(top[: max(1, k // 10)].sum() / sizes.sum()),
    "rarity": rarity.cpu().numpy(),
  }
  if heldout is not None and len(heldout):
    radius = cluster_radii(x, centers, labels, quantile)
    h = torch.as_tensor(heldout, dtype=torch.float32, device=device)
    near, dist = nearest(h, centers)
    out["heldout_coverage"] = float((dist <= radius[near]).float().mean())
    out["heldout_median_distance_ratio"] = float(
      (dist.sqrt() / radius[near].sqrt()).median()
    )
  return out


def cluster_radii(x, centers, labels, quantile: float) -> torch.Tensor:
  """Squared radius of each cluster: the ``quantile`` of its members' distances to the
  center (the median over clusters for those with fewer than two members)."""
  own = ((x - centers[labels]) ** 2).sum(1)
  by_distance = torch.argsort(own)
  order = by_distance[torch.argsort(labels[by_distance], stable=True)]
  sizes = torch.bincount(labels, minlength=len(centers))
  start = torch.cumsum(sizes, 0) - sizes
  pick = start + ((sizes - 1).clamp_min(0).float() * quantile).round().long()
  radius = own[order][pick.clamp_max(len(own) - 1)]
  small = sizes < 2
  if bool(small.any()) and bool((~small).any()):
    radius = torch.where(small, radius[~small].median(), radius)
  return radius.clamp_min(1e-12)


# ------------------------------------------------------------------------------- output
def link_dataset(
  sources: Sequence[str], out_dir: Path | str, *, copy: bool = False
) -> Path:
  """A dataset folder ``out_dir/raw`` of links (or copies) to ``sources``."""
  import shutil

  raw = Path(out_dir) / "raw"
  raw.mkdir(parents=True, exist_ok=True)
  for src in sources:
    src = Path(src).resolve()
    dst = raw / src.name
    if dst.exists() or dst.is_symlink():
      dst.unlink()
    if copy:
      shutil.copy2(src, dst)
    else:
      os.symlink(src, dst)
  return raw.parent
