"""Detached jobs (conversion, training) and run / checkpoint discovery.

Jobs survive a Studio restart: each launch writes ``logs/studio/jobs/<job>.json`` (pid, kind,
command, stdout log); the Studio re-attaches to every job whose pid is still alive. Runs are
the ``logs/rsl_rl/<experiment>/<run>`` folders holding ``model_<it>.pt`` checkpoints, plus
the bundled demo checkpoint (``demos/wbc_g1``).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from wbc_mjlab.data_paths import project_root


def logs_dir() -> Path:
  return project_root() / "logs"


def checkpoint_iteration(path: str | Path) -> int:
  m = re.search(r"model_(\d+)\.pt$", str(path))
  return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------- jobs
@dataclass
class Job:
  name: str
  kind: str  # "train" | "convert"
  pid: int
  log: str
  started: float
  command: list[str]
  meta: dict = field(default_factory=dict)  # task / dataset / robot …
  finished: float = 0.0
  stopped: bool = False

  @property
  def alive(self) -> bool:
    try:
      os.kill(self.pid, 0)
    except OSError:
      return False
    try:  # a zombie of ours is not alive
      done, _ = os.waitpid(self.pid, os.WNOHANG)
      return done == 0
    except ChildProcessError:
      return True

  def tail(self, n: int = 30, nbytes: int = 40000) -> str:
    p = Path(self.log)
    if not p.is_file():
      return ""
    with p.open("rb") as f:
      f.seek(0, os.SEEK_END)
      size = f.tell()
      f.seek(max(0, size - nbytes))
      text = f.read().decode(errors="replace")
    # tqdm redraws a line with \r: keep what is visible last on each line.
    lines = [ln.rstrip("\r").rsplit("\r", 1)[-1] for ln in text.split("\n")]
    return "\n".join([ln for ln in lines if ln.strip()][-n:])

  def progress(self) -> dict:
    """Latest progress parsed from the job's console output."""
    text = self.tail(600, nbytes=200000)
    out: dict = {}
    if self.kind == "train":
      for pattern, key in (
        (r"Learning iteration (\d+)/(\d+)", "iteration"),
        (r"Mean reward:\s*([-\d.eE+]+)", "mean_reward"),
        (r"Mean episode length:\s*([-\d.eE+]+)", "episode_length"),
        (r"Logging experiment in directory: (\S+)", "log_dir"),
        (r"Total timesteps:\s*(\d+)", "timesteps"),
        (r"ETA:\s*([\d:.]+)", "eta"),
      ):
        found = re.findall(pattern, text)
        if found:
          out[key] = found[-1]
      if "iteration" in out:
        it, total = out["iteration"]
        out["iteration"], out["total"] = int(it), int(total)
        out["fraction"] = (int(it) + 1) / max(1, int(total))
    else:
      p = Path(self.log)
      full = p.read_text(errors="replace") if p.is_file() else ""
      m = re.search(r"Starting FK pool: \d+ env\(s\), (\d+) clip\(s\)", full)
      done = full.count("[INFO] Exported clip:")
      out["done"] = done
      if m:
        out["total"] = int(m.group(1)) * (2 if self.meta.get("mirror") else 1)
        out["fraction"] = min(1.0, done / max(1, out["total"]))
      fin = re.search(r"Finished (\d+) clip", full)
      if fin:
        out["finished_clips"] = int(fin.group(1))
    return out

  def status(self) -> str:
    if self.alive:
      return "running"
    if self.stopped:
      return "stopped"
    text = self.tail(40)
    if self.kind == "convert":
      return "done" if "[INFO] Finished" in text else "failed"
    if re.search(r"Traceback|Error|CUDA out of memory", text):
      return "failed"
    return "done"

  def save(self) -> None:
    (_jobs_dir() / f"{self.name}.json").write_text(json.dumps(asdict(self), indent=2))


def _jobs_dir() -> Path:
  d = logs_dir() / "studio" / "jobs"
  d.mkdir(parents=True, exist_ok=True)
  return d


def launch(kind: str, args: list[str], *, label: str, meta: dict | None = None) -> Job:
  """Start ``python -m <args…>`` detached from the Studio (own session, own log)."""
  safe = re.sub(r"[^\w.-]+", "-", label)
  name = f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{safe}"
  log = _jobs_dir() / f"{name}.log"
  cmd = [sys.executable, "-m", *args]
  with log.open("w") as f:
    f.write(f"$ {' '.join(cmd)}\n")
    f.flush()
    proc = subprocess.Popen(
      cmd,
      cwd=str(project_root()),
      stdout=f,
      stderr=subprocess.STDOUT,
      start_new_session=True,  # survives the Studio
      env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
  job = Job(
    name=name,
    kind=kind,
    pid=proc.pid,
    log=str(log),
    started=time.time(),
    command=cmd,
    meta=dict(meta or {}),
  )
  job.save()
  return job


def stop(job: Job) -> None:
  """SIGINT first (the latest checkpoint stays), SIGKILL after 5 s."""
  job.stopped = True
  job.save()
  try:
    os.killpg(job.pid, signal.SIGINT)
  except OSError:
    return
  for _ in range(50):
    if not job.alive:
      return
    time.sleep(0.1)
  try:
    os.killpg(job.pid, signal.SIGKILL)
  except OSError:
    pass


def list_jobs(kind: str | None = None) -> list[Job]:
  jobs = []
  for p in sorted(_jobs_dir().glob("*.json"), reverse=True):
    try:
      job = Job(**json.loads(p.read_text()))
    except Exception:  # noqa: BLE001
      continue
    if kind is None or job.kind == kind:
      jobs.append(job)
  return jobs


def find_job(name: str) -> Job | None:
  p = _jobs_dir() / f"{name}.json"
  if not p.is_file():
    return None
  return Job(**json.loads(p.read_text()))


def delete_job(name: str) -> None:
  for suffix in (".json", ".log"):
    (_jobs_dir() / f"{name}{suffix}").unlink(missing_ok=True)


# ---------------------------------------------------------------------- runs / checkpoints
DEMO_LABEL = "demo/wbc_g1"


@dataclass
class Run:
  experiment: str
  name: str
  path: Path

  @property
  def label(self) -> str:
    return f"{self.experiment}/{self.name}"

  @property
  def is_demo(self) -> bool:
    return self.label == DEMO_LABEL

  def checkpoints(self) -> list[Path]:
    return sorted(self.path.glob("*.pt"), key=checkpoint_iteration)

  def preset_snapshot(self) -> dict | None:
    """The Studio preset the run trained with (``params/studio_preset.yaml``), if any."""
    from wbc_mjlab.studio.presets import snapshot_of

    return snapshot_of(self.path)

  def task_id(self) -> str:
    """Task of the run: the bundled demo is ``Wbc-G1``; a preset run's own snapshot; else
    the task logging into ``<experiment>`` (task ids map 1:1 to experiment names)."""
    if self.is_demo:
      return "Wbc-G1"
    snap = self.preset_snapshot()
    if snap:
      from wbc_mjlab.studio.presets import task_id

      return task_id(snap)
    from wbc_mjlab.tasks import all_wbc_tasks

    for t in all_wbc_tasks():
      if t.experiment_name == self.experiment:
        return t.task_id
    from wbc_mjlab.studio.presets import task_for_experiment

    return task_for_experiment(self.experiment)

  def robot_id(self) -> str:
    from wbc_mjlab.studio.presets import split_task_id
    from wbc_mjlab.tasks import get_task_config

    tid = self.task_id()
    if not tid:
      return ""
    if split_task_id(tid):
      tid = split_task_id(tid)[0]
    try:
      return get_task_config(tid).robot_id
    except KeyError:
      return ""

  def motion_source(self) -> str:
    """Motion library the run trained on (``params/env.yaml``), if it still exists."""
    env_yaml = self.path / "params" / "env.yaml"
    if not env_yaml.is_file():
      return ""
    text = env_yaml.read_text(errors="replace")
    m = re.search(r"motion_file:\s*(\S+)", text)
    if m and Path(m.group(1)).exists():
      return m.group(1)
    return ""


def list_runs() -> list[Run]:
  runs = []
  root = logs_dir() / "rsl_rl"
  if root.is_dir():
    for exp in sorted(p for p in root.iterdir() if p.is_dir()):
      for d in sorted(exp.iterdir(), reverse=True):
        if d.is_dir() and any(d.glob("model_*.pt")):
          runs.append(Run(exp.name, d.name, d))
  demo = project_root() / "demos" / "wbc_g1"
  if (demo / "model.pt").is_file():
    runs.append(Run("demo", "wbc_g1", demo))
  return runs


def find_run(label: str) -> Run | None:
  return next((r for r in list_runs() if r.label == label), None)


def scalars(
  run_dir: Path, tags: tuple[str, ...] | None = None
) -> dict[str, tuple[list, list]]:
  """Scalar series from the TensorBoard event files of a run (``tags=None``: all)."""
  try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  except ImportError:
    return {}
  acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 2000})
  try:
    acc.Reload()
  except Exception:  # noqa: BLE001
    return {}
  have = acc.Tags().get("scalars", [])
  out = {}
  for tag in have if tags is None else [t for t in tags if t in have]:
    ev = acc.Scalars(tag)
    out[tag] = ([e.step for e in ev], [float(e.value) for e in ev])
  return out


__all__ = [
  "DEMO_LABEL",
  "Job",
  "Run",
  "checkpoint_iteration",
  "delete_job",
  "find_job",
  "find_run",
  "launch",
  "list_jobs",
  "list_runs",
  "logs_dir",
  "scalars",
  "stop",
]
