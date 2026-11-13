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
import shutil
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
    """The last ``n`` non-empty lines of the output (within its last ``nbytes``), as a
    terminal shows them: progress bars redrawn in place, colour codes dropped."""
    p = Path(self.log)
    if not p.is_file():
      return ""
    with p.open("rb") as f:
      f.seek(0, os.SEEK_END)
      size = f.tell()
      f.seek(max(0, size - nbytes))
      text = ANSI_RE.sub("", f.read().decode(errors="replace"))
    # tqdm redraws a line with \r: keep what is visible last on each line.
    lines = [ln.rstrip("\r").rsplit("\r", 1)[-1] for ln in text.split("\n")]
    return "\n".join([ln for ln in lines if ln.strip()][-n:])

  def run_dir(self) -> str:
    """The run folder a training job writes, as its log announces it near the start; ``""``
    before that (or for other jobs)."""
    p = Path(self.log)
    if self.kind != "train" or not p.is_file():
      return ""
    with p.open("rb") as f:
      head = ANSI_RE.sub("", f.read(1 << 16).decode(errors="replace"))
    m = re.search(r"Logging experiment in directory: (\S+)", head)
    if not m:
      return ""
    d = Path(m.group(1))
    return str(d if d.is_absolute() else project_root() / d)

  def log_size(self) -> int:
    p = Path(self.log)
    return p.stat().st_size if p.is_file() else 0

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
      # the world model's first update compiles for minutes without other output
      compiling = text.rfind("[INFO] wm: first update: compiling")
      if compiling >= 0 and "[INFO] wm: first update: compiled" not in text[compiling:]:
        out["phase"] = "compiling the world model's update (first time: minutes)"
      if "iteration" in out:
        it, total = out["iteration"]
        out["iteration"], out["total"] = int(it), int(total)
        out["fraction"] = (int(it) + 1) / max(1, int(total))
    elif self.kind == "evaluate":
      found = re.findall(r"\[EVAL\] clips (\d+)/(\d+)", text)
      if found:
        done, total = map(int, found[-1])
        out.update(done=done, total=total, fraction=done / max(total, 1))
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
    code = re.findall(EXIT_RE, text)
    if code:  # the exit status the job's shell wrapper wrote (launch)
      return "done" if code[-1] == "0" else "failed"
    # jobs started before the wrapper: read the output
    if self.kind == "convert":
      return "done" if "[INFO] Finished" in text else "failed"
    if re.search(r"Traceback|Error|CUDA out of memory|Unrecognized options", text):
      return "failed"
    return "done"

  def save(self) -> None:
    (_jobs_dir() / f"{self.name}.json").write_text(json.dumps(asdict(self), indent=2))


def _jobs_dir() -> Path:
  d = logs_dir() / "studio" / "jobs"
  d.mkdir(parents=True, exist_ok=True)
  return d


EXIT_RE = r"\[job exit (\d+)\]"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
"""Terminal colour / cursor codes (rich and rsl_rl print bold headers)."""


def launch(kind: str, args: list[str], *, label: str, meta: dict | None = None) -> Job:
  """Start ``python -m <args…>`` detached from the Studio (own session, own log); a shell
  around it writes its exit status to the log (``[job exit N]``), which :meth:`Job.status`
  reads."""
  safe = re.sub(r"[^\w.-]+", "-", label)
  name = f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{safe}"
  log = _jobs_dir() / f"{name}.log"
  cmd = [sys.executable, "-m", *args]
  with log.open("w") as f:
    f.write(f"$ {' '.join(cmd)}\n")
    f.flush()
    proc = subprocess.Popen(
      ["/bin/sh", "-c", '"$@"; echo "[job exit $?]"', "sh", *cmd],
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


def clear_jobs(kind: str | None = None, robot: str | None = None) -> list[str]:
  """Forget every finished job (optionally of one kind / robot): its record and log. What it
  produced (run folders, exported clips) stays. Returns the names removed."""
  names = []
  for job in list_jobs(kind):
    if job.alive or (robot and job.meta.get("robot", robot) != robot):
      continue
    delete_job(job.name)
    names.append(job.name)
  return names


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

  def size(self) -> int:
    return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())

  def preset_snapshot(self) -> dict | None:
    """The Studio preset the run trained with (``params/studio_preset.yaml``), if any."""
    from wbc_mjlab.studio.presets import snapshot_of

    return snapshot_of(self.path)

  def task_id(self) -> str:
    """Task of the run: the bundled demo is ``Wbc-G1``; a preset run's own snapshot; else
    the task logging into ``<experiment>`` (task ids map 1:1 to experiment names)."""
    from wbc_mjlab.run_params import run_task_id

    return "Wbc-G1" if self.is_demo else run_task_id(self.path)

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


def list_runs(*, empty: bool = False) -> list[Run]:
  """Run folders with checkpoints (``empty``: also those a job left without one)."""
  runs = []
  root = logs_dir() / "rsl_rl"
  if root.is_dir():
    for exp in sorted(p for p in root.iterdir() if p.is_dir()):
      for d in sorted(exp.iterdir(), reverse=True):
        if d.is_dir() and (empty or any(d.glob("model_*.pt"))):
          runs.append(Run(exp.name, d.name, d))
  demo = project_root() / "demos" / "wbc_g1"
  if (demo / "model.pt").is_file():
    runs.append(Run("demo", "wbc_g1", demo))
  return runs


def find_run(label: str, *, empty: bool = False) -> Run | None:
  return next((r for r in list_runs(empty=empty) if r.label == label), None)


def delete_run(run: Run) -> int:
  """Delete a run folder (checkpoints, curves, params); returns the bytes freed."""
  freed = run.size()
  shutil.rmtree(run.path)
  if not any(run.path.parent.iterdir()):  # the experiment's last run
    run.path.parent.rmdir()
  return freed


def prune_checkpoints(run: Run, every: int = 0) -> int:
  """Delete every ``model_<it>.pt`` but the last (and, with ``every``, those at multiples of
  ``every`` iterations); returns the bytes freed."""
  freed = 0
  for p in run.checkpoints()[:-1]:
    it = checkpoint_iteration(p)
    if it < 0 or (every and it % every == 0):
      continue
    freed += p.stat().st_size
    p.unlink()
  return freed


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
  "clear_jobs",
  "delete_job",
  "delete_run",
  "find_job",
  "find_run",
  "launch",
  "list_jobs",
  "list_runs",
  "logs_dir",
  "prune_checkpoints",
  "scalars",
  "stop",
]
