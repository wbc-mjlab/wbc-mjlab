"""Training service: WBC tasks of the robot, detached ``wbc-mjlab-train`` jobs, runs, curves.

A job trains a preset with a recipe (:mod:`wbc_mjlab.studio.recipes`):
``wbc-mjlab-train --task <task> --dataset <dataset> --algo <method> --agent.<key> <value>…``
plus the run options picked in the Studio (env count, iterations, run name, resume from a
checkpoint of the same task and method), started detached so it outlives the Studio. Curves
come from the run's TensorBoard event files; one TensorBoard over ``logs/rsl_rl`` starts on
demand.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from wbc_mjlab.studio import jobs as jobmod

CURVE_GROUPS = {
  "train": ("Train/mean_reward", "Train/mean_episode_length"),
  "errors": (
    "Metrics/motion/error_anchor_pos",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_joint_pos",
    "Metrics/motion/error_body_rot",
  ),
  "sampling": ("Metrics/motion/sampling_entropy", "Metrics/motion/assist_gain_mean"),
  # "loss": every Loss/* of the run's algorithm (PPO's and the world model's differ), then
  "loss": ("Policy/mean_std", "Perf/total_fps"),
}
LOSS_SKIP = ("Loss/learning_rate",)


def _size(n: int) -> str:
  return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


class TrainingService:
  def __init__(self, core):
    self.c = core
    self._status: dict[str, str] = {}
    self._checks: dict[tuple, dict] = {}
    self._check_lock = threading.Lock()
    self._timing: dict[tuple, dict] = {}

  # ------------------------------------------------------------------ tasks
  def tasks(self) -> list[dict]:
    """Built-in tasks (paper presets) of the robot, then the Studio's custom presets."""
    from wbc_mjlab.studio import presets
    from wbc_mjlab.studio.robots import robot_tasks

    out = [t | {"custom": False} for t in robot_tasks(self.c.robot_id)]
    for p in presets.list_presets(self.c.robot_id):
      out.append(
        {
          "id": p["task"],
          "description": p.get("description")
          or " + ".join([p["base"], *p["modules"], f"{p['changes']} change(s)"]),
          "experiment": "",
          "custom": True,
          "preset": p["name"],
          "base": p["base"],
          "modules": p["modules"],
          "changes": p["changes"],
        }
      )
    return out

  # ------------------------------------------------------------------ methods
  def methods(self, task: str) -> dict:
    """The training methods with every setting of each for ``task`` (defaults: the task's
    runner config), and the recipes (``default``: each method's starting recipe here)."""
    from wbc_mjlab.rl.algos import ALGORITHMS
    from wbc_mjlab.studio import recipes

    self._register(task)
    return {
      "methods": [
        {
          "id": info.id,
          "label": info.label,
          "summary": info.summary,
          "notes": info.notes,
          "iterations": list(info.iterations),
          "settings": recipes.fields(task, info.id) if task else [],
          "default_recipe": recipes.default(info.id),
        }
        for info in ALGORITHMS.values()
      ],
      "recipes": recipes.list_recipes(),
      "architecture": list(recipes.ARCHITECTURE),
    }

  def iteration_time(self, algo: str) -> list[dict]:
    """Measured seconds per iteration of ``algo``'s runs here (newest first):
    ``{run, task, num_envs, seconds}``, from their TensorBoard ``Perf/*_time`` scalars."""
    from wbc_mjlab.rl.algos import check_algo, run_algo
    from wbc_mjlab.run_params import read_run_params

    check_algo(algo)
    out = []
    for r in sorted(jobmod.list_runs(), key=lambda r: r.name, reverse=True):
      if r.is_demo or run_algo(r.path) != algo:
        continue
      events = list(r.path.glob("events.out.tfevents.*"))
      if not events:
        continue
      key = (str(r.path), max(p.stat().st_mtime for p in events))
      if key not in self._timing:
        tags = ("Perf/collection_time", "Perf/learning_time")
        s = jobmod.scalars(r.path, tags)
        if not all(s.get(t, ([], []))[1] for t in tags):
          continue
        tail = [s[t][1][-50:] for t in tags]
        seconds = sum(sum(v) / len(v) for v in tail)
        env = read_run_params(r.path, "env")
        num_envs = (env.get("scene") or {}).get("num_envs")
        self._timing[key] = {
          "run": r.label,
          "task": r.task_id(),
          "num_envs": num_envs,
          "seconds": seconds,
        }
      out.append(self._timing[key])
      if len(out) == 5:
        break
    return out

  def check(self, task: str, algo: str, dataset: str) -> dict:
    """Can ``algo`` train ``task``? A 2-env copy of its env is built in a subprocess
    (:mod:`wbc_mjlab.studio.algo_check`); results are kept per task, preset content and
    dataset. PPO trains every task: no check."""
    from wbc_mjlab.rl.algos import check_algo
    from wbc_mjlab.studio import presets
    from wbc_mjlab.studio.algo_check import MARK

    check_algo(algo)
    if algo == "ppo":
      return {"ok": True, "error": "", "details": {}}
    preset = presets.split_task_id(task)
    stamp = json.dumps(presets.resolve(preset[1]), sort_keys=True) if preset else ""
    key = (task, algo, dataset, stamp)
    with self._check_lock:
      if key in self._checks:
        return self._checks[key]
      cmd = [sys.executable, "-m", "wbc_mjlab.studio.algo_check"]
      cmd += ["--task", task, "--algo", algo, "--dataset", dataset]
      try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
      except subprocess.TimeoutExpired:
        return {"ok": False, "error": "check timed out (10 min)", "details": {}}
      line = next((x for x in proc.stdout.splitlines() if x.startswith(MARK)), None)
      if line is None:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        return {"ok": False, "error": f"check failed: {tail[0]}", "details": {}}
      result = json.loads(line[len(MARK) :])
      self._checks[key] = result
      return result

  def _register(self, task: str) -> None:
    """Make a Studio preset's task known in this process (built-in ones always are)."""
    from wbc_mjlab.studio import presets

    preset = presets.split_task_id(task) if task else None
    if preset:
      presets.register(preset[1])

  # ------------------------------------------------------------------ runs
  def runs(self, robot: str | None = None) -> list[dict]:
    from wbc_mjlab.rl.algos import run_algo
    from wbc_mjlab.studio import recipes

    out = []
    for r in jobmod.list_runs():
      rid = r.robot_id()
      if robot and rid != robot:
        continue
      ckpts = r.checkpoints()
      try:
        mtime = (
          max(p.stat().st_mtime for p in ckpts) if ckpts else r.path.stat().st_mtime
        )
      except OSError:
        mtime = 0.0
      out.append(
        {
          "label": r.label,
          "experiment": r.experiment,
          "name": r.name,
          "task": r.task_id(),
          "robot": rid,
          "demo": r.is_demo,
          "checkpoints": [p.name for p in reversed(ckpts)],
          "updated": mtime,
          "updated_text": time.strftime("%d %b %H:%M", time.localtime(mtime))
          if mtime
          else "",
          "motion": r.motion_source(),
          "onnx": (r.path / "params" / "policy.onnx").is_file(),
          "algo": run_algo(r.path),
          "network": recipes.architecture_of(r.path),
          # TensorBoard curves to open (a run under logs/rsl_rl that logged to TensorBoard)
          "events": not r.is_demo and any(r.path.glob("events.out.tfevents.*")),
          "path": str(r.path),
        }
      )
    out.sort(key=lambda r: (r["demo"], -r["updated"]))
    return out

  # ------------------------------------------------------------------ jobs
  def launch(
    self,
    task: str,
    dataset: str,
    *,
    num_envs: int = 4096,
    max_iterations: int = 30000,
    run_name: str = "",
    resume_run: str = "",
    resume_checkpoint: str = "",
    logger: str = "tensorboard",
    mirror: bool = False,
    debias_z: bool = False,
    algo: str = "ppo",
    algo_settings: dict | None = None,
    recipe: str = "",
    extra: str = "",
  ) -> jobmod.Job:
    """``algo_settings``: the recipe's settings as edited in the form (``recipe``: its name,
    kept in the job's meta)."""
    from wbc_mjlab.rl.algos import ALGORITHMS, check_algo, run_algo
    from wbc_mjlab.studio import presets, recipes
    from wbc_mjlab.tasks import get_task_config

    algo = check_algo(algo)

    preset = presets.split_task_id(task)
    if preset:
      presets.register(preset[1])  # validates the preset and learns its experiment name
    cfg = get_task_config(task)
    settings = recipes.check_settings(cfg.task_id, algo, algo_settings)
    if not dataset:
      raise ValueError("pick a dataset")
    from wbc_mjlab.data_paths import resolve_dataset_root
    from wbc_mjlab.motion.library import has_sources
    from wbc_mjlab.motion.stack_bundle import list_clip_npz_files

    root = resolve_dataset_root(cfg.robot_id, dataset)
    if not has_sources(root) and not list_clip_npz_files(root):
      raise ValueError(f"{dataset} has no clips (source CSV / PKL or converted NPZ)")
    if mirror:
      from wbc_mjlab.robots.symmetry import get_robot_symmetry_config

      if get_robot_symmetry_config(cfg.robot_id) is None:
        raise ValueError(
          f"mirroring needs a left/right symmetry config for {cfg.robot_id}"
        )
    entry = (
      ["wbc_mjlab.studio.train_preset", "--preset", preset[1]]
      if preset
      else ["wbc_mjlab.scripts.train", "--task", cfg.task_id]
    )
    args = [
      *entry,
      "--dataset",
      dataset,
      "--env.scene.num-envs",
      str(int(num_envs)),
      "--agent.max-iterations",
      str(int(max_iterations)),
      "--algo",
      algo,
    ]
    run = None
    if resume_run:
      run = jobmod.find_run(resume_run)
      if run is None:
        raise ValueError(f"no run {resume_run!r}")
      if run.task_id() != cfg.task_id:
        raise ValueError(
          f"{resume_run} was trained with {run.task_id()}, not {cfg.task_id}"
        )
      if run_algo(run.path) != algo:
        label = ALGORITHMS[run_algo(run.path)].label
        raise ValueError(
          f"{resume_run} was trained with {label}: continue it with that"
        )
      # its checkpoint fixes the networks: build them as the run did
      settings |= recipes.architecture_of(run.path)
    for key, value in settings.items():
      args += [f"--agent.{key.replace('_', '-')}", *_cli_values(value)]
    if mirror:
      args += ["--env.commands.motion.mirror", "True"]
    if debias_z:
      args += ["--env.commands.motion.debias-z", "True"]
    if run_name:
      args += ["--agent.run-name", re.sub(r"[^\w.-]+", "-", run_name)]
    if logger == "wandb":
      args += ["--use_wandb"]
    if run is not None:
      args += ["--agent.resume", "True", "--agent.load-run", run.name]
      if resume_checkpoint:
        args += ["--agent.load-checkpoint", resume_checkpoint]
    if extra.strip():
      args += extra.split()
    job = jobmod.launch(
      "train",
      args,
      label=f"{cfg.task_id}-{dataset}",
      meta={
        "task": cfg.task_id,
        "robot": cfg.robot_id,
        "dataset": dataset,
        "num_envs": int(num_envs),
        "max_iterations": int(max_iterations),
        "mirror": bool(mirror),
        "debias_z": bool(debias_z),
        "algo": algo,
        "recipe": recipe,
        "algo_settings": settings,
        "resume": f"{resume_run}/{resume_checkpoint}" if resume_run else "",
        "experiment": cfg.experiment_name,
      },
    )
    self.c.log(
      f"training {cfg.task_id} on {dataset} with {recipe or ALGORITHMS[algo].label}: "
      f"job {job.name}",
      "ok",
    )
    self.c.emit("jobs")
    return job

  def evaluate(
    self, run: str, checkpoint: str = "", num_envs: int = 128, motion: str = ""
  ):
    """Launch the common deterministic benchmark for a saved checkpoint."""
    saved = jobmod.find_run(run)
    if saved is None:
      raise ValueError(f"no run {run!r}")
    checkpoints = saved.checkpoints()
    selected = (
      next((p for p in checkpoints if p.name == checkpoint), None)
      if checkpoint
      else next(iter(reversed(checkpoints)), None)
    )
    if selected is None:
      raise ValueError("select a saved checkpoint to evaluate")
    if not 1 <= num_envs <= 4096:
      raise ValueError("evaluation needs 1–4096 environments")
    args = [
      "wbc_mjlab.scripts.evaluate",
      "--checkpoint-file",
      str(selected),
      "--num-envs",
      str(num_envs),
      "--device",
      self.c.device,
    ]
    if motion:
      args += ["--dataset-path", motion]
    job = jobmod.launch(
      "evaluate",
      args,
      label=f"evaluate-{saved.name}",
      meta={
        "robot": saved.robot_id(),
        "task": saved.task_id(),
        "checkpoint": str(selected),
        "dataset": motion or saved.motion_source(),
        "num_envs": num_envs,
      },
    )
    self.c.emit("jobs")
    return job

  def stop(self, name: str) -> None:
    job = jobmod.find_job(name)
    if job is None:
      raise ValueError(f"no job {name!r}")
    jobmod.stop(job)
    self.c.log(f"stopped {name}", "warn")
    self.c.emit("jobs")

  def delete(self, name: str, run: bool = False) -> int:
    """Remove a finished job (record and log); ``run``: its run folder too. Returns the
    bytes freed."""
    job = jobmod.find_job(name)
    if job is not None and job.alive:
      raise ValueError("stop the job first")
    freed = 0
    if run and job is not None and job.run_dir():
      freed = self._delete_runs([job.run_dir()])
    jobmod.delete_job(name)
    self.c.emit("jobs")
    return freed

  def clear(
    self, kind: str | None = None, robot: str | None = None, runs: bool = False
  ) -> dict:
    """Remove every finished job (of a kind / robot); ``runs``: their run folders too."""
    dirs = {j.name: j.run_dir() for j in jobmod.list_jobs(kind)} if runs else {}
    names = jobmod.clear_jobs(kind, robot)
    paths = [dirs[n] for n in names if dirs.get(n)]
    freed = self._delete_runs(paths, strict=False) if runs else 0
    if names:
      self.c.log(f"removed {len(names)} finished job(s)", "ok")
      self.c.emit("jobs")
    return {"deleted": names, "freed": freed}

  # ------------------------------------------------------------------ disk
  def storage(self) -> dict:
    """Every run folder (the bundled demo aside) with its size, checkpoints and the job that
    wrote it; ``busy``: why it can't be changed now (a running job, or loaded in Play)."""
    from wbc_mjlab.rl.algos import run_algo

    jobs = {j.run_dir(): j for j in jobmod.list_jobs("train")}
    rows = []
    for r in jobmod.list_runs(empty=True):
      if r.is_demo:
        continue
      job = jobs.get(str(r.path))
      status = job.status() if job else ""
      ckpts = r.checkpoints()
      mtime = r.path.stat().st_mtime
      rows.append(
        {
          "label": r.label,
          "task": r.task_id(),
          "algo": run_algo(r.path),
          "checkpoints": len(ckpts),
          "last": jobmod.checkpoint_iteration(ckpts[-1]) if ckpts else -1,
          "size": r.size(),
          "prunable": sum(p.stat().st_size for p in ckpts[:-1]),
          "updated": mtime,
          "updated_text": time.strftime("%d %b %H:%M", time.localtime(mtime)),
          "job": job.name if job else "",
          "status": status,
          "busy": self._busy(str(r.path), status),
        }
      )
    rows.sort(key=lambda r: -r["updated"])
    return {"runs": rows, "total": sum(r["size"] for r in rows)}

  def _busy(self, path: str, status: str) -> str:
    if status == "running":
      return "a job is training it"
    play = getattr(self.c, "play", None)
    if play is not None and play.session is not None:
      r = jobmod.find_run(play.run_label)
      if r is not None and str(r.path) == path:
        return "loaded in Play"
    return ""

  def _runs_at(self, paths: list[str], strict: bool = True) -> list[jobmod.Run]:
    """The runs at ``paths``; ValueError naming the ones that can't change now (``strict``)
    or without them."""
    by_path = {str(r.path): r for r in jobmod.list_runs(empty=True) if not r.is_demo}
    status = {j.run_dir(): j.status() for j in jobmod.list_jobs("train")}
    runs, refused = [], []
    for p in paths:
      r = by_path.get(p)
      if r is None:
        continue
      why = self._busy(p, status.get(p, ""))
      if why:
        refused.append(f"{r.label} ({why})")
      else:
        runs.append(r)
    if refused and strict:
      raise ValueError(f"can't change {', '.join(refused)}")
    return runs

  def _paths(self, labels: list[str]) -> list[str]:
    runs = {r.label: r for r in jobmod.list_runs(empty=True)}
    missing = [x for x in labels if x not in runs]
    if missing:
      raise ValueError(f"no run {', '.join(missing)}")
    return [str(runs[x].path) for x in labels]

  def _delete_runs(self, paths: list[str], strict: bool = True) -> int:
    runs = self._runs_at(paths, strict)
    freed = 0
    for r in runs:
      freed += jobmod.delete_run(r)
      for j in jobmod.list_jobs("train"):
        if j.run_dir() == str(r.path) and not j.alive:
          jobmod.delete_job(j.name)
    if runs:
      self.c.log(f"deleted {len(runs)} run(s), {_size(freed)} freed", "warn")
      self.c.emit("jobs")
    return freed

  def delete_runs(self, labels: list[str]) -> int:
    """Delete run folders and the jobs that wrote them; returns the bytes freed."""
    return self._delete_runs(self._paths(labels))

  def prune_runs(self, labels: list[str], every: int = 0) -> int:
    """Keep each run's last checkpoint (and, with ``every``, those at multiples of ``every``
    iterations); returns the bytes freed."""
    runs = self._runs_at(self._paths(labels))
    freed = sum(jobmod.prune_checkpoints(r, every) for r in runs)
    if runs:
      self.c.log(f"pruned {len(runs)} run(s), {_size(freed)} freed", "ok")
      self.c.emit("jobs")
    return freed

  def jobs(self, kind: str | None = None) -> list[dict]:
    out = []
    for j in jobmod.list_jobs(kind)[:60]:
      status = j.status()
      out.append(
        {
          "name": j.name,
          "kind": j.kind,
          "status": status,
          "started": j.started,
          "started_text": time.strftime("%d %b %H:%M", time.localtime(j.started)),
          "elapsed": (time.time() if status == "running" else (j.finished or j.started))
          - j.started,
          "meta": j.meta,
          "progress": j.progress(),
        }
      )
    return out

  def job_detail(self, name: str, tail: int = 60) -> dict:
    job = jobmod.find_job(name)
    if job is None:
      raise ValueError(f"no job {name!r}")
    return {
      "name": job.name,
      "kind": job.kind,
      "status": job.status(),
      "command": " ".join(job.command),
      "log": job.tail(tail),
      "meta": job.meta,
      "progress": job.progress(),
      "run_dir": self._job_run_dir(job),
    }

  def evaluation_result(self, name: str) -> Path:
    job = jobmod.find_job(name)
    if job is None or job.kind != "evaluate" or job.status() != "done":
      raise ValueError("evaluation has not finished successfully")
    checkpoint = Path(job.meta["checkpoint"])
    result = checkpoint.parent / f"benchmark_{checkpoint.stem}.json"
    if not result.is_file():
      raise ValueError("evaluation result is missing")
    return result

  def job_log(self, name: str, lines: int = 5000) -> dict:
    """The last ``lines`` lines of a job's output (from its last 8 MB) for the log viewer;
    ``size``: the whole log in bytes."""
    job = jobmod.find_job(name)
    if job is None:
      raise ValueError(f"no job {name!r}")
    return {
      "log": job.tail(max(1, min(int(lines), 50000)), nbytes=8 << 20),
      "size": job.log_size(),
      "status": job.status(),
    }

  def job_log_path(self, name: str) -> Path:
    job = jobmod.find_job(name)
    if job is None or not Path(job.log).is_file():
      raise ValueError(f"no log for job {name!r}")
    return Path(job.log)

  def _job_run_dir(self, job: jobmod.Job) -> str:
    if job.kind != "train":
      return ""
    d = job.run_dir()
    if d:
      return d
    # fall back: newest run folder of the experiment created after the job started
    exp = jobmod.logs_dir() / "rsl_rl" / job.meta.get("experiment", "")
    if exp.is_dir():
      cands = [
        p for p in exp.iterdir() if p.is_dir() and p.stat().st_mtime >= job.started - 5
      ]
      if cands:
        return str(max(cands, key=lambda p: p.stat().st_mtime))
    return ""

  def curves(self, *, job: str = "", run: str = "", group: str = "train") -> dict:
    """Scalar series of a job's run folder or of a run (``group`` of :data:`CURVE_GROUPS`,
    ``"rewards"``: every ``Episode_Reward/*`` term, ``"all"``: every tag)."""
    path = ""
    if job:
      j = jobmod.find_job(job)
      path = self._job_run_dir(j) if j else ""
    elif run:
      r = jobmod.find_run(run)
      path = str(r.path) if r else ""
    if not path or not Path(path).is_dir():
      return {"path": path, "series": {}}
    if group == "rewards":
      data = {
        k: v
        for k, v in jobmod.scalars(Path(path)).items()
        if k.startswith("Episode_Reward/")
      }
    elif group == "all":
      data = jobmod.scalars(Path(path))
    elif group == "loss":
      everything = jobmod.scalars(Path(path))
      data = {
        k: v
        for k, v in everything.items()
        if k.startswith("Loss/") and k not in LOSS_SKIP
      } | {k: everything[k] for k in CURVE_GROUPS["loss"] if k in everything}
    else:
      data = jobmod.scalars(Path(path), CURVE_GROUPS.get(group, CURVE_GROUPS["train"]))
    return {"path": path, "series": {k: {"x": x, "y": y} for k, (x, y) in data.items()}}

  # ------------------------------------------------------------------ polling
  def poll(self) -> None:
    """Emit ``jobs`` when a job changes state; a finished conversion refreshes ``data``."""
    changed = False
    for j in jobmod.list_jobs()[:60]:
      status = j.status()
      prev = self._status.get(j.name)
      if prev != status:
        self._status[j.name] = status
        if prev is not None:
          changed = True
          if status != "running" and not j.finished:
            j.finished = time.time()
            j.save()
          level = {"done": "ok", "failed": "error", "stopped": "warn"}.get(
            status, "info"
          )
          self.c.log(f"{j.kind} job {j.name}: {status}", level)
          if j.kind == "convert":
            self.c.emit("data", dataset=j.meta.get("dataset", ""))
      elif status == "running":
        changed = True  # progress moved
    if changed:
      self.c.emit("jobs")


def _cli_values(value) -> list[str]:
  """A setting's value as ``wbc-mjlab-train`` arguments (mjlab's CLI takes tuples as
  Python literals: ``"(512, 512)"``)."""
  if isinstance(value, (list, tuple)):
    return [str(tuple(value))]
  return [str(value)]


__all__ = ["CURVE_GROUPS", "TrainingService"]
