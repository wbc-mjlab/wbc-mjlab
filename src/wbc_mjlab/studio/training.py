"""Training service: WBC tasks of the robot, detached ``wbc-mjlab-train`` jobs, runs, curves.

A job is ``wbc-mjlab-train --task <task> --dataset <dataset>`` plus the run options picked in
the Studio (env count, iterations, run name, resume from a checkpoint of the same task),
started detached so it outlives the Studio. Curves come from the run's TensorBoard event
files; one TensorBoard over ``logs/rsl_rl`` starts on demand.
"""

from __future__ import annotations

import re
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
  "loss": ("Loss/value", "Loss/surrogate", "Policy/mean_std", "Perf/total_fps"),
}


class TrainingService:
  def __init__(self, core):
    self.c = core
    self._status: dict[str, str] = {}

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

  # ------------------------------------------------------------------ runs
  def runs(self, robot: str | None = None) -> list[dict]:
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
    extra: str = "",
  ) -> jobmod.Job:
    from wbc_mjlab.studio import presets
    from wbc_mjlab.tasks import get_task_config

    preset = presets.split_task_id(task)
    if preset:
      presets.register(preset[1])  # validates the preset and learns its experiment name
    cfg = get_task_config(task)
    if not dataset:
      raise ValueError("pick a dataset")
    from wbc_mjlab.data_paths import resolve_dataset_root
    from wbc_mjlab.motion.stack_bundle import list_clip_npz_files

    root = resolve_dataset_root(cfg.robot_id, dataset)
    if not list_clip_npz_files(root):
      raise ValueError(
        f"{dataset} has no converted clips yet: convert it in Data first"
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
    ]
    if run_name:
      args += ["--agent.run-name", re.sub(r"[^\w.-]+", "-", run_name)]
    if logger == "wandb":
      args += ["--use_wandb"]
    if resume_run:
      run = jobmod.find_run(resume_run)
      if run is None:
        raise ValueError(f"no run {resume_run!r}")
      if run.task_id() != cfg.task_id:
        raise ValueError(
          f"{resume_run} was trained with {run.task_id()}, not {cfg.task_id}"
        )
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
        "resume": f"{resume_run}/{resume_checkpoint}" if resume_run else "",
        "experiment": cfg.experiment_name,
      },
    )
    self.c.log(f"training {cfg.task_id} on {dataset}: job {job.name}", "ok")
    self.c.emit("jobs")
    return job

  def stop(self, name: str) -> None:
    job = jobmod.find_job(name)
    if job is None:
      raise ValueError(f"no job {name!r}")
    jobmod.stop(job)
    self.c.log(f"stopped {name}", "warn")
    self.c.emit("jobs")

  def delete(self, name: str) -> None:
    job = jobmod.find_job(name)
    if job is not None and job.alive:
      raise ValueError("stop the job first")
    jobmod.delete_job(name)
    self.c.emit("jobs")

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

  def _job_run_dir(self, job: jobmod.Job) -> str:
    if job.kind != "train":
      return ""
    d = job.progress().get("log_dir", "")
    if d:
      p = Path(d)
      return str(p if p.is_absolute() else jobmod.logs_dir().parent / p)
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


__all__ = ["CURVE_GROUPS", "TrainingService"]
