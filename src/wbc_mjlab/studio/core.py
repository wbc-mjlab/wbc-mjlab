"""Studio core: UI-agnostic state and services shared by the web and the viser frontend.

``StudioCore`` owns the selected robot (its MJCF model), the :class:`SceneState`, the log,
one background task at a time and the services:

  robots    – registered robots (built-in + extensions), joint sliders, tracked keybodies
  data      – datasets per robot, clip browser, preview (NPZ and raw CSV/PKL), clip QA,
              conversion / import jobs
  training  – tasks, detached training jobs, runs, curves, TensorBoard
  play      – a checkpoint on N parallel envs over a motion library, reference ghost,
              live reward terms and tracking errors

Frontends call service methods and subscribe to events (``core.on(topic, fn)``); the
services never import a UI toolkit. Topics: ``log``, ``busy``, ``robot``, ``robots``,
``data``, ``jobs``, ``play``, ``owner``.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from collections.abc import Callable

import mujoco
import numpy as np

from wbc_mjlab.studio.scene import Kinematics, Overlay, SceneState

CONTROL_FPS = 50.0


def discover_robots() -> list[str]:
  """Registered robot ids: built-in G1 plus every extension loaded via ``mjlab.tasks``."""
  import mjlab  # noqa: F401 — runs the mjlab.tasks entry points (extension robots)

  from wbc_mjlab.robots.ids import known_robot_ids
  from wbc_mjlab.tasks import list_robot_ids_from_tasks

  return sorted(set(known_robot_ids()) | set(list_robot_ids_from_tasks()))


def load_robot_model(robot_id: str) -> mujoco.MjModel:
  from wbc_mjlab.motion.robot_assets import get_robot_mjcf_path

  return mujoco.MjModel.from_xml_path(str(get_robot_mjcf_path(robot_id)))


class StudioCore:
  def __init__(self, robot: str | None = None, device: str = "cuda:0"):
    from wbc_mjlab.studio.data import DataService
    from wbc_mjlab.studio.play import PlayService
    from wbc_mjlab.studio.robot_import import RobotImportService
    from wbc_mjlab.studio.robots import RobotsService
    from wbc_mjlab.studio.tensorboard import TensorBoard
    from wbc_mjlab.studio.training import TrainingService

    self.device = device
    self.scene = SceneState()
    self.logs: deque[dict] = deque(maxlen=400)
    self.task: str | None = None
    self.owner = "robots"  # which service drives the scene
    self.model: mujoco.MjModel | None = None
    self.kin: Kinematics | None = None
    self.robot_id = ""
    self._listeners: dict[str, list[Callable]] = {}
    self._lock = threading.Lock()
    self._joint_qpos: dict[str, int] = {}
    robots = discover_robots()
    self.robots = RobotsService(self)
    self.data = DataService(self)
    self.training = TrainingService(self)
    self.play = PlayService(self)
    self.importer = RobotImportService(self)
    self.tensorboard = TensorBoard()  # started on demand
    if robots:
      self.select_robot(robot if robot in robots else robots[0])
    self.log("studio ready")

  # ------------------------------------------------------------------ events
  def on(self, topic: str, fn: Callable) -> None:
    """``fn(**payload)`` on ``topic``; ``topic="*"`` gets every event as ``fn(topic, payload)``."""
    self._listeners.setdefault(topic, []).append(fn)

  def emit(self, topic: str, **payload) -> None:
    calls = [(fn, False) for fn in self._listeners.get(topic, ())]
    calls += [(fn, True) for fn in self._listeners.get("*", ())]
    for fn, wildcard in calls:
      try:
        if wildcard:
          fn(topic, payload)
        else:
          fn(**payload)
      except Exception as exc:  # noqa: BLE001
        print(f"[studio] listener for {topic!r} failed: {exc}")

  # ------------------------------------------------------------------ log / tasks
  def log(self, text: str, level: str = "info") -> None:
    for line in str(text).splitlines() or [""]:
      entry = {"t": time.time(), "text": line, "level": level}
      self.logs.append(entry)
      self.emit("log", entry=entry)

  @property
  def busy(self) -> bool:
    return self.task is not None

  def progress(self, stage: str, done: int, total: int) -> None:
    """Progress of the running task (frontends show a bar)."""
    self.emit("busy", task=self.task, stage=stage, fraction=done / max(1, total))

  def run_task(
    self, name: str, fn: Callable[[], None], *, done: Callable | None = None
  ) -> bool:
    """Run ``fn`` in a worker thread (one task at a time). False if another task runs."""
    with self._lock:
      if self.task is not None:
        self.log(f"{name}: busy with “{self.task}”, try again when it finishes", "warn")
        return False
      self.task = name
    self.emit("busy", task=name)

    def worker():
      t0 = time.time()
      try:
        fn()
        self.log(f"{name}: done in {time.time() - t0:.1f} s", "ok")
      except Exception as exc:  # noqa: BLE001
        self.log(f"{name}: {exc}", "error")
        print(traceback.format_exc())
      finally:
        self.task = None
        self.emit("busy", task=None)
        if done is not None:
          try:
            done()
          except Exception as exc:  # noqa: BLE001
            self.log(f"{name}: refresh failed: {exc}", "error")

    threading.Thread(target=worker, daemon=True).start()
    return True

  # ------------------------------------------------------------------ robot
  def robot_names(self) -> list[str]:
    return discover_robots()

  def select_robot(self, name: str) -> None:
    """Load the robot's MJCF; resets the scene to its default pose. A Play session of another
    robot is closed (its checkpoints belong to that robot)."""
    if self.play.session is not None and self.play.robot != name:
      self.play.stop()
    model = load_robot_model(name)
    self.robot_id = name
    self.model = model
    self.kin = Kinematics(model)
    self._joint_qpos = {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): int(model.jnt_qposadr[j])
      for j in range(model.njnt)
      if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    }
    self.scene.remove_overlays(prefix="")
    self.scene.set_terrain(None)
    with self.scene.lock:
      self.scene.model_version += 1
    self.owner = "robots"
    self.robots.on_robot()
    self.data.on_robot()
    self.emit("robot", name=name)
    self.log(
      f"robot {name}: {model.nq} qpos, {model.nbody - 1} bodies, {model.nv - 6 if model.nv >= 6 else model.nv} dof"
    )

  def show_model(self, model: mujoco.MjModel, owner: str) -> None:
    """Show a model that isn't a registered robot (a robot being added) and hand the scene
    to ``owner``; data / play playback pause. ``select_robot`` brings the robot back."""
    self.data.playing = False
    self.play.playing = False
    self.model = model
    self.kin = Kinematics(model)
    self._joint_qpos = {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): int(model.jnt_qposadr[j])
      for j in range(model.njnt)
      if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    }
    self.scene.remove_overlays(prefix="")
    self.scene.set_terrain(None)
    with self.scene.lock:
      self.scene.model_version += 1
    self.owner = owner
    self.scene.set_hud({})
    self.emit("owner", owner=owner)

  # ------------------------------------------------------------------ scene
  def take_scene(self, owner: str) -> None:
    """Hand the scene to ``owner``: overlays of the previous owner are removed."""
    if owner == self.owner:
      return
    self.scene.remove_overlays(prefix=f"{self.owner}/")
    self.scene.set_terrain(None)
    self.owner = owner
    self.emit("owner", owner=owner)

  def show_qpos(
    self,
    qpos: np.ndarray,
    owner: str,
    *,
    ghost: np.ndarray | None = None,
    labels: list[str] | None = None,
  ) -> None:
    """Pose the robot instances (``(nq,)`` or ``(N, nq)``) and optional reference ghosts if
    ``owner`` holds the scene."""
    if owner != self.owner or self.kin is None:
      return
    ghost_poses = self.kin(ghost) if ghost is not None else None
    self.scene.set_poses(*self.kin(qpos), ghost=ghost_poses, labels=labels)

  def overlay(self, owner: str, key: str, overlay: Overlay | None) -> None:
    if owner != self.owner and overlay is not None:
      return
    if overlay is None:
      self.scene.remove_overlays(f"{owner}/{key}")
    else:
      self.scene.set_overlay(f"{owner}/{key}", overlay)

  def set_focus(self, index: int) -> None:
    with self.scene.lock:
      self.scene.focus = max(0, min(int(index), max(0, self.scene.num_instances - 1)))
      self.scene.pose_version += 1
    self.emit("focus", focus=self.scene.focus)

  def qpos_from_named(self, root_pos, root_quat, joint_names, joint_pos) -> np.ndarray:
    """Model ``qpos`` rows from free-root poses + joint values given by name (batched)."""
    root_pos, root_quat = np.atleast_2d(root_pos), np.atleast_2d(root_quat)
    joint_pos = np.atleast_2d(joint_pos)
    q = np.tile(self.model.qpos0, (root_pos.shape[0], 1))
    q[:, 0:3], q[:, 3:7] = root_pos, root_quat
    cols, adrs = [], []
    for i, name in enumerate(joint_names):
      adr = self._joint_qpos.get(str(name).removeprefix("robot/"))
      if adr is not None:
        cols.append(i)
        adrs.append(adr)
    q[:, adrs] = joint_pos[:, cols]
    return q

  # ------------------------------------------------------------------ loop
  def tick(self, dt: float) -> None:
    for svc in (self.data, self.play):
      try:
        svc.tick(dt)
      except Exception as exc:  # noqa: BLE001
        # pause the service: a failing step would otherwise error at the control rate
        svc.playing = False
        self.log(f"{type(svc).__name__}: {exc} (paused)", "error")
        print(traceback.format_exc())

  def start(self) -> None:
    """Service ticks at the control rate (playback) and job polling every 2 s, in a daemon
    thread. Ticks run on a fixed deadline: the sleep is only what is left of the period."""

    def loop():
      period = 1.0 / CONTROL_FPS
      last = slow = deadline = time.perf_counter()
      while True:
        now = time.perf_counter()
        self.tick(now - last)
        last = now
        if now - slow > 2.0:
          slow = now
          try:
            self.training.poll()
          except Exception as exc:  # noqa: BLE001
            self.log(f"jobs: {exc}", "error")
        deadline += period
        delay = deadline - time.perf_counter()
        if delay > 0:
          time.sleep(delay)
        elif (
          delay < -4 * period
        ):  # far behind (a reset, a slow step): don't try to catch up
          deadline = time.perf_counter()

    threading.Thread(target=loop, daemon=True, name="studio-loop").start()

  def close(self) -> None:
    """Release what outlives the UI server: the playback env (GPU) and TensorBoard (its port).
    Detached training / conversion jobs keep running."""
    for stop in (self.play.stop, self.tensorboard.stop):
      try:
        stop()
      except Exception:  # noqa: BLE001
        pass


__all__ = ["CONTROL_FPS", "StudioCore", "discover_robots", "load_robot_model"]
