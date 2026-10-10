"""Add a robot from its model files (Studio *Robot → Add robot*): upload or copy a folder with
an MJCF / URDF and its meshes, inspect it (suggested wiring and PD gains), preview the wiring on
the model, run the settle check, save. Saved robots are :mod:`wbc_mjlab.robots.custom` robots:
registered in every process, trained and converted like built-in ones."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import mujoco
import numpy as np

from wbc_mjlab.robots import custom
from wbc_mjlab.studio.robots import ANCHOR_COLOR as ANCHOR
from wbc_mjlab.studio.robots import FOOT_COLOR as FOOT
from wbc_mjlab.studio.robots import KEYBODY_COLOR as KEYBODY
from wbc_mjlab.studio.scene import Overlay

OWNER = "import"
EE = np.array((90, 160, 255), dtype=np.uint8)


class RobotImportService:
  def __init__(self, core):
    self.c = core
    self.draft: str = ""  # robot id being added / edited
    self._model: mujoco.MjModel | None = None
    self._qpos: np.ndarray | None = None

  # ------------------------------------------------------------------ files
  def _folder(self, robot_id: str) -> Path:
    return custom.robots_dir() / custom.check_id(robot_id)

  def upload(self, robot_id: str, relpath: str, content: bytes) -> Path:
    """One file of an uploaded folder, at its relative path (meshes keep their layout)."""
    folder = self._folder(robot_id)
    parts = [p for p in Path(relpath).parts if p not in ("", ".", "..")]
    if not parts:
      raise ValueError("empty file name")
    dest = folder.joinpath(*parts).resolve()
    if folder.resolve() not in dest.parents:
      raise ValueError("file outside the robot folder")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest

  def import_path(self, robot_id: str, source: str) -> int:
    """Copy a folder on the Studio machine (model + meshes) into the robot folder."""
    src = Path(source).expanduser().resolve()
    if src.is_file():
      src = src.parent
    if not src.is_dir():
      raise ValueError(f"not a folder: {src}")
    dest = self._folder(robot_id)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return sum(1 for p in dest.rglob("*") if p.is_file())

  def files(self, robot_id: str) -> list[dict]:
    """Model files in the robot folder, the likely main model first."""
    folder = self._folder(robot_id)
    out = []
    for p in sorted(folder.rglob("*")) if folder.is_dir() else []:
      if p.suffix.lower() not in custom.MODEL_SUFFIXES or p.name == custom.PREPARED:
        continue
      text = p.read_text(errors="replace")[:200_000]
      if "<mujoco" not in text and "<robot" not in text:
        continue
      score = (
        ("freejoint" in text or 'type="free"' in text) * 4
        + text.count("<body")
        + text.count("<link")
      )
      score -= 50 * (p.stem.startswith("scene") or "<include" in text)
      out.append(
        {"path": str(p.relative_to(folder)), "score": score, "size": p.stat().st_size}
      )
    out.sort(key=lambda x: -x["score"])
    return out

  def drafts(self) -> list[dict]:
    """Robot folders without a saved spec yet (an upload that wasn't finished)."""
    root = custom.robots_dir()
    return (
      [
        {"id": d.name, "files": len(self.files(d.name))}
        for d in sorted(root.iterdir())
        if root.is_dir()
        if d.is_dir()
        and not (d / custom.SPEC_FILE).is_file()
        and custom.ID_RE.match(d.name)
      ]
      if root.is_dir()
      else []
    )

  # ------------------------------------------------------------------ inspect / preview
  def inspect(self, robot_id: str, model: str) -> dict:
    folder = self._folder(robot_id)
    path = (folder / model).resolve()
    if folder.resolve() not in path.parents or not path.is_file():
      raise ValueError(f"no model {model} in {folder}")
    info = custom.inspect_model(path)
    info["id"] = robot_id
    info["model"] = model
    info["task"] = custom.task_id_for(robot_id)
    try:
      saved = custom.load_spec(robot_id)
      info["saved"] = saved.to_dict() if saved.model == model else None
    except OSError:
      info["saved"] = None
    spec, _ = custom.prepare(path)
    self._model = spec.compile()
    self.draft = robot_id
    q = self._model.qpos0.copy()
    for j in info["joints"]:
      jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
      q[self._model.jnt_qposadr[jid]] = j["default"]
    q[0:3] = (0.0, 0.0, info["init_height"])
    q[3:7] = (1.0, 0.0, 0.0, 0.0)
    self._qpos = q
    self.c.show_model(self._model, OWNER)
    return info

  def preview(self, wiring: dict) -> None:
    """Show the draft in ``init_height`` / ``init_joint_pos`` with the wiring markers."""
    m = self._model
    if m is None or self.c.owner != OWNER:
      return
    q = self._qpos.copy()
    for name, v in (wiring.get("init_joint_pos") or {}).items():
      jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
      if jid >= 0:
        q[m.jnt_qposadr[jid]] = float(v)
    if "init_height" in wiring:
      q[2] = float(wiring["init_height"])
    self.c.show_qpos(q, OWNER)
    pos = self.c.scene.body_pos[0]

    def ids(names):
      return [
        b
        for b in (
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names or ()
        )
        if b >= 0
      ]

    for key, names, color, size in (
      ("keybodies", wiring.get("keybodies"), KEYBODY, 0.045),
      ("ee", wiring.get("ee_bodies"), EE, 0.06),
      ("feet", wiring.get("feet_bodies"), FOOT, 0.075),
      ("anchor", [wiring.get("anchor_body", "")], ANCHOR, 0.085),
    ):
      b = ids(names)
      self.c.overlay(
        OWNER,
        key,
        Overlay("points", pos[b], color, size, opacity=0.9, on_top=True) if b else None,
      )
    floor = np.array(
      [[-0.6, -0.6, 0.0], [0.6, -0.6, 0.0], [0.6, 0.6, 0.0], [-0.6, 0.6, 0.0]]
    )
    self.c.overlay(
      OWNER,
      "floor",
      Overlay(
        "segments",
        np.stack([floor, np.roll(floor, -1, axis=0)], axis=1),
        np.array((255, 176, 32), dtype=np.uint8),
        0.004,
        opacity=0.6,
      ),
    )

  # ------------------------------------------------------------------ check / save
  def _spec(self, data: dict) -> custom.CustomRobotSpec:
    d = dict(data)
    d["joints"] = {
      k: dataclasses.asdict(custom.JointGains(**v))
      for k, v in (d.get("joints") or {}).items()
    }
    return custom.CustomRobotSpec.from_dict(d)

  def check(self, data: dict) -> dict:
    return custom.settle_test(self._spec(data))

  @staticmethod
  def _forget_cached() -> None:
    """Cached per-robot / per-task configs (an edited robot builds differently)."""
    from wbc_mjlab.studio import presets, robots

    for fn in (
      robots._model_summary,
      robots.task_env_cfg,
      presets._library,
      presets._builtin_modules,
    ):  # noqa: SLF001
      fn.cache_clear()

  def save(self, data: dict) -> dict:
    spec = custom.save_spec(self._spec(data))
    self._forget_cached()
    self.draft = ""
    self._model = None
    self.c.log(
      f"robot {spec.id} saved: tasks {spec.task_id}, {spec.task_id}-SE, {spec.task_id}-Terrain-Scan",
      "ok",
    )
    self.c.emit("robots")
    self.c.select_robot(spec.id)
    return {"id": spec.id, "task": spec.task_id}

  def cancel(self) -> None:
    """Leave the import view: back to the selected robot."""
    if self.c.owner == OWNER or self._model is not None:
      self.draft = ""
      self._model = None
      if self.c.robot_id:
        self.c.select_robot(self.c.robot_id)

  def delete(self, robot_id: str) -> None:
    if robot_id == self.c.robot_id:
      others = [r for r in self.c.robot_names() if r != robot_id]
      if others:
        self.c.select_robot(others[0])
    custom.delete(robot_id)
    self._forget_cached()
    from wbc_mjlab.robots import ids

    if robot_id in ids._KNOWN_ROBOT_IDS:  # noqa: SLF001
      ids._KNOWN_ROBOT_IDS.remove(robot_id)  # noqa: SLF001
    self.c.log(f"robot {robot_id} deleted (its data/ and logs/ stay)", "ok")
    self.c.emit("robots")


__all__ = ["RobotImportService"]
