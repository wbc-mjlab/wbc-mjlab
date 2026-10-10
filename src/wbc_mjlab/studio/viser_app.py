"""Viser frontend of the Studio: a viewer block (transport, live readout, display options), the
pipeline tabs (Robot · Data · Train · Play), the activity log and the 3D scene.

All state lives in :class:`StudioCore`; this module only maps it to widgets. Robots and
reference ghosts are drawn with one batched mesh per (body, colour) of the robot model, so N
instances cost one draw call each (the same idea as the web UI's instancing).
"""

from __future__ import annotations

import html
import threading
import time
import webbrowser

import mujoco
import numpy as np
import trimesh
import viser

from wbc_mjlab.studio.core import StudioCore
from wbc_mjlab.studio.play import ZERO

# wbc-mjlab brand (wbc-demo tokens.css, dark)
ACCENT = (54, 207, 224)
AMBER = (230, 162, 60)
FLOOR = (29, 52, 71)
GRID_MINOR = (47, 122, 138)
GRID_MAJOR = (66, 158, 176)
TEXT = "#e3edf6"
MUTED = "#8fa0b3"
NONE = "—"
VISIBLE_GROUPS = 3
MAX_CLIPS = 1500
ENVS = ("1", "2", "4", "8", "16", "32", "64")
SPEEDS = ("0.25×", "0.5×", "1×", "2×", "4×")


# ---------------------------------------------------------------------- html helpers
def _box(inner: str) -> str:
  return f'<div style="font-size:0.8em;line-height:1.45;color:{TEXT};padding:0 0.2em 0.4em">{inner}</div>'


def _micro(text: str) -> str:
  return (
    f'<span style="font-family:monospace;font-size:0.72em;letter-spacing:0.14em;'
    f'text-transform:uppercase;color:{MUTED}">{html.escape(text)}</span>'
  )


def _kv(rows: list[tuple[str, str]]) -> str:
  cells = "".join(
    f"<tr><td style='padding:1px 10px 1px 0;vertical-align:top'>{_micro(k)}</td>"
    f"<td style='padding:1px 0'>{v}</td></tr>"
    for k, v in rows
  )
  return f"<table style='border-collapse:collapse'>{cells}</table>"


def _issue(level: str, text: str) -> str:
  color = {
    "ok": "#4fc78a",
    "warn": "#e6a23c",
    "error": "#ef6a6a",
    "info": "#36cfe0",
  }.get(level, MUTED)
  return f'<div style="margin:2px 0"><span style="color:{color}">●</span> {html.escape(text)}</div>'


def _bars(values: dict[str, float], limit: int = 18) -> str:
  items = sorted(
    ((k, v) for k, v in values.items() if v is not None), key=lambda kv: -abs(kv[1])
  )
  items = items[:limit]
  if not items:
    return _box(f"<i style='color:{MUTED}'>no step yet</i>")
  scale = max(1e-6, *(abs(v) for _, v in items))
  rows = []
  for k, v in items:
    color = "#ef6a6a" if v < 0 else "#36cfe0"
    rows.append(
      f"<div style='display:grid;grid-template-columns:1.3fr 52px 1fr;gap:6px;align-items:center'>"
      f"<span style='font-family:monospace;font-size:0.85em;color:{MUTED};overflow:hidden;"
      f"text-overflow:ellipsis;white-space:nowrap'>{html.escape(k)}</span>"
      f"<span style='font-family:monospace;font-size:0.85em;text-align:right'>{v:+.3f}</span>"
      f"<span style='height:5px;border-radius:3px;background:#111c27'><span style='display:block;"
      f"height:5px;border-radius:3px;background:{color};width:{100 * abs(v) / scale:.0f}%'></span>"
      f"</span></div>"
    )
  return _box("".join(rows))


# ---------------------------------------------------------------------- batched robot
class BatchedRobot:
  """The robot model as one batched mesh per (body, colour), posed for N instances."""

  def __init__(
    self,
    server: viser.ViserServer,
    model: mujoco.MjModel,
    prefix: str,
    *,
    tint=None,
    opacity=None,
  ):
    from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

    self.server = server
    self.prefix = prefix
    self.handles: list[tuple[int, object]] = []
    self.n = 0
    parts: dict[tuple[int, tuple], list[trimesh.Trimesh]] = {}
    for g in range(model.ngeom):
      if (
        model.geom_group[g] >= VISIBLE_GROUPS
        or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
      ):
        continue
      mat = int(model.geom_matid[g])
      rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
      if model.geom_rgba[g, 3] <= 0 or rgba[3] <= 0:
        continue
      try:
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
          mesh = mujoco_mesh_to_trimesh(model, g)
        else:
          mesh = create_primitive_mesh(model, g)
      except Exception:  # noqa: BLE001
        continue
      if mesh is None:
        continue
      mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
      tf = np.eye(4)
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, model.geom_quat[g])
      tf[:3, :3] = rot.reshape(3, 3)
      tf[:3, 3] = model.geom_pos[g]
      mesh.apply_transform(tf)
      color = tuple(int(255 * c) for c in rgba[:3]) if tint is None else tint
      parts.setdefault((int(model.geom_bodyid[g]), color), []).append(mesh)
    self.parts = [
      (body, color, trimesh.util.concatenate(ms)) for (body, color), ms in parts.items()
    ]
    self.opacity = opacity

  def update(self, pos: np.ndarray, quat: np.ndarray) -> None:
    """``pos`` (N, nbody, 3), ``quat`` (N, nbody, 4) wxyz; N = 0 hides the set."""
    n = int(pos.shape[0])
    if n != self.n:
      self.remove()
      self.n = n
      if n == 0:
        return
      for i, (body, color, mesh) in enumerate(self.parts):
        h = self.server.scene.add_batched_meshes_simple(
          f"{self.prefix}/part{i}",
          vertices=np.asarray(mesh.vertices, np.float32),
          faces=np.asarray(mesh.faces, np.uint32),
          batched_wxyzs=quat[:, body].astype(np.float32),
          batched_positions=pos[:, body].astype(np.float32),
          batched_colors=color,
          opacity=self.opacity,
          cast_shadow=self.opacity is None,
          receive_shadow=self.opacity is None,
          lod="off",
        )
        self.handles.append((body, h))
      return
    if n == 0:
      return
    for body, h in self.handles:
      h.batched_positions = pos[:, body].astype(np.float32)
      h.batched_wxyzs = quat[:, body].astype(np.float32)

  def remove(self) -> None:
    for _, h in self.handles:
      h.remove()
    self.handles.clear()
    self.n = 0


# ---------------------------------------------------------------------- app
class ViserStudio:
  def __init__(self, core: StudioCore, host: str = "127.0.0.1", port: int = 8765):
    self.c = core
    self.server = viser.ViserServer(host=host, port=port, label="wbc studio")
    self.url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    core.tensorboard.host = host
    self._pending: set[str] = set()
    self._plock = threading.Lock()
    self._overlay_handles: dict[str, tuple] = {}
    self._robot: BatchedRobot | None = None
    self._ghost: BatchedRobot | None = None
    self._model_version = -1
    self._seen_pose = self._seen_overlay = -1
    self._hud: dict = {}
    self._follow_target: np.ndarray | None = None
    self._cam_offsets: dict[int, np.ndarray] = {}
    self._joint_widgets: list = []
    self._syncing = False
    self._libs: dict[str, str] = {}
    self._build()
    core.on("*", self._on_event)
    self._refresh_all()

  # ================================================================== layout
  def _build(self) -> None:
    gui = self.server.gui
    gui.configure_theme(
      titlebar_content=_titlebar(),
      dark_mode=True,
      show_logo=False,
      show_share_button=False,
      brand_color=ACCENT,
      control_width="large",
    )
    gui.set_panel_label("WBC-MJLab Studio")
    self.server.scene.set_up_direction("+z")
    self._add_ground()
    self.header = gui.add_html("")
    self.task_bar = gui.add_progress_bar(
      0.0, animated=True, visible=False, color=ACCENT
    )
    with gui.add_folder("Viewer"):
      self._build_viewer(gui)
    tabs = gui.add_tab_group()
    with tabs.add_tab("Robot", viser.Icon.ROBOT):
      self._build_robot(gui)
    with tabs.add_tab("Data", viser.Icon.DATABASE):
      self._build_data(gui)
    with tabs.add_tab("Train", viser.Icon.BOLT):
      self._build_train(gui)
    with tabs.add_tab("Play", viser.Icon.PLAYER_PLAY):
      self._build_play(gui)
    with gui.add_folder("Activity", expand_by_default=False):
      self.log_html = gui.add_html("")

  def _add_ground(self) -> None:
    self.server.scene.add_grid(
      "/ground",
      width=60.0,
      height=60.0,
      plane="xy",
      infinite_grid=True,
      cell_size=0.5,
      section_size=1.0,
      cell_color=GRID_MINOR,
      section_color=GRID_MAJOR,
      fade_distance=40.0,
      plane_color=FLOOR,
      plane_opacity=1.0,
      shadow_opacity=0.25,
    )

  # ------------------------------------------------------------------ robot
  def _build_robot(self, gui) -> None:
    robots = self.c.robot_names() or [NONE]
    self.robot_dd = gui.add_dropdown(
      "Robot", robots, initial_value=self.c.robot_id or robots[0]
    )

    @self.robot_dd.on_update
    def _(_e) -> None:
      if self._syncing or self.robot_dd.value in (NONE, self.c.robot_id):
        return
      self.c.robots.select(self.robot_dd.value)

    self.robot_html = gui.add_html("")
    self.wiring_cb = gui.add_checkbox(
      "Show anchor · keybodies · feet", initial_value=True
    )

    @self.wiring_cb.on_update
    def _(_e) -> None:
      self.c.robots.set_options(show_wiring=self.wiring_cb.value)

    row = gui.add_button_group("Pose", ("Default", "Zero"))

    @row.on_click
    def _(e) -> None:
      self.c.robots.reset_pose(zero=e.target.value == "Zero")
      self._rebuild_joints()

    self.joint_folder = gui.add_folder("Joint sliders", expand_by_default=False)

  def _rebuild_joints(self) -> None:
    for w in self._joint_widgets:
      w.remove()
    self._joint_widgets = []
    info = self.c.robots.info()
    with self.joint_folder:
      for j in info.get("joints", []):
        s = self.server.gui.add_slider(
          j["name"].removesuffix("_joint"),
          min=j["min"],
          max=j["max"],
          step=0.01,
          initial_value=float(np.clip(j["value"], j["min"], j["max"])),
        )

        def make(name, slider):
          @slider.on_update
          def _(_e) -> None:
            self.c.robots.set_joint(name, slider.value)

        make(j["name"], s)
        self._joint_widgets.append(s)

  def _refresh_robot(self) -> None:
    info = self.c.robots.info()
    w = info.get("wiring", {})
    tasks = "<br/>".join(
      f"<b>{html.escape(t['id'])}</b> — {html.escape(t['description'])}"
      for t in info.get("tasks", [])
    )
    self.robot_html.content = _box(
      _kv(
        [
          ("dof", str(info.get("dof", NONE))),
          ("bodies", str(info.get("nbody", 1) - 1)),
          ("mass", f"{info.get('mass', 0):.1f} kg"),
          (
            "mjcf",
            f"<code>{html.escape('/'.join(info.get('mjcf', '').split('/')[-3:]))}</code>",
          ),
          ("anchor", f"<code>{html.escape(w.get('anchor', NONE))}</code>"),
          ("keybodies", str(len(w.get("bodies", [])))),
          ("feet", ", ".join(w.get("feet", [])) or NONE),
        ]
      )
      + f"<div style='margin-top:6px'>{_micro('tasks')}<br/>{tasks}</div>"
    )
    self._rebuild_joints()

  # ------------------------------------------------------------------ data
  def _build_data(self, gui) -> None:
    self.ds_dd = gui.add_dropdown("Dataset", [NONE])
    self.ds_html = gui.add_html("")
    self.search = gui.add_text(
      "Search", initial_value="", hint="substring or *wildcard*"
    )
    self.status_dd = gui.add_dropdown("Show", ("all", "converted", "missing"))
    self.clip_dd = gui.add_dropdown("Clip", [NONE])
    load = gui.add_button_group("Preview", ("Converted", "Raw source"))
    self.grid_n = gui.add_slider("Grid clips", min=4, max=49, step=1, initial_value=16)
    grid = gui.add_button("Grid preview of this search", icon=viser.Icon.LAYOUT_GRID)
    self.qa_html = gui.add_html("")
    with gui.add_folder("Convert", expand_by_default=False):
      self.batch_n = gui.add_number("Batch (GPU envs)", initial_value=8, min=1, step=1)
      self.out_fps = gui.add_number("Output Hz", initial_value=50.0, min=10.0, step=1.0)
      self.mirror_cb = gui.add_checkbox("Mirror", initial_value=False)
      self.debias_cb = gui.add_checkbox("Debias z", initial_value=False)
      conv = gui.add_button_group("Convert", ("Missing", "All"))
      scan = gui.add_button("Scan durations", icon=viser.Icon.SCAN)
    with gui.add_folder("Import folder", expand_by_default=False):
      self.imp_src = gui.add_text("Folder", initial_value="")
      self.imp_name = gui.add_text("Dataset name", initial_value="")
      self.imp_link = gui.add_checkbox("Link (don't copy)", initial_value=False)
      imp = gui.add_button("Import", icon=viser.Icon.FOLDER_PLUS)
    self.data_jobs = gui.add_html("")

    @self.ds_dd.on_update
    def _(_e) -> None:
      if not self._syncing and self.ds_dd.value != NONE:
        self.c.data.dataset = self.ds_dd.value
        self._refresh_clips()
        self._refresh_ds_html()

    @self.search.on_update
    def _(_e) -> None:
      self._refresh_clips()

    @self.status_dd.on_update
    def _(_e) -> None:
      self._refresh_clips()

    @self.clip_dd.on_update
    def _(_e) -> None:
      if not self._syncing and self.clip_dd.value != NONE:
        self._guard(lambda: self.c.data.load(self.ds_dd.value, self.clip_dd.value))

    @load.on_click
    def _(e) -> None:
      if self.clip_dd.value != NONE:
        src = "raw" if e.target.value == "Raw source" else "npz"
        self._guard(lambda: self.c.data.load(self.ds_dd.value, self.clip_dd.value, src))

    @grid.on_click
    def _(_e) -> None:
      self._guard(
        lambda: self.c.data.load_grid(
          self.ds_dd.value,
          self.search.value,
          self.status_dd.value,
          int(self.grid_n.value),
        )
      )

    @conv.on_click
    def _(e) -> None:
      self._guard(
        lambda: self.c.data.convert(
          self.ds_dd.value,
          "all" if e.target.value == "All" else "missing",
          batch_size=int(self.batch_n.value),
          output_fps=float(self.out_fps.value),
          mirror=self.mirror_cb.value,
          debias_z=self.debias_cb.value,
        )
      )

    @scan.on_click
    def _(_e) -> None:
      self.c.data.scan(self.ds_dd.value)

    @imp.on_click
    def _(_e) -> None:
      self._guard(
        lambda: self.c.data.import_folder(
          self.imp_src.value, self.imp_name.value, self.imp_link.value
        )
      )

  def _refresh_datasets(self) -> None:
    names = [d["name"] for d in self.c.data.datasets()] or [NONE]
    self._syncing = True
    try:
      self.ds_dd.options = names
      self.ds_dd.value = (
        self.c.data.dataset if self.c.data.dataset in names else names[0]
      )
    finally:
      self._syncing = False
    self._refresh_ds_html()
    self._refresh_clips()

  def _refresh_ds_html(self) -> None:
    d = next((x for x in self.c.data.datasets() if x["name"] == self.ds_dd.value), None)
    if d is None:
      self.ds_html.content = _box(f"<i style='color:{MUTED}'>no dataset</i>")
      return
    s = d["stats"]
    rows = [
      ("clips", f"{d['clips']} · {d['raw']} source · {d['npz']} npz"),
      (
        "missing",
        f"<span style='color:{'#e6a23c' if d['missing'] else '#4fc78a'}'>{d['missing']}</span>",
      ),
      ("folder", f"<code>{html.escape(d['path'].split('/data/')[-1])}</code>"),
    ]
    if s:
      rows.append(("hours", f"{s['hours']:.2f} h · mean {s['mean_s']:.1f} s"))
    self.ds_html.content = _box(_kv(rows))

  def _refresh_clips(self) -> None:
    if self.ds_dd.value == NONE:
      return
    try:
      names = self.c.data.filtered(
        self.ds_dd.value, self.search.value, self.status_dd.value
      )
    except Exception:  # noqa: BLE001
      names = []
    opts = names[:MAX_CLIPS] or [NONE]
    self._syncing = True
    try:
      self.clip_dd.options = opts
      if self.c.data.state()["clip"] in opts:
        self.clip_dd.value = self.c.data.state()["clip"]
    finally:
      self._syncing = False

  def _refresh_qa(self) -> None:
    p = self.c.data.state()
    if not p["active"]:
      self.qa_html.content = ""
      return
    if p["mode"] == "grid":
      self.qa_html.content = _box(
        f"{_micro('grid')} {p['count']} clips · one clock, shorter ones loop"
      )
      return
    qa = p["qa"]
    head = _kv(
      [
        ("clip", f"<b>{html.escape(p['clip'])}</b> · {p['kind']}"),
        (
          "length",
          f"{qa['duration']:.2f} s · {qa['frames']} frames @ {qa['fps']:g} Hz",
        ),
        ("root z", f"{qa['root_z'][0]:.2f} – {qa['root_z'][1]:.2f} m"),
        ("speed", f"{qa['speed_mean']:.2f} m/s (max {qa['speed_max']:.1f})"),
      ]
    )
    self.qa_html.content = _box(
      head + "".join(_issue(i["level"], i["text"]) for i in qa["issues"])
    )

  # ------------------------------------------------------------------ train
  def _build_train(self, gui) -> None:
    self.task_dd = gui.add_dropdown("Task", [NONE])
    self.task_html = gui.add_html("")
    self.train_ds = gui.add_dropdown("Dataset", [NONE])
    self.train_envs = gui.add_number("Envs", initial_value=4096, min=1, step=256)
    self.train_iters = gui.add_number(
      "Iterations", initial_value=30000, min=1, step=1000
    )
    self.train_name = gui.add_text("Run name", initial_value="")
    self.train_resume = gui.add_dropdown("Continue from", ["fresh run"])
    launch = gui.add_button(
      "Launch training", icon=viser.Icon.PLAYER_PLAY, color=ACCENT
    )
    self.tb = gui.add_button("Open TensorBoard", icon=viser.Icon.CHART_LINE)
    self.jobs_html = gui.add_html("")
    stop = gui.add_button(
      "Stop newest running job", icon=viser.Icon.PLAYER_STOP, color="red"
    )

    @self.task_dd.on_update
    def _(_e) -> None:
      self._refresh_train_form()

    @launch.on_click
    def _(_e) -> None:
      resume = "" if self.train_resume.value == "fresh run" else self.train_resume.value
      self._guard(
        lambda: self.c.training.launch(
          self.task_dd.value,
          self.train_ds.value,
          num_envs=int(self.train_envs.value),
          max_iterations=int(self.train_iters.value),
          run_name=self.train_name.value,
          resume_run=resume,
        )
      )

    @self.tb.on_click
    def _(_e) -> None:
      try:
        port = self.c.tensorboard.start()
        self.c.log(f"TensorBoard on http://{self.c.tensorboard.host}:{port}", "ok")
        webbrowser.open(f"http://127.0.0.1:{port}")
      except Exception as exc:  # noqa: BLE001
        self.c.log(f"tensorboard: {exc}", "error")

    @stop.on_click
    def _(_e) -> None:
      running = [j for j in self.c.training.jobs("train") if j["status"] == "running"]
      if running:
        self.c.training.stop(running[0]["name"])

  def _refresh_train_form(self) -> None:
    tasks = self.c.training.tasks()
    ids = [t["id"] for t in tasks] or [NONE]
    self._syncing = True
    try:
      if list(self.task_dd.options) != ids:
        self.task_dd.options = ids
      t = next((x for x in tasks if x["id"] == self.task_dd.value), None)
      self.task_html.content = _box(html.escape(t["description"])) if t else ""
      ds = [d["name"] for d in self.c.data.datasets() if d["npz"]] or [NONE]
      self.train_ds.options = ds
      runs = ["fresh run"] + [
        r["label"]
        for r in self.c.training.runs(self.c.robot_id)
        if r["task"] == self.task_dd.value and not r["demo"]
      ]
      self.train_resume.options = runs
    finally:
      self._syncing = False

  def _refresh_jobs(self) -> None:
    rows = []
    for j in self.c.training.jobs()[:10]:
      pr = j["progress"]
      color = {
        "running": "#36cfe0",
        "done": "#4fc78a",
        "failed": "#ef6a6a",
        "stopped": "#e6a23c",
      }[j["status"]]
      what = (
        f"{j['meta'].get('task', '')} · {j['meta'].get('dataset', '')}"
        if j["kind"] == "train"
        else f"convert {j['meta'].get('dataset', '')}"
      )
      prog = ""
      if "fraction" in pr:
        prog = (
          f"<div style='height:4px;border-radius:2px;background:#233444;margin-top:3px'><div style='height:4px;"
          f"border-radius:2px;background:{color};width:{100 * pr['fraction']:.0f}%'></div></div>"
        )
      extra = (
        f" · it {pr['iteration']}/{pr['total']}"
        if "iteration" in pr
        else (f" · {pr.get('done', 0)}/{pr['total']}" if "total" in pr else "")
      )
      if pr.get("mean_reward"):
        extra += f" · reward {float(pr['mean_reward']):.2f}"
      rows.append(
        f"<div style='padding:5px 0;border-bottom:1px solid #233444'><span style='color:{color}'>●</span> "
        f"<b>{html.escape(what)}</b> <span style='color:{MUTED}'>{j['status']} · {j['started_text']}{extra}</span>{prog}</div>"
      )
    content = "".join(rows) or f"<i style='color:{MUTED}'>no jobs yet</i>"
    self.jobs_html.content = _box(content)
    self.data_jobs.content = _box("".join(r for r in rows if "convert" in r))

  # ------------------------------------------------------------------ play
  def _build_play(self, gui) -> None:
    self.run_dd = gui.add_dropdown("Run", [NONE])
    self.ckpt_dd = gui.add_dropdown("Checkpoint", [NONE])
    self.lib_dd = gui.add_dropdown("Motion library", [NONE])
    self.envs_dd = gui.add_dropdown("Envs", ENVS, initial_value="8")
    self.term_cb = gui.add_checkbox("Terminations", initial_value=False)
    apply = gui.add_button_group("Session", ("Load / apply", "Stop"))
    self.play_html = gui.add_html("")
    with gui.add_folder("Focused env"):
      self.focus_n = gui.add_number("Env", initial_value=0, min=0, step=1)
      self.play_clip = gui.add_dropdown("Clip", [NONE])
      clipb = gui.add_button_group("Clip", ("Random", "Push", "Restart all"))
    self.ghost_mode = gui.add_dropdown(
      "Ghost", ("aligned", "world"), hint="aligned: re-anchored to the robot (xy + yaw)"
    )
    self.targets_cb = gui.add_checkbox("Keybody targets", initial_value=True)
    self.terms_html = gui.add_html("")

    @self.run_dd.on_update
    def _(_e) -> None:
      if not self._syncing:
        self._refresh_play_form(keep_run=True)

    @apply.on_click
    def _(e) -> None:
      if e.target.value == "Stop":
        self.c.play.stop()
        return
      ck = self.ckpt_dd.value
      self._guard(
        lambda: self.c.play.configure(
          self.run_dd.value,
          "" if ck in (ZERO, NONE) else ck,
          motion=self._libs.get(self.lib_dd.value, ""),
          num_envs=int(self.envs_dd.value),
          terminations=self.term_cb.value,
        )
      )

    @self.focus_n.on_update
    def _(_e) -> None:
      if not self._syncing:
        self.c.play.set_options(focus=int(self.focus_n.value))

    @self.play_clip.on_update
    def _(_e) -> None:
      if (
        not self._syncing
        and self.play_clip.value != NONE
        and self.c.play.session is not None
      ):
        self._guard(lambda: self.c.play.set_clip(self.play_clip.value))

    @clipb.on_click
    def _(e) -> None:
      v = e.target.value
      if v == "Random":
        self._guard(lambda: self.c.play.set_clip(""))
      elif v == "Push":
        self.c.play.push(speed=1.2)
      else:
        self.c.play.restart()

    @self.ghost_mode.on_update
    def _(_e) -> None:
      self.c.play.set_options(ghost_mode=self.ghost_mode.value)

    @self.targets_cb.on_update
    def _(_e) -> None:
      self.c.play.set_options(show_targets=self.targets_cb.value)

  def _refresh_play_form(self, keep_run: bool = False) -> None:
    runs = self.c.play.runs()
    labels = [r["label"] for r in runs] or [NONE]
    self._syncing = True
    try:
      if not keep_run:
        self.run_dd.options = labels
        if self.c.play.run_label in labels:
          self.run_dd.value = self.c.play.run_label
      run = next((r for r in runs if r["label"] == self.run_dd.value), None)
      self.ckpt_dd.options = (run["checkpoints"] if run else []) + [ZERO]
      libs = self.c.play.libraries(self.run_dd.value) if run else []
      self._libs = {x["label"]: x["value"] for x in libs}
      self.lib_dd.options = list(self._libs) or [NONE]
      cur = next((k for k, v in self._libs.items() if v == self.c.play.motion), None)
      sample = next((k for k, v in self._libs.items() if v.endswith("/samples")), None)
      if cur or sample:
        self.lib_dd.value = cur or sample
    finally:
      self._syncing = False

  def _refresh_play_state(self) -> None:
    s = self.c.play.state()
    if not s["active"]:
      self.play_html.content = _box(
        f"<i style='color:{MUTED}'>no session · pick a run (the bundled demo works out of the box) and Load</i>"
      )
      return
    self.play_html.content = _box(
      _kv(
        [
          ("task", f"<code>{html.escape(s['task'])}</code>"),
          ("checkpoint", html.escape(s["checkpoint"])),
          (
            "envs",
            f"{s['num_envs']} of {s['capacity']} built · {s['clips_loaded']} clips",
          ),
          (
            "focus",
            f"env {s['focus']} · {html.escape(s['focus_clip'])} ({'pinned' if s['focus_pinned'] else 'random'})",
          ),
        ]
      )
    )
    clips = self.c.play.clips()[:MAX_CLIPS] or [NONE]
    self._syncing = True
    try:
      if list(self.play_clip.options) != clips:
        self.play_clip.options = clips
      if s["focus_clip"] in clips:
        self.play_clip.value = s["focus_clip"]
      self.focus_n.value = s["focus"]
    finally:
      self._syncing = False

  # ------------------------------------------------------------------ viewer panel
  def _build_viewer(self, gui) -> None:
    self.title_html = gui.add_html("")
    tr = gui.add_button_group("Transport", ("⏮", "⏯", "⏭"))
    self.frame_slider = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    self.speed_bg = gui.add_button_group("Speed", SPEEDS)
    self.follow_cb = gui.add_checkbox("Follow focused robot", initial_value=True)
    self.ghost_cb = gui.add_checkbox("Ghost", initial_value=True)
    self.marks_cb = gui.add_checkbox(
      "Marks (targets · path · feet)", initial_value=True
    )
    self.hud_html = gui.add_html("")

    @tr.on_click
    def _(e) -> None:
      self._transport(e.target.value)

    @self.frame_slider.on_update
    def _(_e) -> None:
      if not self._syncing and self._hud.get("owner") == "data":
        if self._hud.get("grid"):
          self.c.data.set_time(self.frame_slider.value / 50.0)
        else:
          self.c.data.set_frame(int(self.frame_slider.value))

    @self.speed_bg.on_click
    def _(e) -> None:
      speed = float(e.target.value.rstrip("×"))
      self.c.data.set_options(speed=speed)
      self.c.play.set_options(speed=speed)

    @self.ghost_cb.on_update
    def _(_e) -> None:
      self._seen_pose = -1

    @self.marks_cb.on_update
    def _(_e) -> None:
      self._seen_overlay = -1

  def _transport(self, action: str) -> None:
    owner = self._hud.get("owner")
    if owner == "data":
      if action == "⏯":
        self.c.data.set_playing(not self._hud.get("playing"))
      elif action == "⏮":
        self.c.data.set_time(0.0)
      elif self._hud.get("grid"):
        self.c.data.set_time(self._hud.get("time", 0.0) + 0.02)
      else:
        self.c.data.set_frame(self._hud.get("frame", 0) + 1)
    elif owner == "play":
      if action == "⏯":
        self.c.play.set_playing(not self._hud.get("playing"))
      elif action == "⏮":
        self.c.play.restart()
      else:
        self.c.play.step_once()

  def _refresh_hud(self) -> None:
    hud = self._hud
    if not hud.get("owner"):
      self.title_html.content = _box(
        f"{_micro(self.c.owner)}<br/><b>{html.escape(self.c.robot_id)}</b> · default pose"
      )
      self.hud_html.content = ""
      return
    state = "live" if hud.get("playing") else "paused"
    color = "#36cfe0" if hud.get("playing") else "#e6a23c"
    self.title_html.content = _box(
      f"<span style='font-family:monospace;font-size:0.7em;letter-spacing:0.18em;text-transform:uppercase;"
      f"color:{color};border:1px solid {color};border-radius:99px;padding:1px 7px'>{state}</span> "
      f"<b>{html.escape(hud.get('title', ''))}</b><br/>"
      f"<span style='font-family:monospace;color:{MUTED}'>{hud.get('time', 0):.2f} / {hud.get('duration', 0):.2f} s · "
      f"frame {hud.get('frame', 0)} / {max(0, hud.get('frames', 1) - 1)}</span>"
    )
    self._syncing = True
    try:
      self.frame_slider.max = max(1, hud.get("frames", 1) - 1)
      self.frame_slider.value = min(hud.get("frame", 0), self.frame_slider.max)
      self.frame_slider.disabled = hud.get("owner") != "data"
    finally:
      self._syncing = False
    if hud.get("owner") == "play":
      m = hud.get("metrics", {})

      def f(v, k=1.0, d=1):
        return NONE if v is None else f"{k * v:.{d}f}"

      self.hud_html.content = _box(
        _kv(
          [
            ("return", f"{f(hud.get('return'))} (mean {f(hud.get('mean_return'))})"),
            ("anchor drift", f"{f(m.get('error_anchor_pos'), 100)} cm"),
            ("body err", f"{f(m.get('error_body_pos'), 100)} cm"),
            ("joint err", f"{f(m.get('error_joint_pos'), 1, 2)} rad"),
            ("env", f"{hud.get('focus', 0)} of {hud.get('num_envs', 1)}"),
          ]
        )
      )
      self.terms_html.content = _bars(hud.get("terms", {}))
    else:
      self.hud_html.content = ""

  # ================================================================== events
  def _on_event(self, topic: str, payload: dict) -> None:
    with self._plock:
      self._pending.add(topic)

  def _process_events(self) -> None:
    with self._plock:
      topics, self._pending = self._pending, set()
    if not topics:
      return
    if "log" in topics:
      lines = list(self.c.logs)[-60:]
      colors = {"ok": "#4fc78a", "warn": "#e6a23c", "error": "#ef6a6a"}
      self.log_html.content = _box(
        "".join(
          f"<div style='font-family:monospace;font-size:0.85em;color:{colors.get(e['level'], TEXT)}'>"
          f"<span style='color:{MUTED}'>{time.strftime('%H:%M:%S', time.localtime(e['t']))}</span> "
          f"{html.escape(e['text'])}</div>"
          for e in reversed(lines)
        )
      )
    if "busy" in topics:
      self.task_bar.visible = self.c.busy
      self._refresh_header()
    if "robot" in topics:
      self._refresh_all()
      return
    if "data" in topics:
      if not self.c.data.state()["active"]:
        self._refresh_datasets()
      self._refresh_qa()
    if "jobs" in topics:
      self._refresh_jobs()
    if "play" in topics or "focus" in topics:
      self._refresh_play_state()

  def _refresh_header(self) -> None:
    task = self.c.task or "idle"
    self.header.content = (
      f"<div style='display:flex;gap:8px;align-items:center;font-family:monospace;font-size:0.8em;"
      f"padding:2px 4px 6px'><span style='color:#36cfe0'>●</span><b style='color:#36cfe0'>wbc</b>·mjlab "
      f"<span style='color:{MUTED}'>studio</span><span style='margin-left:auto;color:{MUTED}'>"
      f"robot <b style='color:{TEXT}'>{html.escape(self.c.robot_id)}</b> · "
      f"<span style='color:{'#e6a23c' if self.c.busy else MUTED}'>{html.escape(task)}</span></span></div>"
    )

  def _refresh_all(self) -> None:
    self._syncing = True
    try:
      names = self.c.robot_names() or [NONE]
      self.robot_dd.options = names
      if self.c.robot_id in names:
        self.robot_dd.value = self.c.robot_id
    finally:
      self._syncing = False
    self._refresh_header()
    self._refresh_robot()
    self._refresh_datasets()
    self._refresh_train_form()
    self._refresh_jobs()
    self._refresh_play_form()
    self._refresh_play_state()

  def _guard(self, fn) -> None:
    try:
      fn()
    except Exception as exc:  # noqa: BLE001
      self.c.log(str(exc), "error")

  # ================================================================== scene
  def _sync_scene(self) -> None:
    sc = self.c.scene
    if sc.model_version != self._model_version and self.c.model is not None:
      for r in (self._robot, self._ghost):
        if r is not None:
          r.remove()
      self._robot = BatchedRobot(self.server, self.c.model, "/robots")
      self._ghost = BatchedRobot(
        self.server, self.c.model, "/ghosts", tint=ACCENT, opacity=0.3
      )
      self._model_version = sc.model_version
      self._seen_pose = -1
    if sc.pose_version != self._seen_pose and self._robot is not None:
      self._seen_pose = sc.pose_version
      snap = sc.snapshot()
      pos, quat = snap["pos"], snap["quat"]
      if pos.ndim == 3 and pos.shape[0]:
        self._robot.update(pos, quat)
        gp, gq = snap["ghost_pos"], snap["ghost_quat"]
        if (
          self.ghost_cb.value
          and gp.ndim == 3
          and gp.shape[0]
          and gp.shape[1] == pos.shape[1]
        ):
          self._ghost.update(gp, gq)
        else:
          self._ghost.update(np.zeros((0, 0, 3)), np.zeros((0, 0, 4)))
        focus = min(snap["focus"], pos.shape[0] - 1)
        self._follow(pos[focus, 1] if pos.shape[1] > 1 else pos[focus, 0])
      self._hud = snap["hud"]
    if sc.overlay_version != self._seen_overlay:
      self._seen_overlay = sc.overlay_version
      self._sync_overlays(sc.snapshot()["overlays"] if self.marks_cb.value else {})

  def _sync_overlays(self, overlays: dict) -> None:
    sc = self.server.scene
    for key in [k for k in self._overlay_handles if k not in overlays]:
      self._overlay_handles.pop(key)[1].remove()
    for key, ov in overlays.items():
      name = f"/overlay/{key}"
      color = np.asarray(ov.color, np.uint8)
      sig = (ov.kind, ov.data.shape, color.shape, ov.size)
      old = self._overlay_handles.get(key)
      if old is not None and old[0] == sig and ov.kind == "points" and color.ndim == 1:
        old[1].points = ov.data.astype(np.float32)
        continue
      if old is not None:
        old[1].remove()
      if ov.kind == "points":
        h = sc.add_point_cloud(
          name,
          ov.data.astype(np.float32),
          color if color.ndim == 2 else tuple(int(v) for v in color),
          point_size=ov.size,
          point_shape="circle",
        )
      elif ov.kind == "segments":
        h = sc.add_line_segments(
          name,
          ov.data.astype(np.float32),
          color if color.ndim == 2 else tuple(int(v) for v in color),
          line_width=3.0,
        )
      else:
        h = sc.add_mesh_simple(
          name,
          ov.data,
          ov.faces,
          color=tuple(int(v) for v in color),
          opacity=ov.opacity,
        )
      self._overlay_handles[key] = (sig, h)

  def _follow(self, target: np.ndarray) -> None:
    """Move each client's orbit target to the focused robot, keeping the user's view offset."""
    if not self.follow_cb.value:
      return
    target = np.asarray(target, np.float64).copy()
    target[2] = 0.8
    if (
      self._follow_target is not None
      and np.linalg.norm(target - self._follow_target) < 2e-3
    ):
      return
    sent = self._follow_target
    self._follow_target = target
    for cid, client in self.server.get_clients().items():
      cam = client.camera
      pos, look = (
        np.asarray(cam.position, np.float64),
        np.asarray(cam.look_at, np.float64),
      )
      offset = self._cam_offsets.get(cid)
      if offset is None or (sent is not None and np.linalg.norm(look - sent) < 0.05):
        offset = pos - look
      if not np.all(np.isfinite(offset)) or np.linalg.norm(offset) < 0.3:
        offset = np.array([2.8, -3.0, 1.0])
      self._cam_offsets[cid] = offset
      with client.atomic():
        cam.look_at = target
        cam.position = target + offset

  # ================================================================== loop
  def run(self, open_browser: bool = True) -> None:
    @self.server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
      client.camera.position = (2.8, -3.0, 1.8)
      client.camera.look_at = (0.0, 0.0, 0.8)

    print(f"\n  WBC Studio (viser) → {self.url}\n")
    if open_browser:
      try:
        webbrowser.open(self.url)
      except Exception:  # noqa: BLE001
        pass
    last_ui = 0.0
    try:
      while True:
        try:
          self._sync_scene()
        except Exception as exc:  # noqa: BLE001
          print(f"[viser] scene sync failed: {exc}")
          time.sleep(0.5)
        now = time.time()
        if now - last_ui > 0.1:
          last_ui = now
          try:
            self._process_events()
            self._refresh_hud()
          except Exception as exc:  # noqa: BLE001
            print(f"[viser] ui refresh failed: {exc}")
        time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
      print("\nStopped.")
      self.server.stop()


def _titlebar():
  """The WBC-MJLab logo (as in the docs and the web Studio), linking to the docs."""
  import base64
  from pathlib import Path

  from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

  logo = Path(__file__).parent / "web" / "static" / "brand" / "logo-64.png"
  url = "data:image/png;base64," + base64.b64encode(logo.read_bytes()).decode()
  docs = "https://wbc-mjlab.github.io/wbc-mjlab/"
  return TitlebarConfig(
    buttons=(TitlebarButton(text="Docs", icon="Description", href=docs),),
    image=TitlebarImage(
      image_url_light=url, image_url_dark=url, image_alt="WBC-MJLab", href=docs
    ),
  )


__all__ = ["BatchedRobot", "ViserStudio"]
