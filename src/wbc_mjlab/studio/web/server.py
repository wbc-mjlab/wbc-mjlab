"""Web frontend of the Studio: a FastAPI app over :class:`StudioCore`.

* ``/api/...`` JSON endpoints, one group per service (robot, data, train / jobs, play) — thin
  wrappers, all logic is in the services.
* ``/api/robot/model`` + ``/api/robot/model.bin``: the robot's visual geoms and meshes, fetched
  once per robot; the browser instances them for every robot copy and every ghost.
* ``/ws``: core events as JSON text (``{"type": "event", "topic", "payload"}``) and scene
  frames as binary messages (see :func:`encode_frame`) at up to 60 Hz.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import math
import struct
from contextlib import asynccontextmanager
from pathlib import Path

import mujoco
import numpy as np
from fastapi import (
  Body,
  FastAPI,
  HTTPException,
  Request,
  WebSocket,
  WebSocketDisconnect,
)
from fastapi.responses import (
  FileResponse,
  HTMLResponse,
  PlainTextResponse,
  RedirectResponse,
  Response,
)
from fastapi.staticfiles import StaticFiles

from wbc_mjlab.studio.core import StudioCore
from wbc_mjlab.studio.plots import encode_rows

STATIC = Path(__file__).parent / "static"
FRAME_HZ = 60.0  # poll rate; a frame is only sent when the scene changed
VISIBLE_GROUPS = 3  # geom groups 0..2 are drawn (MuJoCo viewer default)


# ---------------------------------------------------------------------- robot model
def model_payload(model: mujoco.MjModel) -> tuple[dict, bytes]:
  """Visual geoms of the Studio robot model (JSON) + their mesh buffers (binary)."""
  geoms, meshes, blobs, offset = [], {}, [], 0
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
    gtype = mujoco.mjtGeom(int(model.geom_type[g])).name.removeprefix("mjGEOM_").lower()
    mesh = int(model.geom_dataid[g]) if gtype == "mesh" else -1
    if mesh >= 0 and mesh not in meshes:
      va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
      fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
      verts = np.ascontiguousarray(model.mesh_vert[va : va + vn], np.float32)
      faces = np.ascontiguousarray(model.mesh_face[fa : fa + fn], np.uint32)
      meshes[mesh] = {
        "vertices": [offset, vn * 3],
        "faces": [offset + verts.nbytes, fn * 3],
      }
      blobs += [verts.tobytes(), faces.tobytes()]
      offset += verts.nbytes + faces.nbytes
    geoms.append(
      {
        "body": int(model.geom_bodyid[g]),
        "type": gtype,
        "size": model.geom_size[g].tolist(),
        "pos": model.geom_pos[g].tolist(),
        "quat": model.geom_quat[g].tolist(),
        "rgba": [float(x) for x in rgba],
        "mesh": mesh,
      }
    )
  names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    for b in range(model.nbody)
  ]
  meta = {
    "nbody": int(model.nbody),
    "bodies": names,
    "geoms": geoms,
    "meshes": {str(k): v for k, v in meshes.items()},
    "bytes": offset,
  }
  return meta, b"".join(blobs)


# ---------------------------------------------------------------------- scene frames
def encode_frame(core: StudioCore, with_overlays: bool) -> bytes:
  """``[u32 header length][header JSON][pad to 4][float32 blob]``.

  The blob holds the instance body poses ``(n, nbody, 7)`` (pos + wxyz), then the ghost poses
  ``(ng, nbody, 7)`` and, when ``with_overlays``, every overlay's data (and per-item colours)
  at the offsets listed in the header (in floats)."""
  sc = core.scene
  snap = sc.snapshot()
  pos, quat = snap["pos"], snap["quat"]
  n = int(pos.shape[0])
  nb = int(pos.shape[1]) if pos.ndim == 3 and n else 0
  ng = int(snap["ghost_pos"].shape[0]) if nb and snap["ghost_pos"].ndim == 3 else 0
  if ng and snap["ghost_pos"].shape[1] != nb:
    ng = 0
  parts = []
  if n:
    parts.append(np.concatenate([pos, quat], axis=-1).astype(np.float32).ravel())
  if ng:
    parts.append(
      np.concatenate([snap["ghost_pos"], snap["ghost_quat"]], axis=-1)
      .astype(np.float32)
      .ravel()
    )
  cursor = sum(int(p.size) for p in parts)
  header = {
    "kind": "frame",
    "n": n,
    "ng": ng,
    "nbody": nb,
    "focus": snap["focus"],
    "labels": snap["labels"],
    "hud": snap["hud"],
    "owner": core.owner,
    "model_version": sc.model_version,
    "terrain_version": sc.terrain_version,
    "overlays": None,
  }
  if with_overlays:
    items = []
    for key, ov in snap["overlays"].items():
      data = np.asarray(ov.data, np.float32).ravel()
      color = np.asarray(ov.color)
      item = {
        "key": key,
        "kind": ov.kind,
        "size": float(ov.size),
        "opacity": float(ov.opacity),
        "data": [cursor, int(data.size)],
        "on_top": bool(ov.on_top),
      }
      parts.append(data)
      cursor += int(data.size)
      if color.ndim == 1:
        item["color"] = [int(c) for c in color[:3]]
      else:
        cols = (color.reshape(-1, 3).astype(np.float32) / 255.0).ravel()
        item["colors"] = [cursor, int(cols.size)]
        parts.append(cols)
        cursor += int(cols.size)
      if ov.faces is not None:
        faces = np.asarray(ov.faces, np.float32).ravel()  # exact for indices < 2^24
        item["faces"] = [cursor, int(faces.size)]
        parts.append(faces)
        cursor += int(faces.size)
      items.append(item)
    header["overlays"] = items
  head = json.dumps(
    _finite(header), separators=(",", ":"), default=_json_default
  ).encode()
  pad = (-(4 + len(head))) % 4
  blob = np.concatenate(parts).astype(np.float32).tobytes() if parts else b""
  return struct.pack("<I", len(head) + pad) + head + b" " * pad + blob


def _finite(o):
  """NaN / inf → None (``json.dumps`` writes bare ``NaN``, which browsers reject)."""
  if isinstance(o, float):
    return o if math.isfinite(o) else None
  if isinstance(o, (np.floating, np.integer)):
    return _finite(o.item())
  if isinstance(o, dict):
    return {k: _finite(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)):
    return [_finite(v) for v in o]
  return o


def _json_default(o):
  if isinstance(o, (np.floating, np.integer)):
    return o.item()
  if isinstance(o, np.ndarray):
    return o.tolist()
  if isinstance(o, Path):
    return str(o)
  return str(o)


# ---------------------------------------------------------------------- app
COOKIE = "wbc_studio_token"
LOOPBACK = ("127.0.0.1", "::1", "localhost")
SIGN_IN = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>WBC-MJLab Studio</title>
<link rel="icon" href="/static/brand/logo-64.png" type="image/png">
<style>
:root{--bg:#0d1620;--panel:#16232f;--line:#233444;--text:#e3edf6;--muted:#8fa0b3;--accent:#36cfe0;
--on-accent:#0d1620;--bad:#ef6b6b}
@media (prefers-color-scheme:light){:root{--bg:#eef4f8;--panel:#fff;--line:#c5d4e0;--text:#0d1620;
--muted:#4a6278;--accent:#0e8fa3;--on-accent:#fff}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:16px;
background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
form{width:100%;max-width:360px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:28px}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:14px}
.brand img{width:40px;height:40px;border-radius:9px}
h1{margin:0;font-size:19px;font-weight:650}h1 span{font-weight:500;color:var(--muted)}
p{margin:0 0 20px;color:var(--muted);font-size:14px}
input{width:100%;padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);
color:var(--text);font:inherit;outline:none}input:focus{border-color:var(--accent)}
button{margin-top:12px;width:100%;padding:10px;border:0;border-radius:8px;background:var(--accent);
color:var(--on-accent);font:inherit;font-weight:600;cursor:pointer}.err{color:var(--bad);margin:10px 0 0;font-size:13px}
code{font-size:13px}
</style></head><body><form method="get" action="/">
<div class="brand"><img src="/static/brand/logo-64.png" alt=""><h1>WBC-MJLab <span>Studio</span></h1></div>
<p>Enter the access token. It is in the link printed where the Studio was started
(<code>?token=…</code>).</p>
<input name="token" type="password" placeholder="Token" autocomplete="current-password" autofocus required>
__ERROR__<button type="submit">Open Studio</button></form></body></html>"""


def create_app(core: StudioCore, token: str | None = None) -> FastAPI:
  """``token``: required from non-loopback clients (``?token=`` once, then a cookie)."""
  clients: set[WebSocket] = set()
  loop_holder: dict = {}
  model_cache: dict = {}

  @asynccontextmanager
  async def lifespan(_app: FastAPI):
    loop_holder["loop"] = asyncio.get_running_loop()
    pump = asyncio.create_task(frame_pump())
    yield
    pump.cancel()

  app = FastAPI(title="wbc-mjlab studio", lifespan=lifespan)
  app.mount("/static", StaticFiles(directory=STATIC), name="static")

  def authorized(conn: Request | WebSocket) -> bool:
    if token is None or (conn.client is not None and conn.client.host in LOOPBACK):
      return True
    given = conn.cookies.get(COOKIE) or conn.query_params.get("token") or ""
    return hmac.compare_digest(given, token)

  @app.middleware("http")
  async def _auth(request: Request, call_next):
    public = request.url.path.startswith("/static/brand/") or request.url.path in (
      "/favicon.ico",
      "/favicon.png",
    )  # the logo, shown on the sign-in page
    if not public and not authorized(request):
      if request.method == "GET" and request.url.path == "/":
        wrong = '<p class="err">That token did not match.</p>'
        page = SIGN_IN.replace(
          "__ERROR__", wrong if "token" in request.query_params else ""
        )
        return HTMLResponse(page, 401)
      return PlainTextResponse(
        "WBC Studio: open the link with ?token=… printed where the Studio was started.",
        401,
      )
    given = request.query_params.get("token")
    if (
      token is not None
      and given
      and request.method == "GET"
      and request.url.path == "/"
    ):
      # store the token and drop it from the address bar
      response = RedirectResponse("/", status_code=303)
      response.set_cookie(
        COOKIE, given, httponly=True, samesite="strict", max_age=30 * 86400
      )
      return response
    return await call_next(request)

  @app.middleware("http")
  async def _revalidate_static(request: Request, call_next):
    # ES modules are loaded by name: without this a browser may keep an old module next to a
    # new one after an update (heuristic caching). ETags keep the revalidation cheap (304).
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
      response.headers["Cache-Control"] = "no-cache"
    return response

  def ok(**extra):
    return {"ok": True, **extra}

  def task(started: bool):
    if not started:
      raise HTTPException(409, f"busy with {core.task!r}")
    return ok(task=core.task)

  def guard(fn, *args, **kwargs):
    try:
      return fn(*args, **kwargs)
    except (ValueError, RuntimeError, KeyError, FileNotFoundError, OSError) as exc:
      raise HTTPException(400, str(exc)) from exc

  # -------------------------------------------------------------- events → websockets
  def on_event(topic: str, payload: dict) -> None:
    loop = loop_holder.get("loop")
    if loop is None or not clients:
      return
    msg = json.dumps(
      _finite({"type": "event", "topic": topic, "payload": payload}),
      default=_json_default,
    )
    for ws in list(clients):
      asyncio.run_coroutine_threadsafe(_send_text(ws, msg), loop)

  async def _send_text(ws: WebSocket, msg: str) -> None:
    try:
      await ws.send_text(msg)
    except Exception:  # noqa: BLE001
      clients.discard(ws)

  core.on("*", on_event)

  async def frame_pump() -> None:
    seen: dict[WebSocket, tuple[int, int, int]] = {}
    loop = asyncio.get_running_loop()
    deadline = loop.time()
    while True:
      deadline = max(deadline + 1.0 / FRAME_HZ, loop.time() - 0.05)
      await asyncio.sleep(max(0.0, deadline - loop.time()))
      if not clients:
        continue
      sc = core.scene
      state = (
        sc.pose_version,
        sc.overlay_version,
        sc.model_version,
        sc.terrain_version,
      )
      for ws in list(clients):
        last = seen.get(ws)
        if last == state:
          continue
        with_overlays = last is None or last[1] != state[1] or last[2] != state[2]
        try:
          await ws.send_bytes(encode_frame(core, with_overlays))
          seen[ws] = state
        except Exception:  # noqa: BLE001
          clients.discard(ws)
          seen.pop(ws, None)

  @app.websocket("/ws")
  async def ws_endpoint(ws: WebSocket) -> None:
    if not authorized(ws):
      await ws.close(code=1008)
      return
    await ws.accept()
    clients.add(ws)
    try:
      while True:
        await ws.receive_text()  # keep-alive; the client talks over HTTP
    except WebSocketDisconnect:
      pass
    finally:
      clients.discard(ws)

  # -------------------------------------------------------------- pages / state
  @app.get("/")
  def index():
    return FileResponse(STATIC / "index.html")

  @app.get("/favicon.ico")
  @app.get("/favicon.png")
  def favicon():
    return FileResponse(STATIC / "brand" / "logo-64.png", media_type="image/png")

  @app.get("/api/state")
  def state():
    import torch

    gpu = ""
    if torch.cuda.is_available() and core.device.startswith("cuda"):
      gpu = torch.cuda.get_device_name(torch.device(core.device))
    return {
      "robot": core.robot_id,
      "robots": core.robot_names(),
      "owner": core.owner,
      "task": core.task,
      "logs": list(core.logs)[-200:],
      "device": core.device,
      "gpu": gpu,
    }

  # -------------------------------------------------------------- robots
  @app.get("/api/robot")
  def robot_info():
    return core.robots.info()

  @app.get("/api/robots")
  def robots_catalogue():
    return core.robots.catalogue()

  @app.get("/api/robot/model")
  def robot_model():
    if core.model is None:
      raise HTTPException(404, "no robot")
    key = (core.robot_id, core.scene.model_version)
    if model_cache.get("key") != key:
      meta, blob = model_payload(core.model)
      model_cache.update(
        key=key, meta=meta | {"robot": core.robot_id, "version": key[1]}, blob=blob
      )
    return model_cache["meta"]

  @app.get("/api/scene/terrain.bin")
  def scene_terrain():
    """``[u32 nv][u32 nf][f32 verts nv·3][u8 colours nv·3, padded to 4][u32 faces nf·3]``
    (empty: nv = nf = 0)."""
    mesh = core.scene.terrain
    if not mesh:
      return Response(struct.pack("<II", 0, 0), media_type="application/octet-stream")
    v, c, f = mesh["verts"], mesh["colors"], mesh["faces"]
    cols = c.astype(np.uint8).tobytes()
    cols += b"\0" * (-len(cols) % 4)
    blob = (
      struct.pack("<II", len(v), len(f))
      + v.astype("<f4").tobytes()
      + cols
      + f.astype("<u4").tobytes()
    )
    return Response(blob, media_type="application/octet-stream")

  @app.get("/api/robot/model.bin")
  def robot_model_bin():
    robot_model()
    return Response(model_cache["blob"], media_type="application/octet-stream")

  @app.post("/api/robot/select")
  def robot_select(name: str = Body(embed=True)):
    if name == core.robot_id:
      core.robots.show_pose()
      return ok()
    return task(core.robots.select(name))

  @app.post("/api/robot/joint")
  def robot_joint(name: str = Body(), value: float = Body()):
    guard(core.robots.set_joint, name, value)
    return ok()

  @app.post("/api/robot/reset")
  def robot_reset(zero: bool = Body(False, embed=True)):
    core.robots.reset_pose(zero)
    return ok()

  @app.post("/api/robot/show")
  def robot_show():
    core.robots.show_pose()
    return ok()

  @app.post("/api/robot/options")
  def robot_options(show_wiring: bool = Body(embed=True)):
    core.robots.set_options(show_wiring=show_wiring)
    return ok()

  # -------------------------------------------------------------- add a robot
  @app.get("/api/robots/import")
  def import_state():
    from wbc_mjlab.robots import custom

    return {
      "dir": str(custom.robots_dir()),
      "custom": [s.id for s in custom.list_specs()],
      "drafts": core.importer.drafts(),
    }

  @app.put("/api/robots/import/upload/{robot_id}/{path:path}")
  async def import_upload(robot_id: str, path: str, request: Request):
    content = await request.body()
    return ok(path=str(guard(core.importer.upload, robot_id, path, content)))

  @app.post("/api/robots/import/path")
  def import_path(id: str = Body(), source: str = Body()):  # noqa: A002
    return ok(files=guard(core.importer.import_path, id, source))

  @app.get("/api/robots/import/files")
  def import_files(id: str):  # noqa: A002
    return guard(core.importer.files, id)

  @app.post("/api/robots/import/inspect")
  def import_inspect(id: str = Body(), model: str = Body()):  # noqa: A002
    return guard(core.importer.inspect, id, model)

  @app.post("/api/robots/import/preview")
  def import_preview(wiring: dict = Body(embed=True)):
    guard(core.importer.preview, wiring)
    return ok()

  @app.post("/api/robots/import/check")
  def import_check(spec: dict = Body(embed=True)):
    return guard(core.importer.check, spec)

  @app.post("/api/robots/import/save")
  def import_save(spec: dict = Body(embed=True)):
    return ok(**guard(core.importer.save, spec))

  @app.post("/api/robots/import/cancel")
  def import_cancel():
    core.importer.cancel()
    return ok()

  @app.get("/api/robots/{robot_id}/spec")
  def robot_spec(robot_id: str):
    from wbc_mjlab.robots import custom

    return guard(lambda: custom.load_spec(robot_id).to_dict())

  @app.delete("/api/robots/{robot_id}")
  def robot_delete(robot_id: str):
    from wbc_mjlab.robots import custom

    if robot_id not in {s.id for s in custom.list_specs()} and robot_id not in {
      d["id"] for d in core.importer.drafts()
    }:
      raise HTTPException(400, f"{robot_id} isn't a robot added in the Studio")
    guard(core.importer.delete, robot_id)
    return ok()

  # -------------------------------------------------------------- data
  @app.get("/api/data/datasets")
  def data_datasets():
    return {
      "robot": core.robot_id,
      "selected": core.data.dataset,
      "datasets": core.data.datasets(),
    }

  @app.get("/api/data/clips")
  def data_clips(
    dataset: str = "", q: str = "", status: str = "all", page: int = 0, size: int = 100
  ):
    return guard(core.data.listing, dataset, q, status, page, size)

  @app.get("/api/data/preview")
  def data_preview():
    return core.data.state()

  @app.post("/api/data/load")
  def data_load(dataset: str = Body(), clip: str = Body(), source: str = Body("auto")):
    guard(core.data.load, dataset, clip, source)
    return core.data.state()

  @app.post("/api/data/grid")
  def data_grid(
    dataset: str = Body(),
    q: str = Body(""),
    status: str = Body("all"),
    count: int = Body(16),
  ):
    return task(guard(core.data.load_grid, dataset, q, status, count))

  @app.post("/api/data/close")
  def data_close():
    core.data.close()
    return ok()

  @app.post("/api/data/frame")
  def data_frame(frame: int = Body(embed=True)):
    core.data.set_frame(frame)
    return ok()

  @app.post("/api/data/time")
  def data_time(t: float = Body(embed=True)):
    core.data.set_time(t)
    return ok()

  @app.post("/api/data/playing")
  def data_playing(playing: bool = Body(embed=True)):
    core.data.set_playing(playing)
    return ok()

  @app.post("/api/data/options")
  def data_options(opts: dict = Body()):
    core.data.set_options(**opts)
    return ok()

  @app.get("/api/data/series")
  def data_series(keys: str = "root_z,speed,feet"):
    return core.data.clip_series([k for k in keys.split(",") if k])

  @app.get("/api/plots/clip")
  def plots_clip():
    """Plot channels of the clip open in Data (Plots window)."""
    try:
      return core.data.plot_data()
    except RuntimeError as exc:
      raise HTTPException(409, str(exc)) from exc

  @app.get("/api/plots/play")
  def plots_play(since: int = 0, epoch: int | None = None, env: int | None = None):
    """A Play env's record (default: the focused one) since row ``since`` (all of it for
    another ``epoch``), binary: see :func:`wbc_mjlab.studio.plots.encode_rows`."""
    try:
      out = core.play.plot_data(since, epoch, env)
    except RuntimeError as exc:
      raise HTTPException(409, str(exc)) from exc
    return Response(encode_rows(out), media_type="application/octet-stream")

  @app.get("/api/data/joints")
  def data_joints():
    return core.data.joint_names()

  @app.post("/api/data/scan")
  def data_scan(dataset: str = Body(embed=True)):
    return task(core.data.scan(dataset))

  @app.post("/api/data/convert")
  def data_convert(
    dataset: str = Body(),
    which: str = Body("missing"),
    clips: list[str] = Body(default_factory=list),
    batch_size: int = Body(8),
    output_fps: float = Body(50.0),
    mirror: bool = Body(False),
    debias_z: bool = Body(False),
    input_fps: float = Body(0.0),
  ):
    job = guard(
      core.data.convert,
      dataset,
      which,
      clips,
      batch_size=batch_size,
      output_fps=output_fps,
      mirror=mirror,
      debias_z=debias_z,
      input_fps=input_fps or None,
    )
    return ok(job=job.name)

  @app.post("/api/data/new")
  def data_new(name: str = Body(embed=True)):
    return ok(path=str(guard(core.data.new_dataset, name)))

  @app.post("/api/data/import")
  def data_import(source: str = Body(), name: str = Body(), link: bool = Body(False)):
    return task(guard(core.data.import_folder, source, name, link))

  @app.put("/api/data/upload/{dataset}/{filename}")
  async def data_upload(dataset: str, filename: str, request: Request):
    content = await request.body()
    path = guard(core.data.upload, dataset, filename, content)
    return ok(path=str(path))

  @app.post("/api/data/uploaded")
  def data_uploaded(dataset: str = Body(embed=True)):
    core.emit("data", dataset=dataset)
    return ok()

  # -------------------------------------------------------------- training
  @app.get("/api/tasks")
  def tasks():
    return core.training.tasks()

  @app.get("/api/runs")
  def runs(robot: str = ""):
    return core.training.runs(robot or core.robot_id)

  @app.get("/api/jobs")
  def jobs(kind: str = ""):
    return core.training.jobs(kind or None)

  @app.get("/api/jobs/{name}")
  def job_detail(name: str, tail: int = 60):
    return guard(core.training.job_detail, name, tail)

  @app.post("/api/jobs/{name}/stop")
  def job_stop(name: str):
    guard(core.training.stop, name)
    return ok()

  @app.delete("/api/jobs/{name}")
  def job_delete(name: str):
    guard(core.training.delete, name)
    return ok()

  @app.post("/api/train/launch")
  def train_launch(
    task: str = Body(),
    dataset: str = Body(),
    num_envs: int = Body(4096),
    max_iterations: int = Body(30000),
    run_name: str = Body(""),
    resume_run: str = Body(""),
    resume_checkpoint: str = Body(""),
    logger: str = Body("tensorboard"),
    extra: str = Body(""),
  ):
    job = guard(
      core.training.launch,
      task,
      dataset,
      num_envs=num_envs,
      max_iterations=max_iterations,
      run_name=run_name,
      resume_run=resume_run,
      resume_checkpoint=resume_checkpoint,
      logger=logger,
      extra=extra,
    )
    return ok(job=job.name)

  # -------------------------------------------------------------- presets
  schema_cache: dict = {}

  def schema(base: str, modules: tuple[str, ...]) -> dict:
    """Editable knobs of ``base`` + ``modules`` (the defaults a preset's own changes diff
    against), plus the robot's observation term library."""
    from wbc_mjlab.studio import presets
    from wbc_mjlab.tasks import get_task_config, register_all_wbc_tasks

    register_all_wbc_tasks()
    robot = get_task_config(base).robot_id
    resolved = presets.module_overrides(list(modules), robot)
    key = (base, json.dumps(resolved, sort_keys=True))
    if key not in schema_cache:
      env_cfg, agent_cfg = presets.build(base, {}, modules=resolved)
      out = presets.describe(env_cfg, agent_cfg)
      obs = out["observations"]
      obs["library"] = presets.library(
        robot, obs["joints"], len(obs["bodies"]), env_cfg
      )
      schema_cache[key] = out
    return schema_cache[key]

  def presets_changed(name: str) -> None:
    core.emit("presets", name=name)

  @app.get("/api/presets")
  def presets_list():
    from wbc_mjlab.studio import presets

    return {
      "tasks": core.training.tasks(),
      "custom": presets.list_presets(core.robot_id),
      "modules": presets.list_modules(core.robot_id),
      "config_dir": str(presets.config_dir()),
    }

  @app.get("/api/presets/schema")
  def presets_schema(base: str, modules: str = ""):
    from wbc_mjlab.studio import presets

    split = presets.split_task_id(base)
    if split:
      base = split[0]
    return guard(schema, base, tuple(m for m in modules.split(",") if m))

  @app.get("/api/presets/{name}")
  def presets_get(name: str):
    from wbc_mjlab.studio import presets

    return guard(presets.load, name)

  @app.put("/api/presets/{name}")
  def presets_put(
    name: str,
    base: str = Body(),
    overrides: dict = Body(),
    description: str = Body(""),
    modules: list[str] = Body([]),
  ):
    from wbc_mjlab.studio import presets

    out = guard(presets.save, name, base, overrides, description, modules)
    mods = f" + {', '.join(modules)}" if modules else ""
    core.log(
      f"preset {name} saved ({base}{mods}, {presets.count_changes(overrides)} change(s))",
      "ok",
    )
    presets_changed(name)
    return out

  @app.delete("/api/presets/{name}")
  def presets_delete(name: str):
    from wbc_mjlab.studio import presets

    guard(presets.delete, name)
    presets_changed(name)
    return ok()

  @app.get("/api/modules/{name}")
  def modules_get(name: str):
    from wbc_mjlab.studio import presets

    return guard(presets.load_module, name, core.robot_id)

  @app.put("/api/modules/{name}")
  def modules_put(name: str, overrides: dict = Body(), description: str = Body("")):
    from wbc_mjlab.studio import presets

    out = guard(presets.save_module, name, overrides, description, core.robot_id)
    core.log(
      f"module {name} saved ({presets.count_changes(overrides)} change(s))", "ok"
    )
    schema_cache.clear()  # presets using it diff against new defaults
    presets_changed(name)
    return out

  @app.delete("/api/modules/{name}")
  def modules_delete(name: str):
    from wbc_mjlab.studio import presets

    guard(presets.delete_module, name)
    schema_cache.clear()
    presets_changed(name)
    return ok()

  @app.post("/api/play/rewards")
  def play_rewards(rewards: dict = Body(embed=True)):
    return ok(changed=guard(core.play.apply_rewards, rewards))

  @app.get("/api/curves")
  def curves(job: str = "", run: str = "", group: str = "train"):
    return core.training.curves(job=job, run=run, group=group)

  @app.get("/tensorboard")
  def tensorboard(request: Request, run: str = ""):
    try:
      port = core.tensorboard.start()
    except Exception as exc:  # noqa: BLE001
      raise HTTPException(500, f"tensorboard: {exc}") from exc
    host = request.url.hostname or "127.0.0.1"
    path = core.tensorboard.run_path(run) if run else "/"
    return RedirectResponse(f"http://{host}:{port}{path}")

  # -------------------------------------------------------------- play
  @app.get("/api/play")
  def play_state():
    return core.play.state()

  @app.get("/api/play/runs")
  def play_runs():
    return core.play.runs()

  @app.get("/api/play/libraries")
  def play_libraries(run: str = ""):
    return guard(core.play.libraries, run)

  @app.get("/api/play/clips")
  def play_clips():
    return core.play.clips()

  @app.post("/api/play/configure")
  def play_configure(
    run: str = Body(),
    checkpoint: str = Body(""),
    motion: str = Body(""),
    num_envs: int = Body(1),
    terminations: bool = Body(False),
  ):
    result = guard(
      core.play.configure,
      run,
      checkpoint,
      motion=motion,
      num_envs=num_envs,
      terminations=terminations,
    )
    if result == "busy":
      raise HTTPException(409, f"busy with {core.task!r}")
    return ok(result=result)

  @app.post("/api/play/stop")
  def play_stop():
    core.play.stop()
    return ok()

  @app.post("/api/play/playing")
  def play_playing(playing: bool = Body(embed=True)):
    core.play.set_playing(playing)
    return ok()

  @app.post("/api/play/step")
  def play_step():
    core.play.step_once()
    return ok()

  @app.post("/api/play/restart")
  def play_restart():
    core.play.restart()
    return ok()

  @app.post("/api/play/clip")
  def play_clip(clip: str = Body(""), env: int | None = Body(None)):
    guard(core.play.set_clip, clip, env)
    return ok()

  @app.post("/api/play/options")
  def play_options(opts: dict = Body()):
    core.play.set_options(**opts)
    return ok()

  @app.post("/api/play/push")
  def play_push(speed: float = Body(1.0, embed=True)):
    core.play.push(speed=speed)
    return ok()

  @app.post("/api/focus")
  def focus(index: int = Body(embed=True)):
    if core.owner == "play":
      core.play.set_options(focus=index)
    else:
      core.set_focus(index)
    return ok()

  return app


__all__ = ["create_app", "encode_frame", "model_payload"]
