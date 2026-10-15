.. _reference-studio-api:

Studio HTTP API
===============

Everything the :doc:`Studio <../studio>` does goes through a JSON API on the same port, so the
pipeline can be scripted from CI, a notebook or another machine.

.. note::

   The API serves the Studio's own UI. It is stable enough for scripts but not versioned —
   endpoints may change between releases.

.. contents:: On this page
   :local:
   :depth: 1

Conventions
-----------

.. list-table::
   :widths: 22 78

   * - **Base URL**
     - ``http://<host>:8765`` (``--host`` / ``--port`` of ``wbc-mjlab-studio``)
   * - **Auth**
     - None by default. With ``--token SECRET``, clients other than the machine itself send
       ``Cookie: wbc_studio_token=SECRET`` (or open ``/?token=SECRET`` once in a browser).
       Unauthorised requests get ``401``.
   * - **Requests**
     - JSON bodies (``content-type: application/json``); query parameters on ``GET``.
       Uploads are raw bytes (``PUT``).
   * - **Responses**
     - JSON. Actions return ``{"ok": true, …}``; slow actions run in the background and
       return ``{"ok": true, "task": "<name>"}`` — progress arrives as ``busy`` events on
       the websocket.
   * - **Errors**
     - ``400 {"detail": "<message>"}`` for invalid input (unknown dataset, bad name …)

Example: the whole pipeline
---------------------------

.. code-block:: bash

   S=http://127.0.0.1:8765
   J='content-type: application/json'

   # data: import a folder of clips and convert them
   curl -s -X POST $S/api/data/import -H "$J" \
     -d '{"source": "/path/to/csvs", "name": "my_clips", "link": true}'
   curl -s -X POST $S/api/data/convert -H "$J" \
     -d '{"dataset": "my_clips", "which": "missing", "batch_size": 8}'

   # preset: Wbc-G1 + the se module, twice the joint-velocity noise
   curl -s -X PUT $S/api/presets/se_noisy -H "$J" -d '{
     "base": "Wbc-G1", "modules": ["se"],
     "overrides": {"observations": {"actor": {"terms": {"joint_vel": {"noise_scale": 2}}}}}}'

   # train it and follow the job
   curl -s -X POST $S/api/train/launch -H "$J" -d '{
     "task": "Wbc-G1--se_noisy", "dataset": "my_clips", "num_envs": 4096, "max_iterations": 30000}'
   curl -s $S/api/jobs                         # status, progress, ETA
   curl -s "$S/api/jobs/<job>?tail=40"         # one job + log tail

   # play the run (watch it in the browser)
   curl -s -X POST $S/api/play/configure -H "$J" -d '{
     "run": "wbc_g1--se_noisy/<run>", "checkpoint": "model_29999.pt",
     "motion": "<library path from /api/play/libraries>", "num_envs": 16}'

The same from Python:

.. code-block:: python

   import requests

   S = "http://127.0.0.1:8765"
   job = requests.post(f"{S}/api/train/launch", json={
     "task": "Wbc-G1", "dataset": "samples", "num_envs": 4096, "max_iterations": 30000,
   }).json()["job"]
   print(requests.get(f"{S}/api/jobs/{job}").json()["progress"])

Robots
------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Endpoint
     - Description
   * - ``GET /api/robots``
     - Registered robots with model summary, datasets and runs
   * - ``GET /api/robot``
     - Selected robot: joints, wiring (anchor, keybodies, feet), tasks
   * - ``POST /api/robot/select`` ``{name}``
     - Select a robot (background task)
   * - ``POST /api/robot/joint`` ``{name, value}`` · ``/reset`` ``{zero}`` · ``/show`` ·
       ``/options`` ``{show_wiring}``
     - Pose sliders, default / zero pose, show the robot, wiring markers

Adding a robot (:doc:`../extensions/custom_robots`):

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Endpoint
     - Description
   * - ``GET /api/robots/import``
     - Robots folder, saved robots, unfinished uploads
   * - ``PUT /api/robots/import/upload/{id}/{path}``
     - Upload one model file at its relative path (meshes keep their layout)
   * - ``POST /api/robots/import/path`` ``{id, source}``
     - Copy a folder on the Studio machine
   * - ``GET /api/robots/import/files?id=``
     - Model files in the folder, the likely main model first
   * - ``POST /api/robots/import/inspect`` ``{id, model}``
     - Prepare and inspect the model: bodies, joints, suggested wiring and gains
   * - ``POST /api/robots/import/preview`` ``{wiring}``
     - Show the draft with wiring markers
   * - ``POST /api/robots/import/check`` ``{spec}``
     - Settle check: ``{ok, checks: [{name, ok, text}]}``
   * - ``POST /api/robots/import/save`` ``{spec}`` · ``/cancel``
     - Save and register (``{id, task}``) · leave the import view
   * - ``GET /api/robots/{id}/spec`` · ``DELETE /api/robots/{id}``
     - A Studio robot's ``robot.yaml`` (:ref:`reference-robot-file`) · delete it

Data
----

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Endpoint
     - Description
   * - ``GET /api/data/datasets``
     - Datasets of the robot with source / converted / missing counts
   * - ``GET /api/data/clips?dataset=&q=&status=&page=&size=``
     - Clip listing; ``q`` supports ``*`` wildcards, ``status`` ∈ ``all``, ``converted``,
       ``missing``, ``raw``
   * - ``GET /api/data/series`` · ``/joints`` · ``/preview``
     - Curves of the loaded clip · joint list · preview state
   * - ``POST /api/data/load`` ``{dataset, clip, source}`` · ``/grid`` · ``/close``
     - Preview a clip (``source``: ``auto``, ``npz``, ``raw``) · a grid of clips · close
   * - ``POST /api/data/frame`` · ``/time`` · ``/playing`` · ``/options``
     - Preview transport and display options
   * - ``POST /api/data/scan`` ``{dataset}``
     - Duration scan (hours, length histogram)
   * - ``POST /api/data/convert``
       ``{dataset, which, clips, batch_size, output_fps, mirror, debias_z, input_fps}``
     - Conversion job (``which`` ∈ ``missing``, ``all``, ``selected``) → ``{job}``
   * - ``POST /api/data/new`` ``{name}`` · ``/import`` ``{source, name, link}``
     - New empty dataset · import a folder (copy, or link without copying)
   * - ``PUT /api/data/upload/{dataset}/{file}`` · ``POST /api/data/uploaded``
     - Upload one clip · refresh after an upload batch

Presets and modules
-------------------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Endpoint
     - Description
   * - ``GET /api/presets``
     - Tasks, your presets, modules, the config directory
   * - ``GET /api/presets/schema?base=&modules=``
     - Every editable knob of a task (+ modules) with defaults, and the observation term
       library
   * - ``GET | PUT | DELETE /api/presets/{name}``
     - Read · write ``{base, overrides, modules, description}`` · delete a preset
       (:ref:`reference-preset-file`)
   * - ``GET | PUT | DELETE /api/modules/{name}``
     - Read · write ``{overrides, description}`` · delete a module

Training and jobs
-----------------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Endpoint
     - Description
   * - ``GET /api/tasks`` · ``GET /api/runs``
     - Tasks of the robot (built-in and presets) · runs on disk
   * - ``POST /api/train/launch`` ``{task, dataset, num_envs, max_iterations, run_name,
       resume_run, resume_checkpoint, logger, extra}``
     - Launch a detached training job → ``{job}``
   * - ``GET /api/jobs`` · ``GET /api/jobs/{name}?tail=``
     - Jobs with status, progress and ETA · one job with its log tail
   * - ``POST /api/jobs/{name}/stop`` · ``DELETE /api/jobs/{name}``
     - Stop · forget a job (its run stays)
   * - ``GET /api/curves?job=&run=&group=``
     - Training curves (``group``: ``train``, ``errors``, ``rewards``, ``sampling``,
       ``loss``, ``all``)
   * - ``GET /tensorboard?run=``
     - Start TensorBoard if needed and redirect to it

Play
----

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Endpoint
     - Description
   * - ``GET /api/play`` · ``/api/play/runs`` · ``/api/play/libraries?run=`` · ``/api/play/clips``
     - Playback state · runs · motion libraries for a run · clips of the session
   * - ``POST /api/play/configure`` ``{run, checkpoint, motion, num_envs, terminations}``
     - Load a policy (``checkpoint: ""`` = zero action) — rebuilds in the background
   * - ``POST /api/play/stop`` · ``/playing`` · ``/step`` · ``/restart``
     - Stop · pause / resume · single step · reset every env
   * - ``POST /api/play/clip`` ``{clip, env}`` · ``/api/focus`` ``{index}``
     - Pin a clip to an env (it loops) · focus an env
   * - ``POST /api/play/options`` · ``/push`` ``{speed}`` · ``/rewards`` ``{rewards}``
     - Ghost mode and targets · push the robot · live reward weights
   * - ``GET /api/scene/terrain.bin``
     - Terrain tiles under the robots (binary mesh, see below)

Plots
-----

The data behind the Plots window (:ref:`studio-plots`). Each channel is
``{id, name, group, unit, lower, upper}`` (``unit``: ``rad``, ``m``, ``Nm``, ``reward``,
``raw`` or ``""`` for actions; joint torques with effort limits add ``limit_note``, where the
limits come from); ``ref`` is NaN where a channel has no reference.

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Endpoint
     - Description
   * - ``GET /api/plots/clip``
     - The clip open in Data, JSON: ``{title, fps, frames, channels, data, ref, ref_label}``
       with ``data`` / ``ref`` ``rows × channels`` float32 arrays (little-endian, base64) —
       ``ref`` is the raw source of a converted clip, resampled to its frames (or ``null``).
       409 when no single clip is open
   * - ``GET /api/plots/play?env=&since=&epoch=``
     - A Play env's record (default ``env``: the focused one), binary:
       ``[u32 header length][header JSON][pad to 4]`` then float32 ``data``, ``ref`` and
       ``meta`` rows (little-endian; the header's ``cols`` gives their column counts).
       The header: ``{epoch, env, fps, start, end, fresh}``; rows ``since``… of the record.
       ``ref`` holds the ``ref_cols`` channels only (root and joints); ``meta`` the clip index,
       frame in the clip and episode step per row. Without ``epoch`` or for another one
       (``fresh: true``: another env, a restart) it returns every kept row (last 60 s) and the
       header adds ``channels``, ``ref_cols``, ``clips`` and ``history_s``. Every env of a
       session of up to 32 envs is recorded from the start; in a larger one an env records
       from when it is first focused or fetched. 409 without a session

.. code-block:: python

   import json, struct, numpy as np, requests

   S = "http://127.0.0.1:8765"
   blob = requests.get(f"{S}/api/plots/play", params={"env": 0}).content
   (hlen,) = struct.unpack_from("<I", blob)
   head = json.loads(blob[4 : 4 + hlen])
   rows = head["end"] - head["start"]
   flat = np.frombuffer(blob, "<f4", offset=4 + hlen)
   robot, ref, meta = np.split(flat, np.cumsum([rows * c for c in head["cols"]])[:2])
   robot, ref = robot.reshape(rows, -1), ref.reshape(rows, -1)
   knee = [c["id"] for c in head["channels"]].index("left_knee_joint")
   ref_knee = ref[:, head["ref_cols"].index(knee)]
   rmse = np.sqrt(np.nanmean((robot[:, knee] - ref_knee) ** 2))

Live updates (websocket)
------------------------

``/ws`` streams binary **frames** and JSON **events**.

A frame is ``[u32 header length][header JSON][padding to 4 bytes][float32 blob]``. The
header carries the instance count, the owner of the scene (``robots``, ``data``, ``play``,
``import``), the HUD, ``model_version`` and ``terrain_version``, and the offsets of the
overlays in the blob. The blob holds body poses ``(n, nbody, 7)`` (position + wxyz), then the
reference ghost poses, then overlay data. When ``model_version`` changes, clients fetch
``/api/robot/model`` + ``/api/robot/model.bin``; when ``terrain_version`` changes,
``/api/scene/terrain.bin`` (``[u32 nv][u32 nf][f32 verts][u8 colours, padded][u32 faces]``).

Events are JSON ``{"type": "event", "topic": …, "payload": …}`` with topics ``log``,
``busy``, ``robot``, ``robots``, ``data``, ``jobs``, ``play``, ``presets``, ``owner`` and
``focus``.

.. seealso::

   :doc:`../studio` · :doc:`configuration`
