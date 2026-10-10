.. _studio:

WBC Studio
==========

The Studio runs the whole pipeline in one browser app — **robot → data → preset → train →
play** — on your machine or a GPU server you reach over the network. Everything it does is
also a CLI command or a file on disk, so you can switch between the GUI, the command line and
scripts at any point.

.. contents:: On this page
   :local:
   :depth: 1

Start the Studio
----------------

.. code-block:: console

   $ uv run wbc-mjlab-studio

     WBC Studio
       → http://192.168.x.x:8765/      from any machine on the network
       → http://127.0.0.1:8765/        this machine
       Ctrl+C to stop

Open one of the links. **Ctrl+C**, ``kill`` or closing the terminal stops the Studio cleanly:
the playback env closes, TensorBoard stops and the ports are freed. Training and conversion
jobs are detached — they keep running and reappear on the next start.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Option
     - Meaning
   * - ``--host 0.0.0.0`` (default)
     - Serve on every interface; ``--host 127.0.0.1`` for this machine only
   * - ``--port 8765``
     - UI port
   * - ``--token SECRET``
     - Network clients must open ``/?token=SECRET`` once (or type it on the sign-in page);
       it's kept in a cookie for 30 days. Clients on the machine itself never need it.
   * - ``--robot g1`` · ``--device cuda:0``
     - Robot to start with · torch device for playback and conversion
   * - ``--ui viser``
     - The same Studio as a viser app (no preset editor, no robot import)
   * - ``--no-open-browser``
     - Don't open a browser tab

.. warning::

   Like viser, the Studio has **no access control by default**: anyone who can reach the port
   can use it, and it can launch jobs and read local folders. Keep it on trusted networks or
   start it with ``--token``. If other machines can't connect, open the port
   (``sudo ufw allow 8765/tcp``). TensorBoard, started from *Train*, listens on the network
   too (port 6006 or the next free one).

Layout
------

- **Top bar** — the pipeline: five steps (keys ``1``–``5``), each with its current state
  (``g1 · 29 dof``, ``samples · 22 npz``, ``my_preset + se · 4 changes``,
  ``1 running · 40%`` …); the robot chip; the status chip (opens the activity log); theme.
- **3D view** — whatever the current step is about: the robot and its wiring, a clip, a
  policy on many envs. The icon bar toggles the reference ghost, markers, labels and camera
  follow; ``?`` lists the keys. Over a clip or a policy, the bar at the top picks the clip,
  the focused env, shuffles and pushes; the transport bar at the bottom plays and scrubs.
- **Side panel** — the current step: tabs only where they help, the main action at the
  bottom.

.. list-table:: Keyboard shortcuts
   :widths: 30 70

   * - :kbd:`1` – :kbd:`5`
     - Go to a step
   * - :kbd:`Space` · :kbd:`←` :kbd:`→`
     - Play / pause · step a frame
   * - :kbd:`[` :kbd:`]` · :kbd:`,` :kbd:`.`
     - Previous / next clip · previous / next env
   * - :kbd:`p`
     - Push the robot (Play)
   * - :kbd:`g` · :kbd:`o` · :kbd:`l` · :kbd:`f` · :kbd:`r`
     - Ghost · markers · labels · camera follow · reset camera
   * - :kbd:`?`
     - Show the shortcuts

1 · Robot
---------

Pick a robot: the built-in G1, robots of installed extensions, or robots added in the
Studio. You get its model summary and tasks; **Show on model** draws the tracking wiring
(anchor, tracked keybodies, feet), and the *Pose* sliders move the joints.

Add a robot
~~~~~~~~~~~

**Add robot** turns a humanoid's model files into a trainable robot — no code, no extension
package. Details and the file format: :doc:`extensions/custom_robots`.

1. **Robot id** — lowercase (``h1``, ``my_humanoid``). It names the tasks
   (``Wbc-H1``, ``Wbc-H1-SE``) and the data folder (``data/h1/``).
2. **Model files** — *Upload folder…* (the MJCF / URDF with its meshes, layout kept),
   *Upload files…*, or *Copy* a folder that's on the Studio machine. Pick the model file
   (not a scene that includes it).
3. **Inspect** — wbc-mjlab prepares the model (removes its actuators, keyframes and floor,
   adds a free joint and IMU sensors when missing — listed in the panel) and suggests the
   wiring and gains. The 3D view shows the model with markers:
   anchor (amber), tracked keybodies (cyan), end-effectors (blue), feet (green).
4. **Wiring** — check the anchor, keybodies, end-effectors, feet and foot sites on the model;
   click body chips to change them.
5. **Initial state** — root height (the suggestion puts the feet on the floor), the sole
   height used to correct clip heights in conversion, joint defaults.
6. **Actuators** — effort, stiffness, damping, armature per joint. Values taken from the
   model are kept; others are G1-based guesses (marked ``*``).
7. **Check** — drops the robot on a floor in its initial pose and holds it with these gains
   for 0.5 s (no policy): stable simulation, starts above the floor, lands on its feet,
   holds the pose, torque within limits.
8. **Save robot** — registers it immediately with three tasks — ``Wbc-<Id>`` (default WBC
   stack), ``Wbc-<Id>-SE`` (state-estimation actor) and ``Wbc-<Id>-Terrain-Scan`` (curriculum
   terrain + height scan under the root body) — and selects it. Edit (sliders icon) or delete
   (trash icon) it later from the robot list.

2 · Data
--------

Three tabs.

**Library** — datasets under ``data/<robot>/`` with a ring for how much is converted.
*Import* creates a dataset from a folder of CSV / PKL clips (copy, or link without copying)
or from uploaded files. Pick a dataset to browse its clips: search with ``*`` wildcards,
filter converted / missing, page through tens of thousands of clips, tick clips for
conversion, *Grid* plays up to 49 clips of a search side by side. *Scan* computes durations
(hours, length histogram).

**Clip** — click a clip to play it, **raw sources included, before conversion**. Quality
issues: joint-limit violations, joint-speed spikes, feet below or above the ground, time
airborne. Curves: root height and speed, foot heights, any joint against its limits — click
a curve to seek. For converted clips, *source* overlays the raw clip as a ghost to check the
conversion. ``[`` ``]`` step through clips.

**Convert** — *missing*, *all* or *selected* clips, output fps, batch size (parallel FK on
the GPU), mirror (needs the robot's symmetry config), debias height, input fps. **Convert N**
runs ``wbc-mjlab-data-to-npz`` as a job with progress and log.

3 · Preset
----------

The MDP a policy learns. Start from a **built-in preset** (the paper setups ``Wbc-G1``,
``Wbc-G1-Zest``, ``Wbc-G1-BinaryFailure``, ``Wbc-G1-EE``, ``Wbc-G1-Terrain-Scan`` …), switch on
**modules**, then change anything on these tabs:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Tab
     - What you can change
   * - **Observations**
     - **Actor** (policy input, also on the robot) and **critic** (privileged, training
       only), grouped into *reference* (what to track), *robot state* and *sim only*. Drop
       terms; **Add term** from every term the robot's tasks define; scale training noise
       (``0`` = off); history per term or per group; the **keybodies** a term covers (e.g.
       hands and feet only). Every term and group shows its width (``132 dims``; the
       ``height_scan`` of terrain tasks: one value per ray, 54). Actor terms the export can't
       rebuild on the robot are tagged *sim only*; terms that need a sensor the task lacks
       (``height_scan`` on a flat task) can't be added; changing the actor warns that the
       network input changed (train from scratch).
   * - **Rewards**
     - Weight, parameters and on / off per term, with a bar showing the split between
       tracking, other terms and penalties
   * - **Terminations**
     - Parameters, on / off
   * - **Sampling**
     - Reference state init (RSI: start sampling, adaptive bins), assistive wrench, reset
       noise, episode length
   * - **Randomization**
     - Domain-randomization ranges, on / off
   * - **Terrain**
     - Only for tasks on procedural terrain (``-Terrain-Scan``): the curriculum (start row,
       number of difficulty rows, difficulty range) and each sub-terrain's share and ranges
       (flat, rough noise, Perlin hills, random grid)
   * - **PPO**
     - Optimiser, rollout, network sizes

Each change gets an amber dot (click to reset that field); the footer counts the changes
against base + modules.

**Modules** are reusable pieces of a preset with no base task, shown as toggles under the
preset picker. The built-in ``se`` module is the state-estimation actor, read off the
``<task>-SE`` vs ``<task>`` pair of the robot's tasks (on G1, ``Wbc-G1`` + ``se`` gives
exactly the ``Wbc-G1-SE`` observations). Terms a module adds are tagged with its name.
Modules apply in order, then the preset's own changes.

**Save as…** stores either

- a **preset** (trainable): base + modules + changes, trained as task ``<base>--<name>``,
  logging to ``logs/rsl_rl/<experiment>--<name>/``; or
- a **module**: only the changes, no base — then a toggle in every preset.

Modules can be opened from the picker to edit them (built-in ones: copy with
*Save as module…*). **Apply live** pushes reward weights into a running Play session to watch
the terms react; training is not affected. **Train** saves and moves to the next step.

Files and run snapshots: :ref:`studio-files`.

4 · Train
---------

The preset and dataset come from the previous steps (each with a shortcut back). Set the
budget (envs, iterations — or the *1k / 5k / 30k / 100k* chips), a run name, the logger
(TensorBoard / W&B), fresh or continue from a run's checkpoint; *Advanced* takes extra
arguments and shows the exact command line. **Launch** starts a detached job
(``wbc-mjlab-train``, or ``python -m wbc_mjlab.studio.train_preset`` for your presets — single
GPU).

*Jobs* lists training and conversion jobs. A job shows progress and ETA, curves (reward,
tracking errors, per-term rewards, sampling, losses), the log, **TensorBoard**, Stop / Forget,
and **Play** for its run.

5 · Play
--------

Pick a run (or the bundled demo checkpoint) and a checkpoint — or the **zero action** (the PD
holds the reference: judge a dataset without a policy) — a motion library and 1–64 envs, then
**Load policy**. Every env shows its **reference ghost** (*aligned* to the robot's xy + yaw,
as the body rewards compare, or in *world* frame to see drift) and keybody targets coloured by
error. The focused env's **reward terms** and a HUD (return, body / joint error, anchor drift,
episode time) update live; *Tune* jumps to the rewards. The bar over the 3D view picks the
focused env's clip (it loops), shuffles the others and **pushes** the robot. Another
checkpoint swaps in place; envs that blow up are reset automatically.

On **terrain tasks** (``-Terrain-Scan``) each env stands on a tile of its own terrain, sampled
from the simulation and following the robot — grey flat, rising towards teal; envs are spread
over every difficulty row, as in ``wbc-mjlab-play``.

Runs of your own presets are rebuilt from their snapshot (``params/studio_preset.yaml``), so
they replay as trained even after the preset changed.

.. _studio-files:

Files and scripting
-------------------

The Studio keeps its own files per user, outside the repo, in ``~/.config/wbc-mjlab/``
(``presets/``, ``modules/``, ``robots/``); jobs live in ``logs/studio/jobs/``, runs in
``logs/rsl_rl/`` as usual. Presets are short YAML diffs you can edit by hand or train
without the Studio:

.. code-block:: bash

   uv run python -m wbc_mjlab.studio.train_preset --preset my_preset --dataset samples

Every training run of a preset keeps the preset as trained in ``params/studio_preset.yaml``,
so editing or deleting a preset never changes how older runs replay.

.. note::

   Two meanings of *preset*: code presets (``apply_wbc``, ``apply_zest`` …) build the
   registered tasks; a **Studio preset** is a YAML diff on top of one of those tasks. See
   :doc:`concepts/presets_and_tasks`.

- File formats: :doc:`reference/configuration`
- Everything the GUI does is a JSON call — script it: :doc:`reference/studio_api`

viser version
-------------

``--ui viser`` serves the same core as a viser app (Robot, Data, Train and Play tabs). The
preset editor, modules and robot import are web-only.

.. seealso::

   - :doc:`data` · :doc:`workflows/training` · :doc:`visualization` — the same steps in depth
   - :doc:`extensions/custom_robots` — what *Add robot* does
   - :doc:`troubleshooting` — Studio not reachable, port in use, failing robot check
