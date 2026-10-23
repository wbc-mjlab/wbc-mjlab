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
  follow, and opens **Plots** (:ref:`studio-plots`); ``?`` lists the keys. Over a clip or a
  policy, the bar at the top picks the clip, the focused env, shuffles and pushes; the
  transport bar at the bottom plays and scrubs.
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
   * - :kbd:`j`
     - Plots window (:ref:`studio-plots`)
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

Training reads a dataset's source clips directly — nothing to convert (:ref:`data-online`).

**Library** — datasets under ``data/<robot>/``. *Import* creates a dataset from a folder of
CSV / PKL clips (copy, or link without copying) or from uploaded files. Pick a dataset to
browse its clips: search with ``*`` wildcards, filter exported / not exported, page through
tens of thousands of clips, tick clips for export, *Grid* plays up to 49 clips of a search
side by side. *Scan* computes durations (hours, length histogram).

**Clip** — click a clip to play it, straight from its source file. Quality
issues: joint-limit violations, joint-speed spikes, feet below or above the ground, time
airborne. Click an issue to open it in **Plots** (:ref:`studio-plots`): its joints or feet
plotted against their limits, the clip paused at the first frame where it happens. The
*Plots* button (or :kbd:`j`) opens the clip's curves. For exported clips, *source* overlays
the raw clip as a ghost. ``[`` ``]`` step through clips.

**Export** — optional: writes ``npz/<clip>.npz`` (body poses and velocities) for tools that
want them — *not exported*, *all* or *selected* clips, output fps, mirror, debias height,
input fps. **Export N** runs ``wbc-mjlab-data-to-npz`` as a job with progress and log.

3 · Preset
----------

What a policy learns: what it sees, how it is rewarded, where episodes start and what is
randomized. How it learns — the method and its settings — is a *recipe*, picked in
:ref:`Train <studio-train>`, so any preset trains with PPO or the world model. Start from a
**built-in preset** (the paper setups ``Wbc-G1``,
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
       (``0`` = off); history per term or per group (steps back); for reference terms,
       **frames ahead** and their **stride** (the reference now and later frames, e.g. 10
       frames 2 steps apart: :ref:`observations-ahead`); the **keybodies** a term covers
       (e.g. hands and feet only). Every term and group shows its width (``132 dims``; the
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

Each change gets an amber dot (click to reset that field); the footer counts the changes
against base + modules. A preset saved before recipes may still hold PPO settings: they
keep applying, and a note above the tabs offers **Move to a recipe**.

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

.. _studio-train:

4 · Train
---------

The preset and dataset come from the previous steps (each with a shortcut back); *Start* is a
fresh policy or a run of the same preset and method to continue.

**How to train** picks the **method** — **PPO**, the default, which trains every preset, or
the **World model** (experimental, :doc:`concepts/world_model`) — and a **recipe**: a method
with its settings, saved by name and usable with any preset. The built-in recipes are *PPO*,
*World model · paper* and *World model · 24 GB GPU* (2048 imagined rollouts, the default on
a GPU under 40 GB). For the world model the Studio first checks that it can train the preset
— a 2-env copy of the env is built in the background (~10 s) — and says which rewards it
computes exactly, which it learns and which randomization the model is given; a preset it
can't train (an actor that reads a height scan) is named with the reason and can't be
launched.

**Settings** (closed until opened) lists every setting of the method with the preset's
values as defaults: the main ones first, then the method's own, the policy network (the same
for both methods: the policy that is exported), the value network and rollouts. Changes
apply to this run; the save button keeps them as a recipe (``recipes/<name>.yaml``), and
switching methods keeps the policy network. Continuing a run keeps the networks its
checkpoint was built with, shown locked.

Then the budget: envs and iterations. The chips are typical budgets of the method — an
iteration is 24 env steps for PPO, 5 for the world model — and the estimated training time
comes from earlier runs of the method on this machine (of the same preset when there is
one). A run name, the logger (TensorBoard / W&B). *Motion* adds left/right
mirrored clips (needs the robot's symmetry config) and the height debias; Play applies the
options a run was trained with. *Advanced* takes extra
arguments and shows the exact command line. **Launch** starts a detached job
(``wbc-mjlab-train``, or ``python -m wbc_mjlab.studio.train_preset`` for your presets — single
GPU).

*Jobs* lists training and export jobs with their method and recipe. A job shows progress and ETA,
curves (reward, tracking errors, per-term rewards, sampling, and the losses of its method),
the log, **TensorBoard**, Stop, and **Play** for its run. A job that exits with an
error (a wrong argument included) is marked *failed*. The log's expand button (or a double
click) opens it full size: up to its last 5000 lines, following the end while the job runs,
with a filter (text or ``/regex/``, Ctrl+F), wrap, errors in red and warnings in amber, copy
and **Download** of the whole file. While a world model compiles its first update (minutes,
cached for later runs) the job says so. The form warns when a job is already running: jobs
share the GPU, and two world-model jobs don't fit in 24 GB.

**Cleaning up.** A finished, failed or stopped job has a **×**, and **Clear finished** removes
all of them. Removing a job deletes its record and log under ``logs/studio/jobs``. Tick
*Also delete its run* to delete its run folder (checkpoints, curves) too; that's on by default
when the run saved nothing past ``model_0.pt``. The Data pane's export jobs have the same
buttons; their exported clips always stay.

**Runs on disk** (in *Jobs*, or the disk icon in Play) lists every run folder under
``logs/rsl_rl``: its task, method, the job that wrote it, checkpoint count, last iteration,
size, and the total. Select runs by hand or with *No progress* (at most one checkpoint),
*Failed / stopped* or *All*. Then:

- **Delete** removes the run folders and their jobs.
- **Prune** keeps each run's last checkpoint, and optionally every 1k / 5k / 10k iterations;
  the older checkpoints are deleted.

A world-model checkpoint is about 120 MB, so pruning a long run frees gigabytes. The run's
curves stay in TensorBoard, and Play and resuming use the last checkpoint. You can't change a
run while a job trains it or while it is loaded in Play.

5 · Play
--------

Pick a run (or the bundled demo checkpoint) and a checkpoint — or the **zero action** (the PD
holds the reference: judge a dataset without a policy) — a motion library and the number of
envs (1, 4, 16, 64, or any count up to 4096 typed in the box), then
**Load policy**. Every env shows its **reference ghost** (*aligned* to the robot's xy + yaw,
as the body rewards compare, or in *world* frame to see drift) and keybody targets coloured by
error. The focused env's **reward terms** and a HUD (return, body / joint error, anchor drift,
episode time) update live; *Tune* jumps to the rewards. The bar over the 3D view picks the
focused env's clip (it loops), shuffles the others and **pushes** the robot. Click a robot
(or its label) to focus it. Another checkpoint swaps in place; envs that blow up are reset
automatically. Runs trained with the world model are tagged *world model*; with runs of
both methods on disk, *All / PPO / World model* filters the list. Play builds the policy
the way its method trained it, and refuses a run whose task now builds a different policy
input (other terms, frames ahead or history), naming what changed. **TensorBoard** opens the run's training curves, also for runs trained from the
command line; *All in TensorBoard* over the run list (and *TensorBoard · all runs* under
*Train* → *Jobs*) opens every run under ``logs/rsl_rl`` together.

*Session* changes the running session without reloading the policy: **Environments** resizes
it at once up to the envs already built (a power of two, at least 16; more rebuilds it in the
background), and **Terminations** switches on or off live (off: no episode ends, time-outs
included).

On **terrain tasks** (``-Terrain-Scan``) each env stands on a tile of its own terrain, sampled
from the simulation and following the robot — grey flat, rising towards teal; envs are spread
over every difficulty row, as in ``wbc-mjlab-play``.

Runs of your own presets are rebuilt from their snapshot (``params/studio_preset.yaml``), so
they replay as trained even after the preset changed.

.. _studio-plots:

Plots
-----

The chart icon in the 3D view's icon bar (or :kbd:`j`) opens **Plots**: a window over the
stage you can drag, resize and make full screen (:kbd:`F`). It plots what the stage shows:

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Stage
     - Channels
     - Dashed
   * - **Data** — one clip
     - Root position, root roll / pitch / yaw, root speed (horizontal), every joint (with
       its limits), the feet's sole heights
     - A converted clip's raw source, resampled to the clip's frames: what conversion
       changed
   * - **Play** — the focused env
     - The robot's root and joints, PD targets, joint torques (the motor's, with its effort
       limits; the friction a Unitree actuator takes off it as its own group), actions,
       per-keybody tracking
       error, every reward term **weighted** (as summed into the reward, with the total) and
       **raw** (before the weight: the tracking kernels in [0, 1])
     - The reference the policy tracks — for each row, the frame that step's rewards compared
       the robot to

Pick up to eight channels from the list (search, or *plot* a whole group — for rewards the
largest contributors); each keeps its colour while plotted. One plot shows one unit: picking
a channel of another unit (an angle after a torque) starts a new plot rather than adding a
second y-axis.

- **Compare** — *Reference* draws the reference dashed next to each channel; *Error* plots
  actual − reference instead. The legend shows each channel's RMSE against its reference over
  the visible time span; the tooltip reads actual, reference and their difference.
- **Time** — scroll to zoom, :kbd:`Shift`-drag to pan, :kbd:`Ctrl`-drag (or *Zoom*) a box to
  zoom time and values, double-click to see everything; *from / to* sets an exact range in
  seconds. On a clip, dragging scrubs the clip and the stage follows; setting a range moves
  the clip there.
- **Play** — the window shows the last 60 s of the focused env and follows the newest steps
  (*Live*, over the last 5–60 s), updated with every stage frame; a click pins a cursor to
  read a moment while the policy keeps running. Dashed lines mark each new episode or clip,
  with the clip's name. Every env of a session of up to 32 envs is recorded, so focusing
  another env (also while paused) shows its record at once; in a larger session an env
  records from when it is first focused.
- **Units** — *Pos / Vel / Acc* (finite differences, not across episode boundaries) and
  *deg / rad* for joint and root channels; *Limits* shades the joint limits — and, for
  joint torques, the effort limits: the tightest of the asset's ``actuatorfrcrange`` /
  actuator ``forcerange`` (× gear), the actuator's ``effort_limit`` and, for Unitree
  actuators, the peak of the torque–speed envelope (lower at speed). The status line counts
  the frames outside the joint limits, or at the effort limit (saturated).
- **Table view** — the download button saves the plotted channels (and their references) as
  CSV.

.. _studio-files:

Files and scripting
-------------------

The Studio keeps its own files per user, outside the repo, in ``~/.config/wbc-mjlab/``
(``presets/``, ``modules/``, ``recipes/``, ``robots/``); jobs live in ``logs/studio/jobs/``, runs in
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
