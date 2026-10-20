.. _quickstart:

Quickstart
==========

Train and watch your first whole-body tracking policy on the bundled sample clips. Every step
works in the **Studio** (browser) or on the **command line** — pick a tab once and the others
follow.

.. admonition:: Before you start
   :class: tip

   - wbc-mjlab installed (:doc:`../installation`) — ``uv run wbc-mjlab-list-envs`` lists
     ``Wbc-G1`` and friends
   - An NVIDIA GPU for training (CPU is enough for steps 1–2)
   - About 15 minutes for a smoke run; a real policy trains for hours

   No GPU? Try the `live demo <https://wbc-mjlab.github.io/wbc-demo/>`_ or
   :doc:`Colab <demo>` instead.

1. Open the pipeline
--------------------

.. tab-set::
   :sync-group: ui

   .. tab-item:: Studio
      :sync: studio

      .. code-block:: bash

         uv run wbc-mjlab-studio

      Open the printed link. The top bar shows the pipeline: **Robot → Data → Preset → Train →
      Play**. G1 is selected.

   .. tab-item:: Command line
      :sync: cli

      Everything below runs from the repository root.

2. Look at the sample clips
---------------------------

The 10 bundled G1 clips (walk, run, dance, fall and get up, flips …) are CSV files. Training
reads them as they are — resampling, velocities and body poses are computed on the GPU while
it runs — so there is nothing to convert.

.. tab-set::
   :sync-group: ui

   .. tab-item:: Studio
      :sync: studio

      *Data* → pick **samples** → click a clip to watch it; the *Clip* tab lists anything
      suspicious (feet in the floor, joint limits).

   .. tab-item:: Command line
      :sync: cli

      Nothing to do: ``data/g1/samples/`` holds the CSVs training reads.

3. Train
--------

``Wbc-G1`` is the default stack (:doc:`which task? <../tasks/index>`). A smoke run of 1000
iterations shows the loop works; a full run is about 30k.

.. tab-set::
   :sync-group: ui

   .. tab-item:: Studio
      :sync: studio

      *Preset* → **Wbc-G1** → **Train** → set iterations to **1k** → **Launch**. The job
      shows progress, curves and a link to TensorBoard.

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: bash

         uv run wbc-mjlab-train --task Wbc-G1 --dataset samples --agent.max-iterations 1000

**Check:** ``logs/rsl_rl/wbc_g1/<run>/`` fills with ``model_*.pt`` checkpoints and
``params/policy.onnx`` + ``params/config.yaml``.

4. Watch the policy
-------------------

.. tab-set::
   :sync-group: ui

   .. tab-item:: Studio
      :sync: studio

      *Play* → pick the run → **Load policy**. Each env shows its reference as a ghost; pick a
      clip in the bar above the 3D view, push the robot with :kbd:`p`.

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: bash

         uv run wbc-mjlab-play --task Wbc-G1 --dataset samples --viewer viser

      Open the viser link; the sidebar shows the clip, reference alignment and the adaptive
      sampling bins.

After 1000 iterations the robot follows the easy clips roughly; flips and get-ups need the
full run.

What you have now
-----------------

.. code-block:: text

   logs/rsl_rl/wbc_g1/<run>/
     model_*.pt              checkpoints (continue with --agent.resume True)
     params/policy.onnx      the policy, ready for a robot runtime
     params/config.yaml      observation layout, joints, PD gains for that runtime

Next steps
----------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: :octicon:`database` Your own clips
      :link: ../data
      :link-type: doc

      Import and check a motion library.

   .. grid-item-card:: :octicon:`rocket` A full training run
      :link: training
      :link-type: doc

      Iterations, resume, many GPUs, logs.

   .. grid-item-card:: :octicon:`list-unordered` Other tasks
      :link: ../tasks/index
      :link-type: doc

      ZEST, state estimation, end-effector command, rough terrain.

   .. grid-item-card:: :octicon:`cpu` On the robot
      :link: deploy
      :link-type: doc

      Hand ``policy.onnx`` + ``config.yaml`` to a deploy runtime.
