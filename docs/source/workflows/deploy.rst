.. _deploy:

Deploy export
=============

Train and play write **robot-agnostic artifacts** under each run's ``params/`` folder.
Any hardware runtime that consumes the same contract can load them — no wbc-mjlab
install required on the robot computer.

Pipeline
--------

.. code-block:: text

   motion clips ──► data-to-npz ──► train / play
                                            │
                                            ▼
                              params/policy.onnx + config.yaml
                                            │
                                            ▼
                                   deploy runtime (robot)

Artifacts
---------

.. code-block:: text

   logs/rsl_rl/<experiment>/<run>/
     params/
       policy.onnx
       config.yaml
       env.yaml
       agent.yaml

``config.yaml`` (``schema_version: wbc_tracking_params_v1``) lists joint names,
observation term order, reference command layout, and PD gains — regenerated from the
task config if missing.

Manual export
-------------

.. code-block:: bash

   uv run wbc-mjlab-export-tracking-params --task <TaskId> --out /path/to/config.yaml

Example (in-tree reference task):

.. code-block:: bash

   uv run wbc-mjlab-export-tracking-params --task Wbc-G1 --out /path/to/config.yaml

End-to-end checklist
--------------------

.. code-block:: bash

   # 1. Convert motion library (once per dataset)
   uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 8

   # 2. Train
   uv run wbc-mjlab-train --task Wbc-G1 --dataset samples

   # 3. Validate in sim (also writes params/policy.onnx + config.yaml before the viewer)
   uv run wbc-mjlab-play --task Wbc-G1 --dataset samples --viewer viser

   # 4. Optional: regenerate config.yaml only
   uv run wbc-mjlab-export-tracking-params --task Wbc-G1 \
     --out logs/rsl_rl/wbc_g1/<run>/params/config.yaml

   # 5. Hand off to a deploy runtime (example: wbc-g1-deploy)
   cp logs/rsl_rl/wbc_g1/<run>/params/policy.onnx  /path/to/wbc-g1-deploy/config/policy/
   cp logs/rsl_rl/wbc_g1/<run>/params/config.yaml /path/to/wbc-g1-deploy/config/policy/

Play exports ONNX + ``config.yaml`` into the checkpoint run's ``params/`` **before**
the viewer opens. Train checkpoints also keep ``params/`` when the runner exports.

Reference runtime
-----------------

`wbc-g1-deploy <https://github.com/wbc-mjlab/wbc-g1-deploy>`_ is a **reference
implementation** for one platform (Unitree G1): ONNX inference, ``config.yaml``
parsing, and motion clip playback. Use it as a template when building a deploy stack
for your robot — the export format is not G1-specific.

See the `wbc-g1-deploy README <https://github.com/wbc-mjlab/wbc-g1-deploy>`_ for build
and run instructions. Schema details: :doc:`../api/export`.

Tips
----

- Tasks built with ``apply_wbc`` / ``apply_zest`` use **deploy-style** actor obs
  (no ``ref_joint_vel``) — preferred for sim→real export.
- SE variants (``apply_se_actor``) add anchor error + base velocity — export only if
  your runtime provides the same terms.
- Validate tracking in sim first: ``wbc-mjlab-play --viewer viser``.
