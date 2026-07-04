.. _api_export:

Export
======

Train / play write ``params/policy.onnx`` and ``params/config.yaml`` for deploy
runtimes. Schema version: ``wbc_tracking_params_v1``.

Workflow: :doc:`../workflows/deploy`.

Tracking params YAML
--------------------

.. autodata:: wbc_mjlab.export.tracking_params_yaml.SCHEMA_VERSION
   :annotation:

.. autodata:: wbc_mjlab.export.tracking_params_yaml.REFERENCE_OBS_TERM_NAMES
   :annotation:

``config.yaml`` (written next to the ONNX) includes:

* ``schema_version`` — must be ``wbc_tracking_params_v1``
* Joint names, action mode (``reference_residual`` vs absolute), action scales
* Actor observation term order and dims
* Reference / command layout for clip playback
* PD gains and policy step timing

Manual export:

.. code-block:: bash

   uv run wbc-mjlab-export-tracking-params --task Wbc-G1 --out /path/to/config.yaml
