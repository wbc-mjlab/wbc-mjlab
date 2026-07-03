ZEST + state-estimation
=======================

Task id: ``Wbc-G1-Zest-SE`` · In-tree reference on robot ``g1``.

Same env rewards/RSI/terminations as :doc:`wbc-g1-zest`; actor obs match :doc:`wbc-g1-se`.

- **Presets:** ``apply_zest`` + ``apply_se_actor`` + ``wire_g1_imu_sensors``

Train & play
------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Zest-SE --dataset samples
   uv run wbc-mjlab-play --task Wbc-G1-Zest-SE --dataset samples --viewer viser

Logs: ``logs/rsl_rl/wbc_g1_zest_se/<run>/``

Tips
----

- Compare learning curves against :doc:`wbc-g1-zest` to measure SE obs impact.
