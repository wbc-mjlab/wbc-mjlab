.. _recipes:

How do I…?
==========

Short answers to common goals. Each link jumps to the full guide.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - I want to…
     - Go here
   * - **Try without a GPU**
     - `Live demo <https://wbc-mjlab.github.io/wbc-demo/>`_ ·
       `Colab <https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb>`_ ·
       :doc:`workflows/demo`
   * - **Run the full local loop** (convert → train → play)
     - :doc:`workflows/quickstart`
   * - **Pick the right ``--task``**
     - :ref:`which-task` (on Quickstart)
   * - **Resume training**
     - :doc:`workflows/training`
   * - **Export ONNX + ``config.yaml`` for the robot**
     - :doc:`workflows/deploy`
   * - **Convert LAFAN / GMR / custom clips**
     - :doc:`data`
   * - **Visualize motion NPZ or RSI bins**
     - :doc:`visualization`
   * - **Add a new robot (extension package)**
     - :doc:`extensions/index` · :doc:`api/extension`
   * - **Add a paper-style task / preset**
     - :doc:`tasks/index` · :doc:`api/presets`
   * - **Look up observation term sizes**
     - :ref:`reference-obs-dims` (Shared MDP) · ``config.yaml`` after export
   * - **Look up a reward / RSI / command symbol**
     - :doc:`api/mdp`
   * - **Fix a common error**
     - :doc:`troubleshooting`

Philosophy (shared MDP, tasks not forks): :doc:`concepts/index`.
