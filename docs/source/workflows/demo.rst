.. _demo:

Demo & Colab
============

Try WBC-MJLab without a full training run.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Live web demo
      :link: https://wbc-mjlab.github.io/wbc-demo/
      :link-type: url

      **Open in browser** — MuJoCo WASM policy with deploy-aligned clip switching.

      Source: `wbc-demo <https://github.com/wbc-mjlab/wbc-demo>`_.

   .. grid-item-card:: Google Colab
      :link: https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb
      :link-type: url

      **Open notebook** — bundled checkpoint + sample clips in the cloud.

      Notebook: ``notebooks/demo.ipynb`` in the repo root.

Quick links
-----------

- `Live web demo <https://wbc-mjlab.github.io/wbc-demo/>`_
- `Open in Colab <https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb>`_
- `wbc-demo source <https://github.com/wbc-mjlab/wbc-demo>`_

Bundled local demo
------------------

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 8
   uv run wbc-mjlab-demo

Uses a bundled checkpoint and sample clips for interactive play.

Viser debug (RSI + reference)
-----------------------------

Interactive play with WBC Viser overlays — reference ghost/frames, align xy/yaw,
tracking-error colors, and RSI bin strips:

.. code-block:: bash

   uv run wbc-mjlab-play --task Wbc-G1 --dataset samples --viewer viser

Details: :doc:`../visualization`.

Plot RSI bins offline:

.. code-block:: bash

   uv run wbc-mjlab-plot-rsi-bins logs/rsl_rl/wbc_g1/<run>/params/rsi_bin_stats.npz
