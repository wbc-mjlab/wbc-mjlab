Welcome to WBC-MJLab!
=====================

.. figure:: ../assets/wbc_g1_collage.gif
   :width: 100%
   :alt: WBC motion tracking collage

**WBC-MJLab** is a training library for whole-body motion tracking on
`mjlab <https://github.com/mujocolab/mjlab>`_. One shared **manager-based MDP**
hosts rewards, RSI, and motion commands; **robots register as separate entities**;
paper and deploy choices are **presets and tasks** (``--task Wbc-G1``, ``Wbc-H2``, …).

Try without training
--------------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Live web demo
      :link: https://wbc-mjlab.github.io/wbc-demo/
      :link-type: url

      Run a trained policy in the browser (MuJoCo WASM) with deploy-aligned clip
      switching.

   .. grid-item-card:: Google Colab
      :link: https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb
      :link-type: url

      Cloud notebook with bundled checkpoint and sample clips — no local GPU required.

   .. grid-item-card:: Local demo
      :link: source/workflows/demo
      :link-type: doc

      ``wbc-mjlab-demo`` on bundled samples after a quick ``make sync``.

**Key features:**

- **Modular stack** — shared MDP in ``env/``; each robot is a registered entity;
  presets compose tasks without forking core code
- **Tasks, not forks** — ZEST, BeyondMimic-style RSI, deploy obs, etc. as
  ``--task`` switches on the same CLI and log layout
- **Multi-motion by design** — train on clip libraries; one policy generalizes
  across skills at runtime
- **Plug-in robots** — extension packages via ``register_wbc_extension`` (see
  :doc:`source/extensions/index`)
- **Sim → real** — train/play export ``policy.onnx`` + ``config.yaml`` for deploy runtimes

**Try it** (bundled samples):

.. code-block:: bash

   git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
   make sync
   uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 8
   uv run wbc-mjlab-train --task Wbc-G1 --dataset samples

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   source/workflows/quickstart
   source/installation
   source/workflows/demo

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   source/concepts/index
   source/architecture

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   source/usage
   source/data
   source/tasks/index
   source/mdp/index
   source/workflows/training
   source/workflows/deploy
   source/extensions/index
   source/visualization

.. toctree::
   :maxdepth: 1
   :caption: Development

   source/contributing
   source/roadmap

.. toctree::
   :maxdepth: 1
   :caption: Further Reading

   source/research

License & citation
------------------

WBC-MJLab is licensed under the Apache License, Version 2.0. See the
`LICENSE file <https://github.com/wbc-mjlab/wbc-mjlab/blob/main/LICENSE>`_.

If you use WBC-MJLab in your research, please cite the software and the
**method papers** for the tasks you reproduce (see :doc:`source/research`):

.. code-block:: bibtex

    @software{wbc_mjlab2026,
        author = {Nedelchev, Simeon and Kozlov, Lev and Domrachev, Ivan and Chaplygin, Anton},
        title = {{WBC-MJLab: Unified Whole-Body Motion Tracking on mjlab}},
        url = {https://github.com/wbc-mjlab/wbc-mjlab},
        year = {2026}
    }

Also cite **mjlab** when using the simulation stack (see :doc:`source/research`).

Acknowledgments
---------------

WBC-MJLab builds on `mjlab <https://github.com/mujocolab/mjlab>`_ (manager-based
RL API + MuJoCo Warp), with design inspiration from open WBC codebases such as
`whole_body_tracking <https://github.com/HybridRobotics/whole_body_tracking>`_ and
`GR00T-WholeBodyControl <https://github.com/NVlabs/GR00T-WholeBodyControl>`_ — see
:doc:`source/research`. Method implementations follow published WBC / motion-tracking
work cited there; please cite those papers when comparing against or reproducing their
setups.
