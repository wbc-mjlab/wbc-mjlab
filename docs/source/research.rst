.. _research:

Research & citations
====================

Citing WBC-MJLab
----------------

If you use WBC-MJLab in your research, please cite the software:

.. literalinclude:: _static/refs.bib
   :language: bibtex
   :start-at: @software{wbc_mjlab2026,
   :end-before: @article{zakka2026mjlab,

Also cite **mjlab** — WBC-MJLab is an extension of the simulation and RL stack:

.. literalinclude:: _static/refs.bib
   :language: bibtex
   :start-at: @article{zakka2026mjlab,
   :end-before: @article{zest2026,

Lineage & inspiration
---------------------

WBC-MJLab aims to be a **lightweight, accessible, modular** take on whole-body motion
tracking — presets and tasks on a shared MDP, plug-in robots, and a small CLI — rather
than a monolithic Isaac Lab or vendor-specific stack. The design is informed by prior
open WBC codebases (reward/RSI patterns, multi-motion training, deploy export) while
running on **mjlab** + MuJoCo Warp instead of Isaac Sim.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Project
     - Role
   * - `HybridRobotics/whole_body_tracking <https://github.com/HybridRobotics/whole_body_tracking>`_
     - Reference **BeyondMimic** training stack (Isaac Lab); adaptive RSI and tracking
       MDP ideas. Cite the `BeyondMimic paper <https://arxiv.org/abs/2508.08241>`_ when
       comparing methods; the repo itself is cited below.
   * - `NVlabs/GR00T-WholeBodyControl <https://github.com/NVlabs/GR00T-WholeBodyControl>`_
     - NVIDIA's unified humanoid WBC platform (decoupled controllers, Isaac-GR00T /
       GEAR-SONIC lineage). Useful reference for deploy-oriented WBC layout; WBC-MJLab
       targets a smaller, robot-agnostic training surface on mjlab.

When you build on ideas from these codebases, cite them alongside WBC-MJLab and mjlab:

.. literalinclude:: _static/refs.bib
   :language: bibtex
   :start-at: @software{hybridrobotics_wbt,
   :end-before: @software{wbc_h2_ext,

Method papers (cite when using a task)
--------------------------------------

When you train or compare a registered task, **cite the original method paper**
in addition to wbc-mjlab and mjlab. See :doc:`tasks/index`.

.. list-table::
   :header-rows: 1
   :widths: 18 42 25 15

   * - Method
     - Title
     - wbc-mjlab task(s)
     - Year
   * - **ZEST**
     - `Zero-shot Embodied Skill Transfer <https://arxiv.org/abs/2602.00401>`_
     - ``Wbc-G1``, ``Wbc-G1-Zest``, ``Wbc-G1-SE``, ``Wbc-G1-Zest-SE``
     - 2026
   * - **BeyondMimic**
     - `From Motion Tracking to Versatile Humanoid Control <https://arxiv.org/abs/2508.08241>`_
     - ``Wbc-G1-BinaryFailure``
     - 2025
   * - **SONIC**
     - `Supersized motion tracking <https://arxiv.org/abs/2511.07820>`_
     - partial (body-velocity rewards, obs.\ corruption); full preset planned
     - 2025
   * - **OmniXtreme**
     - `High-dynamic scalable tracking <https://arxiv.org/abs/2602.23843>`_
     - partial (G1 torque envelope)
     - 2026

BibTeX entries for method papers
--------------------------------

.. literalinclude:: _static/refs.bib
   :language: bibtex
   :start-at: @article{zest2026,
   :end-before: @article{clot2026,

Related projects
----------------

Extensions and runtimes in the wbc-mjlab org:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Project
     - Description
   * - `wbc-mjlab/wbc-mjlab <https://github.com/wbc-mjlab/wbc-mjlab>`_
     - Training library (this repo)
   * - `wbc-mjlab/wbc-mjlab-extension-h2 <https://github.com/wbc-mjlab/wbc-mjlab-extension-h2>`_
     - Reference Unitree H2 robot extension
   * - `wbc-mjlab/wbc-g1-deploy <https://github.com/wbc-mjlab/wbc-g1-deploy>`_
     - Reference deploy runtime (example: Unitree G1 ONNX + clips)
   * - `wbc-mjlab/wbc-demo <https://github.com/wbc-mjlab/wbc-demo>`_
     - In-browser MuJoCo WASM demo

Technical report
----------------

A formal tech report lives in ``paper/main.tex`` (local draft; see repo
``.gitignore``). BibTeX for copy-paste citations: ``docs/source/_static/refs.bib``.
Citation metadata: ``CITATION.cff`` at the repo root. arXiv submission tracked in
`issue #31 <https://github.com/wbc-mjlab/wbc-mjlab/issues/31>`_.
