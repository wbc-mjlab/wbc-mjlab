.. _research:

Research & citations
====================

Citing WBC-MJLab
----------------

If you use WBC-MJLab in your research, please cite the software:

.. code-block:: bibtex

    @software{wbc_mjlab2026,
        author = {Nedelchev, Simeon and Kozlov, Lev and Domrachev, Ivan and Chaplygin, Anton},
        title = {{WBC-MJLab: Unified Whole-Body Motion Tracking on mjlab}},
        url = {https://github.com/wbc-mjlab/wbc-mjlab},
        year = {2026}
    }

Also cite **mjlab** — WBC-MJLab is an extension of the simulation and RL stack:

.. code-block:: bibtex

    @article{Zakka_mjlab_A_Lightweight_2026,
        author = {Zakka, Kevin and Liao, Qiayuan and Yi, Brent and Le Lay, Louis and Sreenath, Koushil and Abbeel, Pieter},
        title = {{mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning}},
        url = {https://arxiv.org/abs/2601.22074},
        year = {2026}
    }

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

.. code-block:: bibtex

    @software{hybridrobotics_whole_body_tracking2025,
        author = {{Hybrid Robotics}},
        title = {{whole\_body\_tracking: BeyondMimic Motion Tracking (Isaac Lab)}},
        url = {https://github.com/HybridRobotics/whole_body_tracking},
        year = {2025}
    }

    @software{nvlabs_groot_wbc2026,
        author = {{NVIDIA}},
        title = {{GR00T Whole-Body Control}},
        url = {https://github.com/NVlabs/GR00T-WholeBodyControl},
        year = {2026}
    }

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

.. code-block:: bibtex

    @article{zest2026,
        title = {Zero-shot Embodied Skill Transfer},
        journal = {arXiv preprint arXiv:2602.00401},
        year = {2026},
        url = {https://arxiv.org/abs/2602.00401}
    }

    @article{beyondmimic2025,
        author = {Liao, Qiayuan and Truong, Takara E. and Huang, Xiaoyu and Gao, Yuman and Tevet, Guy and Sreenath, Koushil and Liu, C. Karen},
        title = {{BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion}},
        journal = {arXiv preprint arXiv:2508.08241},
        year = {2025},
        url = {https://arxiv.org/abs/2508.08241}
    }

    @article{sonic2025,
        title = {SONIC: Supersized Motion Tracking},
        journal = {arXiv preprint arXiv:2511.07820},
        year = {2025},
        url = {https://arxiv.org/abs/2511.07820}
    }

    @article{omnixtreme2026,
        title = {OmniXtreme: High-Dynamic Scalable Humanoid Motion Tracking},
        journal = {arXiv preprint arXiv:2602.23843},
        year = {2026},
        url = {https://arxiv.org/abs/2602.23843}
    }

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
``.gitignore``). Canonical BibTeX: ``references/references.bib`` in the
`wbc org workspace <https://github.com/wbc-mjlab/wbc-mjlab>`_ (synced to
``docs/source/_static/refs.bib``). Citation metadata: ``CITATION.cff`` at the
repo root. arXiv submission tracked in
`issue #31 <https://github.com/wbc-mjlab/wbc-mjlab/issues/31>`_.
