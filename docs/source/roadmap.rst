.. _roadmap:

Roadmap
=======

**Backlog:** `GitHub Issues <https://github.com/wbc-mjlab/wbc-mjlab/issues>`_ — one issue ≈ one PR
(``Closes #N`` in commit messages).

WBC-MJLab is an open, modular stack for whole-body motion tracking on mjlab. We welcome
**contributions, bug reports, and community feedback** — whether that is a new robot
extension, a paper preset, deploy/runtime work, or documentation fixes. If something is
missing or unclear, `open an issue <https://github.com/wbc-mjlab/wbc-mjlab/issues/new>`_
or see :doc:`contributing`.

Open — methods & training
-------------------------

- ☐ BeyondMimic as default baseline stack — `#24 <https://github.com/wbc-mjlab/wbc-mjlab/issues/24>`_
- ☐ Remaining BeyondMimic gaps beyond ``Wbc-G1-BinaryFailure``

Open — sim → real & demo
------------------------

- ☐ **Modular deploy + web demo runtimes** — robot-agnostic consumers of WBC-MJLab
  artifacts (``policy.onnx`` + ``config.yaml`` + clips); not G1-only; easy path for
  another Unitree robot (e.g. H2) — `#38 <https://github.com/wbc-mjlab/wbc-mjlab/issues/38>`_
- ☐ **Unify reference format** between deploy and browser demo (shared schema, one
  observation pipeline) — `#29 <https://github.com/wbc-mjlab/wbc-mjlab/issues/29>`_

Open — data & robots
--------------------

- ☐ **Retargeting quality** — fix floating-above-ground references (H2 first); improves
  training and deploy reference quality — `#39 <https://github.com/wbc-mjlab/wbc-mjlab/issues/39>`_
- ☐ More robots via extension packages (beyond in-tree G1 + reference H2)

Open — developer experience
---------------------------

- ☐ Technical report + ``CITATION.cff`` — `#31 <https://github.com/wbc-mjlab/wbc-mjlab/issues/31>`_
- ☐ CI (ruff + ``wbc-mjlab-list-envs`` smoke)
- ☐ Dockerfile (CUDA + uv + ``MUJOCO_GL=egl``)
- ☐ Unit tests beyond import smoke

Shipped
-------

- ☑ PyPI package `wbc-mjlab <https://pypi.org/project/wbc-mjlab/>`_
- ☑ Sphinx docs + GitHub Pages (`docs site <https://wbc-mjlab.github.io/wbc-mjlab/>`_)
- ☑ Extension API (``register_wbc_extension``) + autodoc API reference

Related repos (not tracked in this issue list): `wbc-g1-deploy
<https://github.com/wbc-mjlab/wbc-g1-deploy>`_, `wbc-demo
<https://github.com/wbc-mjlab/wbc-demo>`_, `wbc-mjlab-extension-h2
<https://github.com/wbc-mjlab/wbc-mjlab-extension-h2>`_.
