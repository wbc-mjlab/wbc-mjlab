:html_theme.sidebar_secondary.remove:

WBC-MJLab
=========

.. container:: wbc-hero

   .. container:: wbc-lead

      Whole-body motion tracking for humanoids on `mjlab <https://github.com/mujocolab/mjlab>`_.
      Train **one policy on a whole motion library**, compare methods as **tasks on one shared
      MDP**, and deploy the result to a real robot.

   .. button-ref:: source/workflows/quickstart
      :ref-type: doc
      :color: primary
      :class: wbc-btn

      Get started

   .. button-link:: https://wbc-mjlab.github.io/wbc-demo/
      :color: primary
      :outline:
      :class: wbc-btn

      :octicon:`play` Live demo

   .. button-ref:: source/studio
      :ref-type: doc
      :color: primary
      :outline:
      :class: wbc-btn

      WBC Studio

.. figure:: ../assets/wbc_g1_collage.gif
   :width: 100%
   :alt: A Unitree G1 policy tracking walking, running, dancing, fighting and get-up clips
   :class: wbc-hero-figure

Choose your path
----------------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket` Get started
      :link: source/workflows/quickstart
      :link-type: doc

      Install, then train and watch a first policy in about 15 minutes.

   .. grid-item-card:: :octicon:`checklist` How-to guides
      :link: source/recipes
      :link-type: doc

      Prepare data, pick a task, train, evaluate, deploy, add a robot.

   .. grid-item-card:: :octicon:`light-bulb` Concepts
      :link: source/concepts/index
      :link-type: doc

      The shared MDP, presets and tasks, robots, adaptive sampling.

   .. grid-item-card:: :octicon:`book` Reference
      :link: source/usage
      :link-type: doc

      Commands, file formats, the Studio API, MDP terms, Python API.

The pipeline
------------

.. grid:: 1 2 3 5
   :gutter: 2
   :class-container: wbc-pipeline

   .. grid-item-card:: 1 · Robot
      :link: source/extensions/index
      :link-type: doc

      G1 built in — or add yours from its MJCF / URDF.

   .. grid-item-card:: 2 · Data
      :link: source/data
      :link-type: doc

      Check retargeted clips, convert them to a motion library.

   .. grid-item-card:: 3 · Task
      :link: source/tasks/index
      :link-type: doc

      Pick a method's setup, or tune one without code.

   .. grid-item-card:: 4 · Train
      :link: source/workflows/training
      :link-type: doc

      Thousands of envs on GPU; resume, many GPUs.

   .. grid-item-card:: 5 · Deploy
      :link: source/workflows/deploy
      :link-type: doc

      ``policy.onnx`` + ``config.yaml`` for the robot runtime.

Run it from the command line, or in a browser with **WBC Studio** — locally or on a remote GPU
server.

.. tab-set::

   .. tab-item:: Command line

      .. code-block:: bash

         git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab && make sync
         uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 8
         uv run wbc-mjlab-train --task Wbc-G1 --dataset samples

   .. tab-item:: Studio

      .. code-block:: bash

         git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab && make sync
         uv run wbc-mjlab-studio      # open the printed link

Why WBC-MJLab
-------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`versions` Tasks, not forks
      :class-card: wbc-feature

      ZEST, BeyondMimic-style sampling, state estimation, end-effector command, rough
      terrain — ``--task`` switches on one codebase, with the same CLI and logs for fair
      comparison.

   .. grid-item-card:: :octicon:`stack` One shared MDP
      :class-card: wbc-feature

      Motion command, rewards, observations and terminations written once; robots and
      methods plug in as configuration.

   .. grid-item-card:: :octicon:`database` Multi-motion by design
      :class-card: wbc-feature

      Train on whole clip libraries with adaptive sampling; one controller for walking,
      running, fighting, get-ups and flips.

   .. grid-item-card:: :octicon:`cpu` Any humanoid
      :class-card: wbc-feature

      The G1 reference, extension packages (H2), or a robot added from its model files in
      minutes.

   .. grid-item-card:: :octicon:`browser` WBC Studio
      :class-card: wbc-feature

      Clip QA, conversion, a visual preset editor, training jobs and playback on many envs —
      in the browser, scriptable over HTTP.

   .. grid-item-card:: :octicon:`zap` Sim to real
      :class-card: wbc-feature

      Every run exports an ONNX policy and a robot-agnostic ``config.yaml`` for the deploy
      runtime.

Try it without installing
-------------------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`globe` Live web demo
      :link: https://wbc-mjlab.github.io/wbc-demo/
      :link-type: url

      A trained policy in the browser (MuJoCo WASM).

   .. grid-item-card:: :octicon:`code-square` Google Colab
      :link: https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb
      :link-type: url

      The bundled checkpoint and sample clips, no local GPU.

   .. grid-item-card:: :octicon:`device-desktop` Local demo
      :link: source/workflows/demo
      :link-type: doc

      ``uv run wbc-mjlab-demo`` after install.

Citing
------

If you use WBC-MJLab in your research, cite the software, **mjlab**, and the method papers of
the tasks you use (:doc:`source/research`):

.. literalinclude:: source/_static/refs.bib
   :language: bibtex
   :start-at: @software{wbc_mjlab2026,
   :end-before: @article{zakka2026mjlab,

WBC-MJLab is Apache-2.0 licensed. It builds on `mjlab <https://github.com/mujocolab/mjlab>`_
and draws on open whole-body tracking work such as
`whole_body_tracking <https://github.com/HybridRobotics/whole_body_tracking>`_ and
`GR00T-WholeBodyControl <https://github.com/NVlabs/GR00T-WholeBodyControl>`_. Several
contributors are affiliated with the `Institute of Artificial Intelligence
<https://iai.mipt.ru/en/about>`_, the Robotics Center SBER,
`Innopolis University <https://innopolis.university/en>`_ and
`KAIST <https://www.kaist.ac.kr/en/>`_; we thank them for hosting and hardware.

.. toctree::
   :hidden:
   :caption: Get started

   source/installation
   source/workflows/quickstart
   source/workflows/demo

.. toctree::
   :hidden:
   :caption: How-to guides

   source/recipes
   source/studio
   source/data
   source/tasks/index
   source/workflows/training
   source/visualization
   source/workflows/deploy
   source/extensions/index
   source/troubleshooting

.. toctree::
   :hidden:
   :caption: Concepts

   source/concepts/index

.. toctree::
   :hidden:
   :caption: Reference

   source/usage
   source/reference/configuration
   source/reference/studio_api
   source/reference/data_formats
   source/mdp/index
   source/api/index

.. toctree::
   :hidden:
   :caption: Project

   source/contributing
   source/roadmap
   source/research
