.. _concepts:

Overview
========

How wbc-mjlab is put together — the ideas behind the commands. Read these when you want to
change the stack rather than just run it.

The one-paragraph version
-------------------------

One **shared MDP** (motion command, rewards, observations, terminations) is written once and
knows nothing about any robot or paper. A **robot** plugs in its model, actuators and body
names. A **preset** configures the shared terms the way a method needs (weights, sampling,
observation layout). A **task** names one robot + preset stack so the CLI can train it.
Training samples clips from a **motion library**, starting episodes where the policy fails
most (**adaptive sampling**).

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`stack` Modularity
      :link: modularity
      :link-type: doc

      The four layers and what each one owns.

   .. grid-item-card:: :octicon:`workflow` Architecture
      :link: ../architecture
      :link-type: doc

      How a ``--task`` becomes an environment config; how the Studio is built.

   .. grid-item-card:: :octicon:`tools` Presets and tasks
      :link: presets_and_tasks
      :link-type: doc

      Methods as config mutators; composing them; Studio presets.

   .. grid-item-card:: :octicon:`cpu` Robots and extensions
      :link: robots_and_extensions
      :link-type: doc

      Why robots register, and the ways to add one.

   .. grid-item-card:: :octicon:`graph` Adaptive sampling
      :link: adaptive_sampling
      :link-type: doc

      Where episodes start in a clip library, and why it matters.

   .. grid-item-card:: :octicon:`eye` Observations
      :link: ../mdp/observations
      :link-type: doc

      Modular reference terms — why papers' actor layouts are just term sets.

.. toctree::
   :hidden:

   modularity
   ../architecture
   presets_and_tasks
   robots_and_extensions
   adaptive_sampling
