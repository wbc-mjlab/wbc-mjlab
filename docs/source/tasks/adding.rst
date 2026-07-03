Add a task
==========

1. **Preset** — ``presets/<method>.py`` with ``def apply_<method>(cfg) -> None``
   (or compose existing ``apply_*`` functions). See :doc:`index` for ``apply_zest``
   walkthrough.
2. **Export** from ``presets/__init__.py``
3. **Task builder** in ``robots/<id>/tasks.py``; register ``WbcTaskConfig``
4. **Document** — add a page under :doc:`index`; cite the paper in :doc:`../research`
5. **Smoke test** — ``uv run wbc-mjlab-train --task Wbc-... --dataset samples``

For external robots, stack presets in an extension package — see :doc:`../extensions/extensions`.

MDP changes (new reward/RSI terms) belong in ``env/mdp/`` and require an issue first
— see :doc:`../contributing`.

Related: :doc:`../architecture`, :doc:`../mdp/index`, :doc:`../extensions/index`.
