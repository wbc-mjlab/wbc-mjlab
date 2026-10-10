Wbc-G1-Terrain-Scan — rough terrain
===================================

Task id: ``Wbc-G1-Terrain-Scan`` · In-tree reference on robot ``g1``.

**Perceptive tracking on procedural terrain** — same WBC rewards as
:doc:`wbc-g1`, on a curriculum terrain generator with a base-mounted height scan
in the actor.

- **Presets:** ``apply_wbc`` + ``apply_terrain_scan``
- **Builder:** ``g1_wbc_terrain_scan_env_cfg()``

What this task adds
-------------------

On top of :doc:`wbc-g1`, ``apply_terrain_scan`` (``presets/terrain.py``):

**Scene:** 10-row curriculum generator (flat, uniform noise, Perlin hills, random
grid); all envs start at row 0. Larger ``nconmax`` / ``njmax`` (this task only).

**Sensors:** ``height_scan`` — 9 x 6 grid (1.6 x 1.0 m, 0.2 m), yaw-aligned under
the pelvis; ``keybody_clearance`` — one downward ray per anchor / EE body.

**Actor + critic:** ``height_scan`` = ``clip(pelvis_z - terrain_z - 0.78, ±1)``
(54 dims, ±5 cm noise on the actor).

**Reference:** ``MotionCommandCfg.ground_reference=True`` recenters each episode's
start-frame anchor XY onto the sub-terrain origin.

**Terminations:** ``anchor_pos`` / ``ee_body_pos`` compare *height above terrain*
(``bad_anchor_pos_z_terrain``, ``bad_motion_body_pos_z_terrain``) with the flat
task thresholds, instead of world z. Tracking rewards stay world-frame, so
flat-ground mocap carries a small height bias on rough rows.

**Curriculum:** ``terrain_levels_motion`` — time-out → harder row, early
termination → easier row. Disabled in play, where envs spread over all rows.

Deploy
------

The scan layout (frame body, alignment, size, resolution, ``grid_shape_yx``,
offset, clip) is written under ``actor_observations.height_scan.params`` in the
exported tracking params, so a runtime can sample the same grid from an
elevation map. Rays are x-major within each y row.

Train & play
------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Terrain-Scan --dataset samples
   uv run wbc-mjlab-play --task Wbc-G1-Terrain-Scan --dataset samples --viewer viser

Logs: ``logs/rsl_rl/wbc_g1_terrain_scan/<run>/``
