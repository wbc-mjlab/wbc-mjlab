Wbc-G1-Terrain-Dataset
======================

Tracks custom dataset motions on their paired MuJoCo XML scenes. The manifest
must contain a ``clips`` list with unique ``id``, ``motion``, and
``fps`` fields. Motion and scene paths are relative to the dataset root.
``joint_names`` defines the CSV joint column order.
``scene`` is optional: an absent, null, or empty value, or a missing XML file,
automatically creates a flat tile with local surface z=0. The clip keeps its own terrain column,
trajectory ID, and original reference coordinates. Existing malformed XML files
and missing assets still raise errors rather than silently using flat ground.

``XmlTerrainCfg`` implements ``SubTerrainCfg.function``. It imports direct
``worldbody`` geometries and their mesh assets, excluding articulated bodies,
actuators, and keyframes. Static body/frame hierarchies are flattened into world
coordinates, preserving translations, rotations, and inherited geometry defaults.
Asset paths become absolute and imported names receive
unique prefixes. Infinite ground planes become bounded boxes with the same
surface height. Scene files and meshes are not modified.

Selected scenes are physically arranged on a centered, near-square grid with
``ceil(sqrt(N))`` columns. Tile spacing follows ``size`` in each axis; the last
row can be incomplete. Selection/exclusion rebuilds the layout for the remaining
clips. Internally, origins retain one logical row and one column per clip to
preserve trajectory-to-terrain indexing. On each reset,
the sampled trajectory ID selects its terrain column before robot placement.
Reference positions retain their recorded coordinates relative to the tile
origin. Mirroring, height debias, per-reset reference centering, and terrain
level curriculum are disabled. Height scans are enabled; pose terminations
compare world-space errors because the reference already includes terrain
elevation. Manifest FPS and joint order are used when loading source motions.

Each clip receives a reproducible random yaw in ``[-pi, pi)`` derived from its
ID. Filtering clips or excluding sources does not change the angle of a retained
clip. Geometry rotates about its tile origin; the reference uses the same yaw
for body positions, orientations, linear/angular velocities, and accelerations,
including lookahead windows and partial resets. Joint coordinates are unchanged.
The shared world floor and tile origins do not rotate. A tile size must cover
the rotated scene extents; arbitrary yaw can increase the required XY margin.

Vertical placement
------------------

A separate infinite collision floor is always added at world ``z=0``, whether
or not XML scenes contain ground geometry. It is shared by all clips, belongs
to the terrain body and sensor group 0, and is never included in per-tile
translation or vertical bounds. XML geometry and flat fallback tiles retain
their independent upward shifts above this floor.

Each terrain and its trajectory receive one shared upward translation. The
offset is ``max(0, floor_clearance - min(terrain_min_z, root_path_min_z))``, with
``floor_clearance=0.01`` meters by default. Geometry bounds include primitive
sizes, rotated/scaled mesh vertices, and heightfield base thickness. The minimum
recorded root height is read from the motion clip; it is not a bound on the full
articulated robot. No model compilation or simulation is used to compute bounds.

The offset is applied to every terrain geom and to ``TerrainOutput.origin.z``.
Motion references and reset poses inherit the same offset through the environment
origin, preserving their positions relative to obstacles. Flat fallback tiles
are shifted too, including their full solid thickness. Source XML, mesh, and
motion files remain unchanged. Set ``XmlTerrainCfg.floor_clearance=None`` to
disable automatic vertical placement.

Bounds computation reads mesh vertices and motion root heights at import time;
the progress display identifies this as ``compute vertical bounds`` and reports
the resulting vertical shift before geometry attachment.

Select a small subset explicitly when constructing the configuration:

For train or play, pass ``--num-trajectories N`` to select the first N manifest
IDs in sorted order before any terrain geometry or motion frames are loaded.
The same selection is used for motions and terrain tiles. No dataset files are
copied or regenerated. Without this option the full configured selection is used.
This flag currently applies only to dataset-terrain tasks.

Pass ``--exclude-sources ego grail`` to exclude source groups before selecting
the first N IDs. Supported groups are ``ego``, ``grail``, and ``lafan``. The
source is taken from the manifest's ``source`` field, or the first component of
the clip ID if that field is absent. Comma-separated values and repeated flags
are also accepted (``--exclude-sources=ego,grail``). The flag works for both
train and play, with or without ``--num-trajectories``. Excluded motion clips and
terrain tiles are never loaded. An empty remaining selection raises an error.

.. code-block:: bash

   uv run wbc-mjlab-play --task Wbc-G1-Terrain-Dataset --dataset custom \
       --exclude-sources ego grail --num-trajectories 5 \
       --agent zero --num-envs 1 --viewer native

Alternatively, select explicit IDs in Python:

.. code-block:: python

   from wbc_mjlab.robots.g1.tasks import g1_wbc_terrain_dataset_env_cfg

   cfg = g1_wbc_terrain_dataset_env_cfg(
       dataset_root="data/g1/custom",
       clip_ids=(
           "ego/terrain/exomc-parkour_xhall_02.npz_6f9369_scene_smplx",
           "ego/terrain/exomc-parkour_xhall_04.npz_de0619_scene_smplx",
       ),
       size=(30.0, 30.0),
   )

``clip_ids=None`` selects the whole manifest. Every selected scene contributes
geometry to the shared model, so select clips before constructing large models.
The loader uses the same selection as the terrain generator. Tile size must
cover both the full scene and the trajectory, with a margin; mesh extents are
not inferred automatically. Train/play wrappers automatically rebuild paired
terrain configuration when ``--dataset``, ``--dataset-path``, or ``--motion-file``
selects another dataset, including when no subset flags are specified. When
constructing environments directly in Python, use this helper to configure both.

Supported XML layout
--------------------

Static geometries may be direct children of ``worldbody`` or nested inside static
``body`` and ``frame`` elements. Articulated subtrees are excluded. XML includes,
explicitly oriented planes, mocap bodies, and dynamic object scenes are rejected
explicitly. Meshes referenced through inherited defaults are not
supported; mesh references must be explicit on each geometry.

Loading progress
----------------

Dataset terrain import automatically displays a terminal progress bar with
completed/total scenes, percentage, elapsed time, estimated remaining time,
current clip ID, and import stage (XML reading, spec parsing, transform
resolution, geometry attachment, or flat fallback). Errors report the clip ID
and XML path. Redirected progress output is throttled to one update per second.
The counter advances only after a scene has been imported successfully.

100% means that terrain specs have been imported. Native scene/model compilation
and simulation initialization follow; their internal progress is not measured
by this bar. No background monitoring is started.

Validation
----------

The two XML scenes in the example were parsed and attached to ``MjSpec`` without
model compilation: 330 and 350 geometries respectively, with no robot joints.
Synthetic tests cover bounded ground geometry, coordinates, namespace isolation,
manifest IDs/FPS/joint order, subset selection, and reset origin assignment.
Simulation and training have not been run.
