"""Import static world geometry from custom dataset MuJoCo scenes."""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import mujoco
import numpy as np
from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

from wbc_mjlab.terrains.bounds import geometry_min_z, trajectory_min_z
from wbc_mjlab.terrains.progress import terrain_load_progress

DATASET_FLOOR_NAME = "dataset_floor"
_IMPORT_COUNTS: WeakKeyDictionary[mujoco.MjSpec, int] = WeakKeyDictionary()


@dataclass(kw_only=True)
class XmlTerrainCfg(SubTerrainCfg):
  """Static world geoms from a scene XML, excluding its articulated robot.

  XML coordinates remain relative to the tile center. References must keep their
  recorded coordinates, with only the environment origin added (no per-reset
  centering, mirroring, or height debias). Infinite planes become bounded boxes.
  Static body/frame hierarchies are flattened while preserving world transforms.
  Missing or unset XML paths produce a flat tile, preserving the trajectory ID.
  """

  xml_path: str | None = None
  trajectory_name: str
  motion_path: str = ""
  ground_thickness: float = 0.1
  progress_total: int = 0
  floor_clearance: float | None = 0.01
  """Raise the whole scene and recorded root path above z=0; None disables it."""
  grid_index: int | None = None
  grid_count: int = 0
  """Physical grid placement; logical generator columns still identify clips."""
  yaw: float = 0.0
  """Rotation about the tile origin, shared with the paired motion reference."""

  def terrain_xml(self) -> str:
    """Build an isolated XML with absolute asset paths; never alter the dataset."""
    if not self.xml_path:
      raise ValueError("No XML scene configured; use function() for flat terrain")
    path = Path(self.xml_path).expanduser().resolve()
    source = ET.parse(path).getroot()
    if source.find(".//include") is not None:
      raise ValueError(f"Scene includes must be expanded before import: {path}")
    world = source.find("worldbody")
    if world is None:
      raise ValueError(f"Missing worldbody in {path}")

    def static_tree(element: ET.Element) -> ET.Element | None:
      if element.tag == "body":
        if element.find("joint") is not None or element.find("freejoint") is not None:
          return None
        if element.get("mocap", "false").lower() == "true":
          raise ValueError(f"Mocap bodies are not static terrain: {path}")
      result = ET.Element(element.tag, element.attrib)
      for child in element:
        if child.tag in ("geom", "body", "frame"):
          retained = static_tree(child)
          if retained is not None:
            result.append(retained)
      return result

    target = static_tree(world)
    assert target is not None
    geoms = list(target.iter("geom"))
    if not geoms:
      raise ValueError(f"No static world geometry in {path}")
    if self.ground_thickness <= 0 or min(self.size) <= 0:
      raise ValueError("Tile size and ground thickness must be positive")

    result = ET.Element("mujoco", model="dataset_terrain")
    compiler = copy.deepcopy(source.find("compiler"))
    if compiler is None:
      compiler = ET.Element("compiler")
    # Files are resolved below, independently of the target scene's compiler.
    assetdir = compiler.get("assetdir", "")
    meshdir = compiler.get("meshdir", assetdir)
    texturedir = compiler.get("texturedir", assetdir)
    for key in ("assetdir", "meshdir", "texturedir"):
      compiler.attrib.pop(key, None)
    result.append(compiler)
    for default in source.findall("default"):
      result.append(copy.deepcopy(default))
    assets = ET.SubElement(result, "asset")
    meshes = {geom.get("mesh") for geom in geoms if geom.get("mesh")}
    fields = {geom.get("hfield") for geom in geoms if geom.get("hfield")}
    for asset in source.findall("asset/*"):
      if asset.tag == "mesh" and asset.get("name") not in meshes:
        continue
      if asset.tag == "hfield" and asset.get("name") not in fields:
        continue
      item = copy.deepcopy(asset)
      if item.get("file"):
        directory = texturedir if item.tag == "texture" else meshdir
        if item.tag == "hfield":
          directory = assetdir
        item.set("file", str((path.parent / directory / item.get("file")).resolve()))
      assets.append(item)

    result.append(target)
    for index, geom in enumerate(geoms):
      if not geom.get("name"):
        geom.set("name", f"geom_{index}")
      pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
      if pos.shape != (3,):
        raise ValueError(f"Invalid geom position in {path}")
      if geom.get("type") == "plane":
        if any(
          key in geom.attrib
          for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis")
        ):
          raise ValueError(f"Oriented planes are not supported: {path}")
        geom.set("type", "box")
        geom.set(
          "size", f"{self.size[0] / 2} {self.size[1] / 2} {self.ground_thickness / 2}"
        )
        pos[:2] = 0
        pos[2] -= self.ground_thickness / 2
      if "fromto" not in geom.attrib:
        geom.set("pos", " ".join(map(str, pos)))
      geom.set("group", "0")
      geom.set("mass", "0")
    return ET.tostring(result, encoding="unicode")

  def _flatten_geometry(self, source: mujoco.MjSpec) -> None:
    """Resolve spec poses without compiling a model or evaluating kinematics.

    TerrainGenerator translates each returned geom.pos in world axes. All geoms
    must therefore share an identity frame under the terrain body, including
    geoms originally attached to rotated bodies or frames.
    """

    def rotation(element) -> np.ndarray:
      if element.alt.type == mujoco.mjtOrientation.mjORIENTATION_QUAT:
        quat = np.array(element.quat)
      else:
        quat = np.array(
          source.resolve_orientation(
            source.compiler.degree, source.compiler.eulerseq, element.alt
          )
        )
      norm = np.linalg.norm(quat)
      if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid terrain orientation")
      return quat / norm

    def rotate(quat, vector):
      result = np.empty(3)
      mujoco.mju_rotVecQuat(result, np.asarray(vector, dtype=float), quat)
      return result

    def compose(left, right):
      result = np.empty(4)
      mujoco.mju_mulQuat(result, left, right)
      return result

    def parent_pose(element):
      parent = element.frame if element.frame is not None else element.parent
      return (
        pose(parent)
        if parent is not None
        else (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
      )

    def pose(element):
      # MuJoCo's world body has no valid parent; do not query its parent binding.
      if isinstance(element, mujoco.MjsBody) and element.name == source.worldbody.name:
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
      pos, quat = parent_pose(element)
      return pos + rotate(quat, element.pos), compose(quat, rotation(element))

    center = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
    transforms = {}
    for geom in source.geoms:
      if np.isfinite(geom.fromto[0]):
        if geom.type not in (
          mujoco.mjtGeom.mjGEOM_CAPSULE,
          mujoco.mjtGeom.mjGEOM_CYLINDER,
        ):
          raise ValueError("Terrain fromto requires a capsule or cylinder")
        start, end = np.asarray(geom.fromto).reshape(2, 3)
        direction = end - start
        if np.linalg.norm(direction) == 0:
          raise ValueError("Terrain fromto endpoints must differ")
        parent_pos, parent_quat = parent_pose(geom)
        pos = parent_pos + rotate(parent_quat, (start + end) / 2)
        quat = np.empty(4)
        mujoco.mju_quatZ2Vec(quat, direction)
        quat = compose(parent_quat, quat)
        geom.size[1] = np.linalg.norm(direction) / 2
        geom.fromto[0] = np.nan
      else:
        pos, quat = pose(geom)
      transforms[geom.name] = (pos + center, quat)

    for body in reversed(list(source.bodies)[1:]):
      body.to_frame()
    identity = source.worldbody.add_frame()
    for geom in source.geoms:
      geom.set_frame(identity)
      geom.pos, geom.quat = transforms[geom.name]
      geom.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    with terrain_load_progress(
      spec, self.progress_total, self.trajectory_name, self.xml_path
    ) as stage:
      output = self._generate(difficulty, spec, rng, stage)
      self._rotate_output(output)
      if self.grid_index is not None:
        self._place_on_grid(output)
      return output

  def _rotate_output(self, output: TerrainOutput) -> None:
    if not np.isfinite(self.yaw):
      raise ValueError("Terrain yaw must be finite")
    if self.yaw == 0.0:
      return
    quat = np.array([math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)])
    for item in output.geometries:
      geom = item.geom
      position = np.empty(3)
      mujoco.mju_rotVecQuat(position, geom.pos - output.origin, quat)
      geom.pos = output.origin + position
      orientation = np.empty(4)
      mujoco.mju_mulQuat(orientation, quat, geom.quat)
      geom.quat = orientation

  def _place_on_grid(self, output: TerrainOutput) -> None:
    """Compensate for mjlab's logical 1xN layout without changing clip indices."""
    if not 0 <= self.grid_index < self.grid_count:
      raise ValueError("Terrain grid index must lie within grid_count")
    columns = math.isqrt(self.grid_count - 1) + 1
    rows = (self.grid_count + columns - 1) // columns
    row, column = divmod(self.grid_index, columns)
    sx, sy = self.size
    logical_corner = np.array(
      [-sx / 2, (self.grid_index - self.grid_count / 2) * sy, 0.0]
    )
    grid_corner = np.array([(row - rows / 2) * sx, (column - columns / 2) * sy, 0.0])
    offset = grid_corner - logical_corner
    for item in output.geometries:
      item.geom.pos += offset
    output.origin += offset

  def _generate(self, difficulty, spec, rng, stage) -> TerrainOutput:
    if self.flat_patch_sampling:
      raise ValueError("XML terrain does not support flat patch sampling")
    terrain = spec.body("terrain")
    if terrain is None:
      raise ValueError("Target spec must contain the terrain body")
    # One infinite world floor, separate from the translated per-clip geometry.
    # Do not include it in TerrainOutput: TerrainGenerator translates that list.
    if spec.geom(DATASET_FLOOR_NAME) is None:
      terrain.add_geom(
        name=DATASET_FLOOR_NAME,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.1),
        group=0,
        contype=1,
        conaffinity=1,
        mass=0.0,
        friction=(0.8, 0.005, 0.0001),
        rgba=(0.3, 0.35, 0.4, 1.0),
      )
    if not self.xml_path or not Path(self.xml_path).expanduser().exists():
      stage("flat fallback")
      output = BoxFlatTerrainCfg(size=self.size).function(difficulty, spec, rng)
      self._raise_above_floor(output, spec, stage)
      return output
    stage("read XML")
    xml = self.terrain_xml()
    stage("parse spec")
    source = mujoco.MjSpec.from_string(xml)
    stage("resolve transforms")
    self._flatten_geometry(source)
    output = TerrainOutput(
      origin=np.array([self.size[0] / 2, self.size[1] / 2, 0.0]),
      geometries=[TerrainGeometry(geom=geom) for geom in source.geoms],
    )
    self._raise_above_floor(output, source, stage)
    # Namespace allocation must not materialize every geom already in the model.
    import_index = _IMPORT_COUNTS.get(spec, 0)
    _IMPORT_COUNTS[spec] = import_index + 1
    prefix = f"dataset_{import_index}_"
    geometries = output.geometries
    attached_names = [prefix + item.geom.name for item in geometries]
    frame = terrain.add_frame()
    stage(f"attach {len(geometries)} geoms, {len(source.meshes)} meshes")
    spec.attach(source, prefix=prefix, frame=frame)
    # Default attach moves the source elements, so these references now point
    # into the target spec. With copy_during_attach=True, rebind by known names.
    for item, name in zip(geometries, attached_names, strict=True):
      if item.geom.name != name:
        item.geom = spec.geom(name)
        if item.geom is None:
          raise ValueError(f"Attached terrain geometry not found: {name}")
    return TerrainOutput(
      origin=output.origin,
      geometries=geometries,
    )

  def _raise_above_floor(self, output: TerrainOutput, spec, stage) -> None:
    if self.floor_clearance is None:
      return
    if not np.isfinite(self.floor_clearance) or self.floor_clearance < 0:
      raise ValueError("floor_clearance must be finite and nonnegative")
    stage("compute vertical bounds")
    minimum = min(geometry_min_z(item.geom, spec) for item in output.geometries)
    minimum = min(minimum, trajectory_min_z(self.motion_path))
    if not np.isfinite(minimum):
      raise ValueError("Terrain vertical bounds must be finite")
    shift = max(0.0, self.floor_clearance - minimum)
    for item in output.geometries:
      item.geom.pos[2] += shift
    # MotionCommand adds this same origin to every reference position at reset.
    output.origin[2] += shift
    stage(f"vertical shift +{shift:.3f} m")
