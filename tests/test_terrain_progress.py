"""Progress reporting without loading datasets or compiling models."""

import mujoco
import pytest

from wbc_mjlab.terrains.progress import terrain_load_progress


def test_progress_counts_scenes_and_restarts_for_next_build(capsys):
  spec = mujoco.MjSpec()
  for name in ("first", "second"):
    with terrain_load_progress(spec, 2, name, None) as stage:
      stage("flat fallback")
  output = capsys.readouterr().err
  assert "2/2 (100%)" in output
  assert "Terrain specs ready" in output
  with terrain_load_progress(spec, 1, "new-build", None):
    pass
  output = capsys.readouterr().err
  assert "1/1 (100%)" in output


def test_progress_reports_failed_scene_and_can_restart(capsys):
  spec = mujoco.MjSpec()
  with pytest.raises(ValueError, match="invalid scene"):
    with terrain_load_progress(spec, 2, "broken", "/scenes/broken.xml") as stage:
      stage("parse spec")
      raise ValueError("invalid scene")
  output = capsys.readouterr().err
  assert "parse spec" in output
  assert "[ERROR] Terrain import failed: broken | /scenes/broken.xml" in output
  assert "Terrain specs ready" not in output
  with terrain_load_progress(spec, 1, "retry", None):
    pass
  assert "1/1 (100%)" in capsys.readouterr().err
