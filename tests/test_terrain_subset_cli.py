"""Small metadata-only checks for CLI trajectory selection."""

import json
from types import SimpleNamespace

import pytest

from wbc_mjlab.scripts.terrain_subset import (
  apply_training_trajectory_limit,
  apply_trajectory_limit,
  configure_terrain_selection,
  pop_exclude_sources,
  pop_num_trajectories,
)


@pytest.mark.parametrize(
  "flags", [["--num-trajectories", "2"], ["--num-trajectories=2"]]
)
def test_parse_trajectory_limit(flags):
  assert pop_num_trajectories(["--agent", "zero", *flags]) == (["--agent", "zero"], 2)


@pytest.mark.parametrize(
  "flags",
  [
    ["--num-trajectories"],
    ["--num-trajectories=0"],
    ["--num-trajectories=-1"],
    ["--num-trajectories=no"],
    ["--num-trajectories=2", "--num-trajectories=3"],
  ],
)
def test_invalid_trajectory_limit(flags):
  with pytest.raises(ValueError, match="--num-trajectories"):
    pop_num_trajectories(flags)


def test_subset_uses_same_sorted_ids_for_both_entry_points(tmp_path, monkeypatch):
  from mjlab.tasks import registry

  records = [dict(id=name, motion=f"{name}.csv", fps=30) for name in ("c", "a", "b")]
  (tmp_path / "manifest.json").write_text(json.dumps(dict(clips=records)))

  def config():
    return SimpleNamespace(
      commands={"motion": SimpleNamespace(dataset_terrain=True, motion_file="unused")},
      scene=SimpleNamespace(
        terrain=SimpleNamespace(
          terrain_generator=SimpleNamespace(size=(12.0, 8.0)),
        )
      ),
    )

  playing = config()
  apply_trajectory_limit(playing, 2, motion_file=str(tmp_path))
  training = config()
  monkeypatch.setattr(
    registry, "_REGISTRY", {"task": SimpleNamespace(env_cfg=training)}
  )
  apply_training_trajectory_limit(
    "task",
    [
      "--env.commands.motion.motion-file",
      str(tmp_path),
    ],
    2,
  )
  for cfg in (playing, training):
    generator = cfg.scene.terrain.terrain_generator
    assert list(generator.sub_terrains) == ["a", "b"]
    assert generator.size == (12.0, 8.0)
    assert [s.trajectory_name for s in generator.sub_terrains.values()] == ["a", "b"]
    assert all(s.progress_total == 2 for s in generator.sub_terrains.values())
    assert cfg.commands["motion"].motion_file == str(tmp_path)


def test_other_tasks_reject_trajectory_limit():
  with pytest.raises(ValueError, match="dataset-terrain task"):
    apply_trajectory_limit(SimpleNamespace(commands={}), 2)


def test_dataset_switch_rebuilds_terrain_without_subset_flags(tmp_path, monkeypatch):
  from mjlab.tasks import registry

  from wbc_mjlab.presets.terrain_dataset import terrain_from_dataset_cfg

  original, selected = tmp_path / "custom", tmp_path / "custom_cut"
  for root, names in ((original, ["same"]), (selected, ["same", "new"])):
    root.mkdir()
    records = [
      dict(id=name, motion=f"{name}.csv", scene=f"{name}.xml", fps=30) for name in names
    ]
    (root / "manifest.json").write_text(json.dumps(dict(clips=records)))

  def config():
    return SimpleNamespace(
      commands={
        "motion": SimpleNamespace(dataset_terrain=True, motion_file=str(original))
      },
      scene=SimpleNamespace(
        terrain=SimpleNamespace(
          terrain_generator=terrain_from_dataset_cfg(str(original), size=(12.0, 8.0)),
        )
      ),
    )

  playing, training = config(), config()
  playing.commands["motion"].motion_file = str(selected)
  configure_terrain_selection(playing)
  monkeypatch.setattr(
    registry, "_REGISTRY", {"task": SimpleNamespace(env_cfg=training)}
  )
  apply_training_trajectory_limit(
    "task", ["--env.commands.motion.motion-file", str(selected)], None
  )
  for cfg in (playing, training):
    generator = cfg.scene.terrain.terrain_generator
    assert list(generator.sub_terrains) == ["new", "same"]
    assert generator.size == (12.0, 8.0)
    assert cfg.commands["motion"].motion_file == str(selected)
    for name, patch in generator.sub_terrains.items():
      assert patch.xml_path == str(selected / f"{name}.xml")
      assert patch.motion_path == str(selected / f"{name}.csv")
    # An unchanged dataset retains its existing selection and custom settings.
    configure_terrain_selection(cfg)
    assert cfg.scene.terrain.terrain_generator is generator


def test_terrain_sync_leaves_other_tasks_unchanged():
  configure_terrain_selection(SimpleNamespace(commands={}))


@pytest.mark.parametrize(
  "flags",
  [
    ["--exclude-sources", "ego", "grail"],
    ["--exclude-sources=ego,grail"],
    ["--exclude-sources", "grail", "--exclude-sources", "ego"],
  ],
)
def test_parse_source_exclusions(flags):
  assert pop_exclude_sources([*flags, "--agent", "zero"]) == (
    ["--agent", "zero"],
    ("ego", "grail"),
  )


@pytest.mark.parametrize(
  "flags",
  [
    ["--exclude-sources"],
    ["--exclude-sources="],
    ["--exclude-sources", "typo"],
  ],
)
def test_invalid_source_exclusions(flags):
  with pytest.raises(ValueError, match="--exclude-sources"):
    pop_exclude_sources(flags)


@pytest.mark.parametrize(
  "excluded,limit,expected",
  [
    (("ego",), 2, ["grail/a", "lafan/a"]),
    (("ego", "grail"), None, ["lafan/a", "lafan/b"]),
  ],
)
def test_exclusions_precede_limit_for_train_and_play(
  tmp_path, monkeypatch, excluded, limit, expected
):
  from mjlab.tasks import registry

  records = [
    dict(id=name, motion=f"{name}.csv", fps=30)
    for name in ("ego/a", "ego/b", "grail/a", "lafan/a", "lafan/b")
  ]
  records[0]["source"] = "ego"
  (tmp_path / "manifest.json").write_text(json.dumps(dict(clips=records)))

  def config():
    return SimpleNamespace(
      commands={
        "motion": SimpleNamespace(dataset_terrain=True, motion_file=str(tmp_path))
      },
      scene=SimpleNamespace(
        terrain=SimpleNamespace(
          terrain_generator=SimpleNamespace(size=(12.0, 8.0)),
        )
      ),
    )

  playing, training = config(), config()
  apply_trajectory_limit(playing, limit, exclude_sources=excluded)
  monkeypatch.setattr(
    registry, "_REGISTRY", {"task": SimpleNamespace(env_cfg=training)}
  )
  apply_training_trajectory_limit("task", [], limit, exclude_sources=excluded)
  for cfg in (playing, training):
    patches = cfg.scene.terrain.terrain_generator.sub_terrains
    assert list(patches) == expected
    assert [patch.trajectory_name for patch in patches.values()] == expected
  original = playing.scene.terrain.terrain_generator
  with pytest.raises(ValueError, match="No trajectories remain"):
    apply_trajectory_limit(playing, None, exclude_sources=("ego", "grail", "lafan"))
  assert playing.scene.terrain.terrain_generator is original
