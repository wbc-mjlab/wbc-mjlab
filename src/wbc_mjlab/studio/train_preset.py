"""``wbc-mjlab-train`` for a Studio preset: registers ``<base>--<name>`` in this process first.

Usage (what the Studio launches)::

  python -m wbc_mjlab.studio.train_preset --preset <name> --dataset samples [train args…]

The resolved preset (modules inlined) is stored as ``params/studio_preset.yaml`` in the run.
Single-GPU only: multi-GPU workers (torchrunx) would not see the preset task.
"""

from __future__ import annotations

import sys


def main() -> None:
  argv = sys.argv[1:]
  if "--preset" not in argv or argv.index("--preset") + 1 >= len(argv):
    raise SystemExit(
      "usage: python -m wbc_mjlab.studio.train_preset --preset <name> [train args]"
    )
  i = argv.index("--preset")
  name = argv[i + 1]
  rest = argv[:i] + argv[i + 2 :]

  from pathlib import Path

  import mjlab  # noqa: F401 — entry points before wbc_mjlab task modules
  import mjlab.utils.os as mjlab_os
  import yaml

  from wbc_mjlab.studio import presets

  snapshot = presets.resolve(name)
  task_id = presets.register(name, snapshot=snapshot)
  print(f"[studio] preset {name!r} → task {task_id}", flush=True)
  sys.argv = [sys.argv[0], "--task", task_id, *rest]
  from wbc_mjlab.scripts.train import main as train_main

  # next to params/env.yaml, keep the preset exactly as trained (Play rebuilds from it)
  dump = mjlab_os.dump_yaml

  def dump_with_snapshot(filename, data, *args, **kwargs):
    dump(filename, data, *args, **kwargs)
    path = Path(filename)
    if path.name == "env.yaml" and path.parent.name == "params":
      (path.parent / presets.SNAPSHOT).write_text(
        yaml.safe_dump(snapshot, sort_keys=False)
      )

  mjlab_os.dump_yaml = dump_with_snapshot
  if "mjlab.scripts.train" in sys.modules:  # bound by name there
    sys.modules["mjlab.scripts.train"].dump_yaml = dump_with_snapshot
  train_main()


if __name__ == "__main__":
  main()
