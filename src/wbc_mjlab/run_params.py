"""Read a training run's ``params/<name>.yaml`` (``env``, ``agent``), as written by mjlab, and
check that a task still builds the observation its policy was trained on."""

from __future__ import annotations

from pathlib import Path

import yaml


class _Loader(yaml.SafeLoader):
  pass


# mjlab writes tuples (and other python objects) with python/* tags: read them as plain data
_Loader.add_multi_constructor(
  "tag:yaml.org,2002:python/",
  lambda loader, _suffix, node: (
    loader.construct_sequence(node)
    if isinstance(node, yaml.SequenceNode)
    else loader.construct_mapping(node)
    if isinstance(node, yaml.MappingNode)
    else loader.construct_scalar(node)
  ),
)


def read_run_params(run_dir: Path | str, name: str) -> dict:
  """``params/<name>.yaml`` of a run; empty when missing or unreadable."""
  path = Path(run_dir) / "params" / f"{name}.yaml"
  if not path.is_file():
    return {}
  try:
    doc = yaml.load(path.read_text(errors="replace"), Loader=_Loader)  # noqa: S506
  except yaml.YAMLError:
    return {}
  return doc if isinstance(doc, dict) else {}


# Term parameters that change what a term's values mean or where they sit.
_LAYOUT_PARAMS = ("frames", "stride", "num_frames", "body_names")


def _layout(group: dict) -> list[tuple]:
  hist = group.get("history_length")
  out = []
  for name, term in (group.get("terms") or {}).items():
    if term is None:
      continue
    params = term.get("params") or {}
    shape = tuple(
      (k, tuple(v) if isinstance(v, (list, tuple)) else v)
      for k, v in sorted(params.items())
      if k in _LAYOUT_PARAMS
    )
    out.append(
      (name, shape, int(hist if hist is not None else term.get("history_length") or 0))
    )
  return out


def actor_layout_change(run_dir: Path | str, env_cfg) -> str:
  """What differs between the actor observation run ``run_dir`` was trained on
  (``params/env.yaml``) and the one ``env_cfg`` builds: term order, frames ahead, history
  ("" when they match or the run has no record). A policy fed another layout of the same
  size would load and act on scrambled inputs."""
  import dataclasses

  trained = read_run_params(run_dir, "env").get("observations", {}).get("actor")
  if not trained:
    return ""
  group = env_cfg.observations["actor"]
  now = {
    "history_length": group.history_length,
    "terms": {
      n: {"params": t.params, "history_length": t.history_length}
      for n, t in group.terms.items()
      if dataclasses.is_dataclass(t)
    },
  }
  a, b = _layout(trained), _layout(now)
  if a == b:
    return ""
  if [x[0] for x in a] != [x[0] for x in b]:
    return (
      f"trained with actor terms {[x[0] for x in a]}, the task now builds "
      f"{[x[0] for x in b]}"
    )
  diff = [f"{x[0]}: {x[1:]} → {y[1:]}" for x, y in zip(a, b, strict=True) if x != y]
  return "the actor's terms changed since training: " + "; ".join(diff)
