"""Robot id resolution for CLI ``--robot`` and ``data/<robot>/`` paths."""

from __future__ import annotations

RobotId = str

_BUILTIN_ROBOT_IDS: tuple[str, ...] = ("g1",)
_KNOWN_ROBOT_IDS: list[str] = list(_BUILTIN_ROBOT_IDS)

ROBOT_ALIASES: dict[str, str] = {
  "unitree_g1": "g1",
  "unitree_g1_with_hands": "g1",
  "g1_23dof": "g1",
  "g1_29dof": "g1",
}

# Back-compat: read-only view of registered ids (do not mutate).
KNOWN_ROBOT_IDS: tuple[str, ...] = _BUILTIN_ROBOT_IDS


def known_robot_ids() -> tuple[str, ...]:
  return tuple(_KNOWN_ROBOT_IDS)


def register_robot_id(robot_id: str, *, aliases: tuple[str, ...] = ()) -> None:
  """Register a robot id (and optional CLI aliases) for ``--robot`` / data paths."""
  rid = robot_id.strip().lower()
  if not rid:
    raise ValueError("robot_id must be non-empty")
  if rid not in _KNOWN_ROBOT_IDS:
    _KNOWN_ROBOT_IDS.append(rid)
  for alias in aliases:
    key = alias.strip().lower()
    if key:
      ROBOT_ALIASES[key] = rid


def resolve_robot_id(name: str) -> RobotId:
  key = name.strip().lower()
  if key in _KNOWN_ROBOT_IDS:
    return key
  if key in ROBOT_ALIASES:
    return ROBOT_ALIASES[key]
  raise KeyError(
    f"Unknown robot {name!r}. Known: {list(_KNOWN_ROBOT_IDS)}; "
    f"aliases: {sorted(ROBOT_ALIASES)}"
  )
