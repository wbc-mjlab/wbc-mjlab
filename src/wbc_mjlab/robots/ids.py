"""Robot id resolution for CLI ``--robot`` and ``data/<robot>/`` paths."""

from __future__ import annotations

from typing import Literal

RobotId = Literal["g1"]

KNOWN_ROBOT_IDS: tuple[RobotId, ...] = ("g1",)

ROBOT_ALIASES: dict[str, RobotId] = {
  "unitree_g1": "g1",
  "unitree_g1_with_hands": "g1",
  "g1_23dof": "g1",
  "g1_29dof": "g1",
}


def resolve_robot_id(name: str) -> RobotId:
  key = name.strip().lower()
  if key in KNOWN_ROBOT_IDS:
    return key  # type: ignore[return-value]
  if key in ROBOT_ALIASES:
    return ROBOT_ALIASES[key]
  raise KeyError(
    f"Unknown robot {name!r}. Known: {list(KNOWN_ROBOT_IDS)}; "
    f"aliases: {sorted(ROBOT_ALIASES)}"
  )

