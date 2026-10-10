"""Per-user wbc-mjlab configuration, outside any repo: Studio presets / modules and custom
robots. ``$WBC_MJLAB_CONFIG_DIR``, else ``$XDG_CONFIG_HOME/wbc-mjlab`` (``~/.config/wbc-mjlab``)."""

from __future__ import annotations

import os
from pathlib import Path


def config_dir() -> Path:
  env = os.environ.get("WBC_MJLAB_CONFIG_DIR")
  if env:
    return Path(env).expanduser()
  return (
    Path(os.environ.get("XDG_CONFIG_HOME") or "~/.config").expanduser() / "wbc-mjlab"
  )


__all__ = ["config_dir"]
