"""WBC-specific Viser viewer extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from wbc_mjlab.viewer.viser_play import WbcViserPlayViewer

__all__ = ["WbcViserPlayViewer"]


def __getattr__(name: str):
  if name == "WbcViserPlayViewer":
    from wbc_mjlab.viewer.viser_play import WbcViserPlayViewer

    return WbcViserPlayViewer
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
