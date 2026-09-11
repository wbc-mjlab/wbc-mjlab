"""Paper / deploy env presets (expressive cfg mutators)."""

from wbc_mjlab.presets.binary_failure import apply_binary_failure
from wbc_mjlab.presets.end_effector import apply_end_effector
from wbc_mjlab.presets.se_actor import apply_se_actor
from wbc_mjlab.presets.wbc import apply_wbc
from wbc_mjlab.presets.zest import apply_zest

__all__ = [
  "apply_binary_failure",
  "apply_end_effector",
  "apply_se_actor",
  "apply_wbc",
  "apply_zest",
]
