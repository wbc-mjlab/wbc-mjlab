"""Robot-agnostic actuation math."""

from wbc_mjlab.actuation.envelope import (
  TorqueSpeedEnvelope,
  TorqueSpeedEnvelopeTensors,
  build_envelope_tensors,
  clip_unitree_effort,
  tapered_peak,
  torque_speed_limits,
)

__all__ = [
  "TorqueSpeedEnvelope",
  "TorqueSpeedEnvelopeTensors",
  "build_envelope_tensors",
  "clip_unitree_effort",
  "tapered_peak",
  "torque_speed_limits",
]
