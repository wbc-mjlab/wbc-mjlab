"""WBC Viser play viewer — extends mjlab with motion-context GUI updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.viewer.viser.viewer import ViserPlayViewer
from typing_extensions import override

if TYPE_CHECKING:
  from wbc_mjlab.env.mdp.commands import MotionCommand


class WbcViserPlayViewer(ViserPlayViewer):
  """ViserPlayViewer that refreshes WBC motion-command context overlays."""

  def __init__(self, *args, task_id: str | None = None, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._task_id = task_id

  @override
  def setup(self) -> None:
    super().setup()
    motion = self._get_motion_command()
    if motion is not None:
      motion.set_viewer_task_id(self._task_id)

  @override
  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    motion = self._get_motion_command()
    if motion is not None:
      motion.update_viewer_gui(self._scene.env_idx)

  def _get_motion_command(self) -> MotionCommand | None:
    from wbc_mjlab.env.mdp.commands import MotionCommand

    env = self.env.unwrapped
    if "motion" not in env.command_manager.active_terms:
      return None
    term = env.command_manager.get_term("motion")
    return term if isinstance(term, MotionCommand) else None
