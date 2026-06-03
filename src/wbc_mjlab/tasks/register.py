"""mjlab entry-point module: importing it registers all WBC task presets."""


def register_default() -> None:
  from wbc_mjlab.tasks import DEFAULT_TASK_ID, register_all_wbc_tasks, _set_active_run

  register_all_wbc_tasks()
  _set_active_run("g1", DEFAULT_TASK_ID)


register_default()
