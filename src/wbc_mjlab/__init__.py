"""WBC motion tracking extension for mjlab."""

from importlib.metadata import PackageNotFoundError, version

try:
  __version__ = version("wbc-mjlab")  # single source: pyproject.toml
except PackageNotFoundError:  # source tree without an install
  __version__ = "0+unknown"

# Load mjlab before any submodule: its import registers task packages, including
# ``wbc_mjlab.mjlab_entry``. Importing e.g. ``wbc_mjlab.env.mdp.sampling`` first
# would otherwise run that registration against a half-imported ``wbc_mjlab.env.mdp``
# and drop every wbc-mjlab task from the registry.
import mjlab  # noqa: E402, F401
