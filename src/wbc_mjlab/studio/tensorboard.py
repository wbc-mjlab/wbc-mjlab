"""TensorBoard over ``logs/rsl_rl``, started on demand by the Studio.

One TensorBoard serves every run (``<experiment>/<run>`` in its run list). A TensorBoard already
serving the same folder (e.g. started by hand) is reused; one started here is stopped when the
Studio exits, also when it is killed.
"""

from __future__ import annotations

import atexit
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

from wbc_mjlab.studio.jobs import logs_dir

DEFAULT_PORT = 6006


def _logdir_of(port: int) -> str | None:
  """``--logdir`` of the TensorBoard on ``port`` (None: nothing / not a TensorBoard there)."""
  try:
    with urllib.request.urlopen(
      f"http://127.0.0.1:{port}/data/environment", timeout=0.5
    ) as r:
      return str(json.loads(r.read().decode()).get("data_location", ""))
  except Exception:  # noqa: BLE001
    return None


def _die_with_parent() -> None:
  """Child-side (Linux): SIGTERM when the Studio dies, however it dies (atexit misses signals)."""
  try:
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(
      1, signal.SIGTERM
    )  # PR_SET_PDEATHSIG
  except Exception:  # noqa: BLE001 — not Linux: atexit still stops it on a normal exit
    pass


def _port_free(port: int) -> bool:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
      s.bind(("127.0.0.1", port))
    except OSError:
      return False
  return True


class TensorBoard:
  def __init__(self, host: str = "127.0.0.1", logdir: Path | None = None):
    self.host = host
    self.logdir = (logdir or logs_dir() / "rsl_rl").resolve()
    self.port: int | None = None
    self.proc: subprocess.Popen | None = None
    self.log = logs_dir() / "studio" / "tensorboard.log"
    self._lock = threading.Lock()
    atexit.register(self.stop)

  def _ours(self, port: int) -> bool:
    found = _logdir_of(port)
    return found is not None and Path(found).resolve() == self.logdir

  @property
  def running(self) -> bool:
    if self.port is None:
      return False
    if self.proc is not None and self.proc.poll() is not None:
      self.proc, self.port = None, None
      return False
    return self._ours(self.port)

  def start(self, timeout: float = 30.0) -> int:
    """Port of a TensorBoard serving the run logs (started if none is)."""
    with self._lock:
      if self.running:
        return self.port
      for port in range(DEFAULT_PORT, DEFAULT_PORT + 20):
        if self._ours(port):  # reuse
          self.port = port
          return port
        if _port_free(port):
          break
      else:
        raise RuntimeError(f"no free port in {DEFAULT_PORT}–{DEFAULT_PORT + 19}")
      self.logdir.mkdir(parents=True, exist_ok=True)
      self.log.parent.mkdir(parents=True, exist_ok=True)
      cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(self.logdir),
        "--port",
        str(port),
        "--host",
        self.host,
        "--reload_interval",
        "15",
      ]
      with self.log.open("w") as f:
        self.proc = subprocess.Popen(
          cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=_die_with_parent
        )
      t0 = time.time()
      while time.time() - t0 < timeout:
        if self.proc.poll() is not None:
          raise RuntimeError(f"tensorboard exited: see {self.log}")
        if self._ours(port):
          self.port = port
          return port
        time.sleep(0.25)
      self.stop()
      raise RuntimeError(
        f"tensorboard did not come up in {timeout:.0f} s: see {self.log}"
      )

  def stop(self) -> None:
    proc, self.proc = self.proc, None
    if proc is not None and proc.poll() is None:
      proc.terminate()
      try:
        proc.wait(5)
      except subprocess.TimeoutExpired:
        proc.kill()
    self.port = None

  def state(self) -> dict:
    running = self.running
    return {
      "running": running,
      "port": self.port if running else None,
      "logdir": str(self.logdir),
    }

  @staticmethod
  def run_path(log_dir: str | Path) -> str:
    """Path (with the run filter) of one run, relative to the TensorBoard root URL."""
    name = Path(log_dir).name
    return f"/?runFilter={quote(name)}#timeseries" if name else "/"


__all__ = ["DEFAULT_PORT", "TensorBoard"]
