"""WBC Studio: robots, motion data (browse / QA / convert), training jobs and policy playback.

Usage::

  wbc-mjlab-studio                      # web UI, reachable over the network
  wbc-mjlab-studio --host 127.0.0.1     # this machine only
  wbc-mjlab-studio --token SECRET       # network clients need ?token=SECRET once
  wbc-mjlab-studio --ui viser           # the same Studio as a viser app
  wbc-mjlab-studio --port 8765 --robot g1 --device cuda:0

Like viser, the Studio is open to anyone who can reach the port — it can start jobs and read
local folders, so use it on trusted networks or pass ``--token``. Ctrl+C stops it and frees
the port (and TensorBoard's); detached training / conversion jobs keep running.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import webbrowser
from typing import Literal

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def lan_addresses() -> list[str]:
  """IPv4 addresses other machines can reach this one at (best effort)."""
  found: list[str] = []
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      # UDP connect sends no packet; it just picks the default route's address
      s.connect(("10.255.255.255", 1))
      found.append(s.getsockname()[0])
  except OSError:
    pass
  try:
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
      ip = info[4][0]
      if ip not in found and not ip.startswith("127."):
        found.append(ip)
  except OSError:
    pass
  return found


def main(
  ui: Literal["web", "viser"] = "web",
  host: str = "0.0.0.0",
  port: int = 8765,
  robot: str | None = None,
  device: str | None = None,
  open_browser: bool = True,
  token: str | None = None,
) -> None:
  """Start the Studio.

  Args:
    ui: ``web`` (FastAPI + three.js) or ``viser``.
    host: Interface to serve on: ``0.0.0.0`` (default) for the network, ``127.0.0.1`` for
      this machine only.
    port: Port of the UI.
    robot: Robot to start with (default: the first registered, ``g1``).
    device: Torch device for playback and conversion (default: ``cuda:0`` if available).
    open_browser: Open the UI in the default browser.
    token: Require this token from network clients (web UI; default: no access control).
  """
  import mjlab  # noqa: F401 — load mjlab.tasks entry points first (extension robots)
  import torch

  from wbc_mjlab.studio.core import StudioCore

  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  core = StudioCore(robot=robot, device=device)
  core.tensorboard.host = "127.0.0.1" if host in LOCAL_HOSTS else "0.0.0.0"
  core.start()

  remote = host not in LOCAL_HOSTS
  if not remote or ui != "web":
    token = None
  query = f"/?token={token}" if token else "/"
  local_url = f"http://127.0.0.1:{port}/"  # loopback needs no token
  hosts = (
    (lan_addresses() or [socket.gethostname()]) if host in ("0.0.0.0", "::") else [host]
  )
  urls = [f"http://{h}:{port}{query}" for h in hosts] if remote else [local_url]

  print("\n  WBC Studio")
  for u in urls:
    print(f"    → {u}")
  if remote:
    print(f"    → {local_url}  (this machine{', no token needed' if token else ''})")
    if not token:
      print("    (open to anyone who can reach this port: trusted networks only)")
  print("    Ctrl+C to stop", flush=True)

  # Ctrl+C, kill (SIGTERM) and closing the terminal (SIGHUP) all shut down cleanly — also
  # when started in the background, where SIGINT comes in ignored (uvicorn installs its own
  # INT / TERM handlers while it runs; HUP is forwarded to whichever is current)
  signal.signal(signal.SIGINT, signal.default_int_handler)
  signal.signal(signal.SIGTERM, signal.default_int_handler)
  signal.signal(signal.SIGHUP, lambda *_: os.kill(os.getpid(), signal.SIGINT))
  try:
    if ui == "viser":
      from wbc_mjlab.studio.viser_app import ViserStudio

      ViserStudio(core, host=host, port=port).run(open_browser=open_browser)
    else:
      import uvicorn

      from wbc_mjlab.studio.web import create_app

      if open_browser:
        try:
          webbrowser.open(local_url)
        except Exception:  # noqa: BLE001
          pass
      uvicorn.run(
        create_app(core, token=token),
        host=host,
        port=port,
        log_level="warning",
        ws_ping_interval=None,
        timeout_graceful_shutdown=3,  # open browser tabs don't hold the port
      )
  except KeyboardInterrupt:
    pass
  finally:
    core.close()
    print("\n  WBC Studio stopped: port", port, "is free.", flush=True)
    sys.stdout.flush()
    # skip interpreter teardown: the daemon loop thread / CUDA can stall it, keeping the
    # process (and so the port) alive
    os._exit(0)


def cli() -> None:
  import tyro

  tyro.cli(main)


if __name__ == "__main__":
  cli()
