#!/usr/bin/env bash
set -euo pipefail

# Named volumes can retain the image UID after the editor matches the host UID.
for directory in /opt/venv /home/developer/.cache /home/developer/.config/wbc-mjlab; do
  if [[ "$(stat -c %u "$directory")" != "$(id -u)" ]]; then
    sudo chown -R "$(id -u):$(id -g)" "$directory"
  fi
done

uv sync --locked --extra cu128 --group dev --group docs
uv run --no-sync wbc-mjlab-list-envs
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "GPU unavailable: check the host NVIDIA driver and Container Toolkit"; print("CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))'
