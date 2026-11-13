#!/usr/bin/env bash
set -euo pipefail

rm -f /tmp/wbc-ready
bash .devcontainer/setup.sh
touch /tmp/wbc-ready
exec "$@"
