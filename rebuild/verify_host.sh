#!/usr/bin/env bash
set -euo pipefail

REBUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "architecture=$(uname -m)"
test "$(uname -m)" = "x86_64"

sed -n '1,12p' /etc/os-release
nvidia-smi
docker version
docker compose version
nvidia-ctk --version

docker run --rm --gpus all quantitize-platform-api:latest nvidia-smi -L
docker run --rm --gpus all \
  -v "$REBUILD_DIR:/rebuild:ro" \
  quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python \
  /rebuild/verify_runtime.py
