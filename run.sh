#!/usr/bin/env bash
# Usage:
#   ./run.sh build              build the qwen-ft image
#   ./run.sh python train.py    run a command inside the container
#   ./run.sh bash               interactive shell
set -euo pipefail

cd "$(dirname "$0")"

# Fall back to sudo until the docker group is active in this login session.
DOCKER=docker
if ! docker info >/dev/null 2>&1; then DOCKER="sudo docker"; fi

IMAGE=qwen-ft
HF_CACHE="$HOME/.cache/huggingface"
mkdir -p "$HF_CACHE"

if [ "${1:-}" = "build" ]; then
  exec $DOCKER build -t "$IMAGE" .
fi

# --user keeps files written to the mounted project owned by you, not root.
# HF_HOME/HOME point at writable paths since the mapped uid has no home dir.
exec $DOCKER run --rm -it --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/hf \
  -v "$PWD":/workspace \
  -v "$HF_CACHE":/hf \
  -p 8000:8000 \
  "$IMAGE" "$@"
