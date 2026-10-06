#!/usr/bin/env bash
# Usage:
#   ./run.sh build                                   build the qwen-ft image
#   ./run.sh python -m qwen_ft train                 run a command inside the container
#   ./run.sh python -m agent "What is 17 * 23?"      run the tool-calling agent
#   ./run.sh bash                                    interactive shell
#   PORT=8000 ./run.sh python -m qwen_ft serve       also publish a port (on 127.0.0.1 only)
#   nohup ./run.sh python -m qwen_ft train > train.log 2>&1 &   works too (no TTY needed)
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

# -it only with a real terminal; otherwise docker fails with "the input device is not a TTY".
TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS=(-it); fi

# Publish a port only when asked, so training and a shell can run side by side.
# Bound to 127.0.0.1 so the chat server isn't exposed to the local network.
PORT_FLAGS=()
if [ -n "${PORT:-}" ]; then PORT_FLAGS=(-p "127.0.0.1:$PORT:$PORT"); fi

# --user keeps files written to the mounted project owned by you, not root.
# HF_HOME/HOME point at writable paths since the mapped uid has no home dir;
# USER covers libraries that look up the user name (the uid has no passwd entry).
exec $DOCKER run --rm "${TTY_FLAGS[@]}" --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/hf -e USER="$(id -un)" \
  -v "$PWD":/workspace \
  -v "$HF_CACHE":/hf \
  "${PORT_FLAGS[@]}" \
  "$IMAGE" "$@"
