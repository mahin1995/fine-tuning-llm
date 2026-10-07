#!/usr/bin/env bash
# model_serving in its own container (independent of the training image).
# Usage:
#   ./run.sh build                                build the model-serving image
#   ./run.sh download qwen3-0.6b                  fetch a model into ~/.cache/huggingface
#   ./run.sh serve --model qwen3-0.6b             OpenAI-compatible API on 127.0.0.1:8001
#   ./run.sh serve --model qwen3-ft               a fine-tuned output from ../outputs/qwen3-ft
#   ./run.sh check --model qwen3-0.6b             contract check against the running server
#   ./run.sh test                                 the component's unit tests (CPU)
#   ./run.sh <any other model_serving command>    list, vllm-command, ollama-help, ...
# PORT=9000 ./run.sh serve ... changes the published port.
set -euo pipefail

cd "$(dirname "$0")"

DOCKER=docker
if ! docker info >/dev/null 2>&1; then DOCKER="sudo docker"; fi

IMAGE=model-serving
PORT="${PORT:-8001}"
HF_CACHE="$HOME/.cache/huggingface"
OUTPUTS="$(cd .. && pwd)/outputs"
mkdir -p "$HF_CACHE" "$OUTPUTS"

if [ "${1:-}" = "build" ]; then
  exec $DOCKER build -t "$IMAGE" .
fi

TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS=(-it); fi

COMMON=(--rm "${TTY_FLAGS[@]}" --user "$(id -u):$(id -g)" -e USER="$(id -un)"
        -v "$HF_CACHE":/hf -v "$OUTPUTS":/models/outputs:ro)

case "${1:-}" in
  serve)
    # Bound to 127.0.0.1: the API is not exposed to the local network.
    # MODEL_SERVING_API_KEY (if set on the host) turns on bearer auth.
    shift
    exec $DOCKER run "${COMMON[@]}" --gpus all --ipc=host -p "127.0.0.1:$PORT:$PORT" \
      -e MODEL_SERVING_API_KEY="${MODEL_SERVING_API_KEY:-}" \
      "$IMAGE" serve --host 0.0.0.0 --port "$PORT" "$@" ;;
  check)
    # Host network so localhost reaches the server published above (or a vLLM / Ollama one).
    shift
    exec $DOCKER run "${COMMON[@]}" --network host -e MODEL_SERVING_API_KEY="${MODEL_SERVING_API_KEY:-}" \
      "$IMAGE" check --base-url "http://localhost:$PORT/v1" "$@" ;;
  test)
    exec $DOCKER run "${COMMON[@]}" --entrypoint python "$IMAGE" -m pytest -q -p no:cacheprovider tests ;;
  *)
    exec $DOCKER run "${COMMON[@]}" "$IMAGE" "$@" ;;
esac
