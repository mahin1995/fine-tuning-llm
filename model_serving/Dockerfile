# Serving image: only what inference needs. Build context is this folder (model_serving/),
# so the training code and the apps can't leak in.
FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/hf \
    HOME=/tmp

COPY requirements.txt /tmp/requirements.txt

# Same torch guard as the training image: the constraint pins the base image's CUDA torch,
# the final check fails the build if a dependency ever swaps it.
RUN python -c "from importlib.metadata import version; print('torch==' + version('torch'))" > /tmp/constraints.txt \
 && python -m pip install --no-cache-dir --break-system-packages \
      --default-timeout=100 --retries 10 \
      -c /tmp/constraints.txt -r /tmp/requirements.txt \
 && python -c "import torch; assert torch.version.cuda, 'CPU-only torch was installed'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

WORKDIR /app
COPY models.yaml /app/models.yaml
COPY src /app/src
COPY tests /app/tests

# Non-root by default; run.sh passes your uid so the mounted HF cache stays yours.
USER 1000:1000
EXPOSE 8001
ENTRYPOINT ["python", "-m", "model_serving"]
CMD ["serve", "--model", "qwen3-0.6b", "--host", "0.0.0.0", "--port", "8001"]
