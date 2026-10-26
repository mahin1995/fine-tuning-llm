FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    CREWAI_DISABLE_TELEMETRY=true \
    OTEL_SDK_DISABLED=true \
    CREWAI_TRACING_ENABLED=false

COPY requirements.txt /tmp/requirements.txt
COPY apps/requirements.txt /tmp/requirements-apps.txt

# Throwaway container: overriding PEP 668 is safe here. `python -m pip` makes
# sure packages land in the same interpreter that already has torch.
# The constraint pins the base image's CUDA torch so no dependency can swap it
# for a different build; the final check fails the build if that ever happens.
RUN python -c "from importlib.metadata import version; print('torch==' + version('torch'))" > /tmp/constraints.txt \
 && python -m pip install --no-cache-dir --break-system-packages \
      --default-timeout=100 --retries 10 \
      -c /tmp/constraints.txt -r /tmp/requirements.txt -r /tmp/requirements-apps.txt \
 && python -c "import torch; assert torch.version.cuda, 'CPU-only torch was installed'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

WORKDIR /workspace
