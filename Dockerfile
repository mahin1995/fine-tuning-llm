FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

# Throwaway container: overriding PEP 668 is safe here. `python -m pip` makes
# sure packages land in the same interpreter that already has torch.
RUN python -m pip install --no-cache-dir --break-system-packages \
    --default-timeout=100 --retries 10 \
    transformers trl datasets accelerate peft bitsandbytes \
    fastapi "uvicorn[standard]"

WORKDIR /workspace
