"""Supervised fine-tuning with TRL.

    options  TrainOptions: every knob, validated, no argparse
    trainer  run_training(options) -> TrainResult

Memory notes (RTX 3060, 12GB, Qwen3-0.6B):
- Full FT loads weights in fp32 and trains with bf16 autocast. Pure-bf16 weights
  would round small updates (lr ~2e-5) away. fp32 weights + grads (~6GB) +
  8-bit AdamW states (~1.5GB) + activations fit with gradient checkpointing.
- LoRA keeps the frozen base in bf16 and trains small fp32 adapters (~3-4GB).
"""
