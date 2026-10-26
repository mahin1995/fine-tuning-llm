"""Supervised fine-tuning behind a swappable engine.

    options     TrainOptions: every knob, validated, no argparse
    engine      TrainingEngine port, engine registry, run_training(options, engine) -> TrainResult,
                run_info.json writer (the model-artifact contract). No torch import.
    trl_engine  TrlEngine: TRL SFTTrainer implementation (default)

Model-family settings (chat template kwargs, LoRA target layers) come from the model profile
(finetune/model_profiles.yaml), resolved from TrainOptions.model unless TrainOptions.profile is set.

Memory notes (RTX 3060, 12GB, Qwen3-0.6B):
- Full FT loads weights in fp32 and trains with bf16 autocast. Pure-bf16 weights
  would round small updates (lr ~2e-5) away. fp32 weights + grads (~6GB) +
  8-bit AdamW states (~1.5GB) + activations fit with gradient checkpointing.
- LoRA keeps the frozen base in bf16 and trains small fp32 adapters (~3-4GB).
"""
