"""Fine-tune a Qwen model.

    python -m qwen_ft train                                   # full fine-tuning, defaults
    python -m qwen_ft train --lora --merge                    # LoRA adapter (+ merged copy)
    python -m qwen_ft train --epochs 10 --batch-size 2 --grad-accum 1   # smoke test on tiny data
    python -m qwen_ft train --eval-ratio 0.1                  # hold out 10% for eval loss
"""
import argparse
import sys

from qwen_ft.config import DEFAULT_BASE_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_TRAIN_DATA
from qwen_ft.training.options import LoraOptions, TrainOptions


def parse_options(argv=None) -> TrainOptions:
    p = argparse.ArgumentParser(prog="python -m qwen_ft train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_BASE_MODEL, help="base model (Hub id or local path)")
    p.add_argument("--data", default=DEFAULT_TRAIN_DATA)
    p.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--eval-ratio", type=float, default=0.0, help="fraction of data held out for eval loss")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=None, help="default: 2e-5 full FT, 2e-4 LoRA")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024, help="max tokens per example (longer are truncated)")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-checkpoints", action="store_true", help="also save a checkpoint each epoch (keeps 1)")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    lora = p.add_argument_group("LoRA")
    lora.add_argument("--lora", action="store_true", help="train a LoRA adapter instead of all weights")
    lora.add_argument("--lora-r", type=int, default=16)
    lora.add_argument("--lora-alpha", type=int, default=32)
    lora.add_argument("--lora-dropout", type=float, default=0.05)
    lora.add_argument("--merge", action="store_true", help="also save a merged full model to <output>/merged")
    a = p.parse_args(argv)

    if a.merge and not a.lora:
        p.error("--merge only applies with --lora")
    try:
        return TrainOptions(
            model=a.model, data=a.data, output=a.output, eval_ratio=a.eval_ratio, epochs=a.epochs,
            lr=a.lr, batch_size=a.batch_size, grad_accum=a.grad_accum, max_length=a.max_length,
            warmup_ratio=a.warmup_ratio, seed=a.seed, save_checkpoints=a.save_checkpoints,
            gradient_checkpointing=not a.no_gradient_checkpointing,
            lora=LoraOptions(r=a.lora_r, alpha=a.lora_alpha, dropout=a.lora_dropout, merge=a.merge)
            if a.lora else None,
        )
    except ValueError as e:
        p.error(str(e))


def main(argv=None):
    opts = parse_options(argv)
    # Heavy imports (torch, trl) only after argument parsing succeeded.
    from qwen_ft.training.trainer import TrainingOutOfMemory, run_training

    try:
        run_training(opts)
    except TrainingOutOfMemory as e:
        print(f"\n{e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
