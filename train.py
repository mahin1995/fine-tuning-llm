"""Supervised fine-tuning of a Qwen model with TRL's SFTTrainer.

Examples (inside the container, via ./run.sh):
    python train.py                                  # full fine-tuning, defaults
    python train.py --lora                           # LoRA adapter instead of full weights
    python train.py --epochs 10 --batch-size 2 --grad-accum 1   # smoke test on tiny data
    python train.py --eval-ratio 0.1                 # hold out 10% for eval loss

Memory notes (RTX 3060, 12GB, Qwen3-0.6B):
- Full FT loads weights in fp32 and trains with bf16 autocast. Pure-bf16 weights
  would round small updates (lr ~2e-5) away. fp32 weights + grads (~6GB) +
  8-bit AdamW states (~1.5GB) + activations fit with gradient checkpointing.
- LoRA keeps the frozen base in bf16 and trains small fp32 adapters (~3-4GB).
"""
import argparse
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from data_utils import load_conversations, to_prompt_completion, train_eval_split
from inference import DEFAULT_BASE_MODEL

MIN_RECOMMENDED_STEPS = 10


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_BASE_MODEL, help="base model (Hub id or local path)")
    p.add_argument("--data", default="data.jsonl")
    p.add_argument("--output", default="outputs/qwen3-ft")
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
    args = p.parse_args(argv)
    if args.lr is None:
        args.lr = 2e-4 if args.lora else 2e-5
    if not 0 <= args.warmup_ratio < 1:
        p.error("--warmup-ratio must be in [0, 1)")
    if args.merge and not args.lora:
        p.error("--merge only applies with --lora")
    return args


def pick_optimizer(use_cuda):
    """8-bit AdamW saves ~75% of optimizer memory but needs CUDA + bitsandbytes."""
    if not use_cuda:
        return "adamw_torch"
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        print("WARNING: bitsandbytes not installed, falling back to adamw_torch (more VRAM)")
        return "adamw_torch"
    return "adamw_8bit"


def warn_if_too_few_steps(n_train, args):
    steps_per_epoch = math.ceil(n_train / (args.batch_size * args.grad_accum))
    total = math.ceil(steps_per_epoch * args.epochs)
    print(f"train examples: {n_train} | effective batch: {args.batch_size * args.grad_accum} "
          f"| optimizer steps: {total}")
    if total < MIN_RECOMMENDED_STEPS:
        print(f"WARNING: only {total} optimizer steps. The model will barely change. "
              f"Add data, raise --epochs, or lower --batch-size/--grad-accum.")
    return total


def build_peft_config(args):
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )


def to_dataset(conversations):
    return Dataset.from_list([to_prompt_completion(c) for c in conversations])


def write_run_info(args, output_dir, total_steps, metrics):
    info = {
        "base_model": args.model,
        "method": "lora" if args.lora else "full",
        "args": vars(args),
        "optimizer_steps": total_steps,
        "metrics": metrics,
        "versions": {pkg: version(pkg) for pkg in ("torch", "transformers", "trl", "peft", "datasets")},
    }
    (Path(output_dir) / "run_info.json").write_text(json.dumps(info, indent=2, default=str))


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    if not use_cuda:
        print("WARNING: CUDA not available, training on CPU (only useful for smoke tests)")

    train_rows, eval_rows = train_eval_split(load_conversations(args.data), args.eval_ratio, args.seed)
    total_steps = warn_if_too_few_steps(len(train_rows), args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        raise SystemExit("tokenizer has no pad token; set one explicitly before training")

    # Full FT: fp32 master weights (bf16 autocast does the fast math). LoRA: frozen base can stay bf16.
    load_dtype = torch.bfloat16 if (args.lora and use_bf16) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=load_dtype)

    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_ratio,  # transformers 5: a float in [0, 1) is a ratio of total steps
        max_length=args.max_length,
        bf16=use_bf16,
        gradient_checkpointing=use_cuda and not args.no_gradient_checkpointing,
        optim=pick_optimizer(use_cuda),
        logging_steps=1,
        eval_strategy="epoch" if eval_rows else "no",
        save_strategy="epoch" if args.save_checkpoints else "no",
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=to_dataset(train_rows),
        eval_dataset=to_dataset(eval_rows) if eval_rows else None,
        processing_class=tokenizer,
        peft_config=build_peft_config(args) if args.lora else None,
    )
    if args.lora:
        trainer.model.print_trainable_parameters()

    try:
        result = trainer.train()
    except torch.OutOfMemoryError:
        print(
            "\nCUDA out of memory. Try, in order:\n"
            f"  1. --batch-size {max(1, args.batch_size // 2)} --grad-accum {args.grad_accum * 2}"
            "  (same effective batch, less memory)\n"
            f"  2. --max-length {args.max_length // 2}\n"
            "  3. --lora\n"
            "Check nothing else is using the GPU: nvidia-smi",
            file=sys.stderr,
        )
        return 2

    metrics = dict(result.metrics)
    if eval_rows:
        metrics.update(trainer.evaluate())

    trainer.save_model(args.output)  # LoRA: adapter only; full FT: all weights
    tokenizer.save_pretrained(args.output)
    write_run_info(args, args.output, total_steps, metrics)
    print(f"saved {'adapter' if args.lora else 'model'} to {args.output}")

    if args.merge:
        merged_dir = Path(args.output) / "merged"
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"saved merged model to {merged_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
