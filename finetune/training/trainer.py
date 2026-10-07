"""Supervised fine-tuning of a Qwen model with TRL's SFTTrainer."""
import json
import math
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from finetune.data.io import load_conversations
from finetune.data.transforms import to_prompt_completion, train_eval_split
from finetune.training.options import TrainOptions

MIN_RECOMMENDED_STEPS = 10


class TrainingOutOfMemory(RuntimeError):
    """CUDA ran out of memory; the message says which options to change."""


@dataclass
class TrainResult:
    output_dir: str
    optimizer_steps: int
    metrics: dict
    merged_dir: str | None = None


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


def count_optimizer_steps(n_train, opts):
    steps_per_epoch = math.ceil(n_train / opts.effective_batch_size)
    return math.ceil(steps_per_epoch * opts.epochs)


def warn_if_too_few_steps(n_train, opts):
    total = count_optimizer_steps(n_train, opts)
    print(f"train examples: {n_train} | effective batch: {opts.effective_batch_size} | optimizer steps: {total}")
    if total < MIN_RECOMMENDED_STEPS:
        print(f"WARNING: only {total} optimizer steps. The model will barely change. "
              f"Add data, raise --epochs, or lower --batch-size/--grad-accum.")
    return total


def build_peft_config(lora):
    from peft import LoraConfig

    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        task_type="CAUSAL_LM",
    )


def to_dataset(conversations):
    return Dataset.from_list([to_prompt_completion(c) for c in conversations])


def oom_advice(opts):
    return (
        "CUDA out of memory. Try, in order:\n"
        f"  1. --batch-size {max(1, opts.batch_size // 2)} --grad-accum {opts.grad_accum * 2}"
        "  (same effective batch, less memory)\n"
        f"  2. --max-length {opts.max_length // 2}\n"
        "  3. --lora\n"
        "Check nothing else is using the GPU: nvidia-smi"
    )


def write_run_info(opts, total_steps, metrics):
    info = {
        "base_model": opts.model,
        "method": opts.method,
        "options": asdict(opts),
        "optimizer_steps": total_steps,
        "metrics": metrics,
        "versions": {pkg: version(pkg) for pkg in ("torch", "transformers", "trl", "peft", "datasets")},
    }
    (Path(opts.output) / "run_info.json").write_text(json.dumps(info, indent=2, default=str))


def build_sft_config(opts, use_cuda, use_bf16, has_eval):
    return SFTConfig(
        output_dir=opts.output,
        num_train_epochs=opts.epochs,
        per_device_train_batch_size=opts.batch_size,
        per_device_eval_batch_size=opts.batch_size,
        gradient_accumulation_steps=opts.grad_accum,
        learning_rate=opts.lr,
        lr_scheduler_type="cosine",
        warmup_steps=opts.warmup_ratio,  # transformers 5: a float in [0, 1) is a ratio of total steps
        max_length=opts.max_length,
        bf16=use_bf16,
        gradient_checkpointing=use_cuda and opts.gradient_checkpointing,
        optim=pick_optimizer(use_cuda),
        logging_steps=1,
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch" if opts.save_checkpoints else "no",
        save_total_limit=1,
        report_to="none",
        seed=opts.seed,
    )


def run_training(opts: TrainOptions) -> TrainResult:
    set_seed(opts.seed)
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    if not use_cuda:
        print("WARNING: CUDA not available, training on CPU (only useful for smoke tests)")

    train_rows, eval_rows = train_eval_split(load_conversations(opts.data), opts.eval_ratio, opts.seed)
    total_steps = warn_if_too_few_steps(len(train_rows), opts)

    tokenizer = AutoTokenizer.from_pretrained(opts.model)
    if tokenizer.pad_token is None:
        raise ValueError("tokenizer has no pad token; set one explicitly before training")

    # Full FT: fp32 master weights (bf16 autocast does the fast math). LoRA: frozen base can stay bf16.
    load_dtype = torch.bfloat16 if (opts.lora and use_bf16) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(opts.model, dtype=load_dtype)

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(opts, use_cuda, use_bf16, has_eval=bool(eval_rows)),
        train_dataset=to_dataset(train_rows),
        eval_dataset=to_dataset(eval_rows) if eval_rows else None,
        processing_class=tokenizer,
        peft_config=build_peft_config(opts.lora) if opts.lora else None,
    )
    if opts.lora:
        trainer.model.print_trainable_parameters()

    try:
        result = trainer.train()
    except torch.OutOfMemoryError as e:
        raise TrainingOutOfMemory(oom_advice(opts)) from e

    metrics = dict(result.metrics)
    if eval_rows:
        metrics.update(trainer.evaluate())

    trainer.save_model(opts.output)  # LoRA: adapter only; full FT: all weights
    tokenizer.save_pretrained(opts.output)
    write_run_info(opts, total_steps, metrics)
    print(f"saved {'adapter' if opts.lora else 'model'} to {opts.output}", file=sys.stderr)

    merged_dir = None
    if opts.lora and opts.lora.merge:
        merged_dir = str(Path(opts.output) / "merged")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"saved merged model to {merged_dir}", file=sys.stderr)

    return TrainResult(output_dir=opts.output, optimizer_steps=total_steps, metrics=metrics, merged_dir=merged_dir)
