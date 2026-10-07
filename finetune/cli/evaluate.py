"""Compare the base model and the fine-tuned model on held-out questions.

For each example it reports the answer loss (mean per-token NLL of the reference
answer, lower is better) and both models' greedy answers side by side.

    python -m finetune evaluate --model outputs/qwen3-ft
    python -m finetune evaluate --model outputs/qwen3-ft --base Qwen/Qwen3-0.6B
"""
import argparse
import sys
from pathlib import Path

from finetune.config import DEFAULT_EVAL_DATA, DEFAULT_OUTPUT_DIR
from finetune.data.io import load_conversations
from finetune.evaluation.evaluator import base_model_of, compare_models
from finetune.evaluation.report import render_report
from finetune.modeling.params import GenerationParams


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m finetune evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_OUTPUT_DIR, help="fine-tuned model or adapter dir")
    p.add_argument("--base", default=None, help="base model (default: read from run_info.json)")
    p.add_argument("--eval-data", default=DEFAULT_EVAL_DATA)
    p.add_argument("--report", default=None, help="markdown report path (default: <model>/eval_report.md)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args(argv)

    from finetune.modeling.chat_model import ChatModel

    base_path = args.base or base_model_of(args.model)
    report_path = Path(args.report or Path(args.model) / "eval_report.md")
    conversations = load_conversations(args.eval_data)
    params = GenerationParams(temperature=0.0, max_new_tokens=args.max_new_tokens)  # greedy: reproducible

    print(f"evaluating {base_path} vs {args.model} on {len(conversations)} examples ...")
    base, tuned = compare_models(ChatModel.load, base_path, args.model, conversations, params)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(conversations, base, tuned), encoding="utf-8")
    print(f"\nmean answer loss  base: {base.mean_loss:.4f}  fine-tuned: {tuned.mean_loss:.4f}")
    print(f"report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
