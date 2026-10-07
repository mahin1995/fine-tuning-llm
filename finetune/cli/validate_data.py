"""Validate chat datasets before training and print basic statistics.

    python -m finetune validate                      # data/train.jsonl and data/eval.jsonl
    python -m finetune validate data/train.jsonl --tokenizer Qwen/Qwen3-0.6B --max-length 1024

Checks: valid JSON, chat structure (roles alternate, ends with assistant), no
duplicate questions, and no question shared between train and eval files.
With --tokenizer it also reports token lengths and examples that would be truncated.
"""
import argparse
import statistics
import sys

from finetune.config import CHAT_TEMPLATE_KWARGS, DEFAULT_EVAL_DATA, DEFAULT_TRAIN_DATA
from finetune.data.io import load_conversations
from finetune.data.schema import DataError
from finetune.data.transforms import question_of


def token_lengths(conversations, tokenizer_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [
        len(tokenizer.apply_chat_template(m, tokenize=True, **CHAT_TEMPLATE_KWARGS)["input_ids"])
        for m in conversations
    ]


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m finetune validate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", default=[DEFAULT_TRAIN_DATA, DEFAULT_EVAL_DATA])
    p.add_argument("--tokenizer", default=None, help="also report token lengths using this tokenizer")
    p.add_argument("--max-length", type=int, default=1024)
    args = p.parse_args(argv)

    ok = True
    seen = {}  # question -> file it first appeared in
    for path in args.files:
        try:
            conversations = load_conversations(path)
        except (DataError, OSError) as e:
            print(f"FAIL {e}")
            ok = False
            continue

        dupes, leaks = 0, 0
        for messages in conversations:
            q = question_of(messages)
            if q in seen:
                if seen[q] == path:
                    dupes += 1
                else:
                    leaks += 1
                    print(f"  WARN question in both {seen[q]} and {path}: {q[:70]}")
            else:
                seen[q] = path

        answer_chars = [len(m[-1]["content"]) for m in conversations]
        print(f"OK   {path}: {len(conversations)} examples, "
              f"answer chars median {statistics.median(answer_chars):.0f} / max {max(answer_chars)}")
        if dupes:
            print(f"  WARN {dupes} duplicate question(s) inside {path}")
        if leaks:
            ok = False  # eval questions seen in training make evaluation meaningless

        if args.tokenizer:
            lengths = token_lengths(conversations, args.tokenizer)
            too_long = sum(n > args.max_length for n in lengths)
            print(f"  tokens median {statistics.median(lengths):.0f} / max {max(lengths)}"
                  f"; {too_long} over --max-length {args.max_length} (would be truncated)")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
