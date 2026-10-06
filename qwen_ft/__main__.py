"""python -m qwen_ft <command> [options]

Commands:
    download   download the base model and check tokenizer / chat template
    validate   validate datasets (structure, duplicates, train/eval leakage, token lengths)
    train      fine-tune (full or --lora)
    evaluate   compare base vs fine-tuned on held-out questions
    chat       interactive terminal chat
    serve      HTTP API + browser chat UI

Run `python -m qwen_ft <command> --help` for a command's options.
"""
import importlib
import sys

COMMANDS = {
    "download": "qwen_ft.cli.download_model",
    "validate": "qwen_ft.cli.validate_data",
    "train": "qwen_ft.cli.train",
    "evaluate": "qwen_ft.cli.evaluate",
    "chat": "qwen_ft.cli.chat",
    "serve": "qwen_ft.cli.serve",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"unknown command {command!r}\n{__doc__}", file=sys.stderr)
        return 2
    # Imported lazily so `--help` and light commands don't load torch.
    module = importlib.import_module(COMMANDS[command])
    return module.main(rest)


if __name__ == "__main__":
    sys.exit(main())
