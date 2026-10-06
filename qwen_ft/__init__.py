"""Fine-tuning harness for Qwen chat models.

Packages, in dependency order (each may only import the ones above it):

    config       shared constants, no dependencies
    data         dataset format, validation, splitting (pure Python, no torch)
    modeling     model/tokenizer loading and chat generation (torch, transformers)
    training     supervised fine-tuning with TRL (data, modeling config)
    evaluation   base vs fine-tuned comparison (data; model loader is injected)
    serving      FastAPI app (data, modeling.params; model is injected)
    cli          command-line entry points: the only place that wires packages together

Run commands with `python -m qwen_ft <command>`; see `python -m qwen_ft --help`.
"""
