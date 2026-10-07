"""Chat dataset format, validation and conversion. Pure Python: no torch/transformers imports.

    schema      message-structure rules (DataError, validate_messages)
    io          JSONL loading
    transforms  prompt-completion conversion, train/eval split
"""
