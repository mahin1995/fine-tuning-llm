"""Fixtures: a tiny randomly initialised Qwen3 model + tokenizer, built locally.

No network needed, runs on CPU in seconds. The chat template mirrors the parts of
Qwen3's template this project depends on (ChatML turns and the empty <think>
block in non-thinking mode). download_model.py re-checks the real template.
"""
import json
from pathlib import Path

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parent.parent

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]

# Qwen2/3 pre-tokenizer split pattern.
QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

QWEN3_LIKE_TEMPLATE = (
    "{%- if messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- set ns = namespace(last_query_index=messages|length - 1) %}"
    "{%- for message in messages[::-1] %}"
    "{%- set index = (messages|length - 1) - loop.index0 %}"
    "{%- if message.role == 'user' %}{%- set ns.last_query_index = index %}{%- break %}{%- endif %}"
    "{%- endfor %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' or (message.role == 'system' and not loop.first) %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{%- if loop.index0 > ns.last_query_index and loop.last %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + message.content.lstrip('\\n') }}"
    "{%- else %}"
    "{{- '<|im_start|>assistant\\n' + message.content }}"
    "{%- endif %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
    "{%- endif %}"
)


def corpus():
    for name in ("data.jsonl", "eval.jsonl"):
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            for msg in json.loads(line)["messages"]:
                yield msg["content"]


def build_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus(), trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
        chat_template=QWEN3_LIKE_TEMPLATE,
    )


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-qwen3")
    tokenizer = build_tokenizer()
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    Qwen3ForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tokenizer(tiny_model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model_dir)
