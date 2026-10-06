"""Fixtures: a tiny randomly initialised Qwen3 model + tokenizer, built locally.

No network needed, runs on CPU in seconds. The chat template mirrors the parts of
Qwen3's template this project depends on: ChatML turns, the empty <think> block in
non-thinking mode, and tool calling (<tools> system block, <tool_call> in assistant
turns, <tool_response> for tool results). `python -m qwen_ft download` re-checks
the real template.
"""
import json
from pathlib import Path

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA = ROOT / "data" / "train.jsonl"
EVAL_DATA = ROOT / "data" / "eval.jsonl"

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
# Added as normal (non-special) tokens, like Qwen3, so decoding keeps them.
TOOL_TOKENS = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]

# Qwen2/3 pre-tokenizer split pattern.
QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

QWEN3_LIKE_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\n' }}"
    "{%- if messages[0].role == 'system' %}{{- messages[0].content + '\n\n' }}{%- endif %}"
    "{{- '# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>' }}"
    "{%- for tool in tools %}{{- '\n' }}{{- tool | tojson }}{%- endfor %}"
    "{{- '\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n' }}"
    "{%- elif messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- set ns = namespace(last_query_index=messages|length - 1) %}"
    "{%- for message in messages[::-1] %}"
    "{%- set index = (messages|length - 1) - loop.index0 %}"
    "{%- if message.role == 'user' and not message.content.startswith('<tool_response>') %}"
    "{%- set ns.last_query_index = index %}{%- break %}{%- endif %}"
    "{%- endfor %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' or (message.role == 'system' and not loop.first) %}"
    "{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{%- if loop.index0 > ns.last_query_index and loop.last %}"
    "{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' + message.content.lstrip('\n') }}"
    "{%- else %}"
    "{{- '<|im_start|>assistant\n' + message.content }}"
    "{%- endif %}"
    "{%- if message.tool_calls %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if (loop.first and message.content) or (not loop.first) %}{{- '\n' }}{%- endif %}"
    "{%- if tool_call.function %}{%- set tool_call = tool_call.function %}{%- endif %}"
    "{{- '<tool_call>\n{\"name\": \"' + tool_call.name + '\", \"arguments\": ' }}"
    "{%- if tool_call.arguments is string %}{{- tool_call.arguments }}"
    "{%- else %}{{- tool_call.arguments | tojson }}{%- endif %}"
    "{{- '}\n</tool_call>' }}"
    "{%- endfor %}"
    "{%- endif %}"
    "{{- '<|im_end|>\n' }}"
    "{%- elif message.role == 'tool' %}"
    "{%- if loop.first or (messages[loop.index0 - 1].role != 'tool') %}{{- '<|im_start|>user' }}{%- endif %}"
    "{{- '\n<tool_response>\n' + message.content + '\n</tool_response>' }}"
    "{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}{{- '<|im_end|>\n' }}{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}"
    "{%- endif %}"
)


def corpus():
    for path in (TRAIN_DATA, EVAL_DATA):
        for line in path.read_text(encoding="utf-8").splitlines():
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
    tok.add_tokens(TOOL_TOKENS)
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
