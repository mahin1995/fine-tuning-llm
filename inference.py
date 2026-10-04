"""Shared model loading and chat generation for evaluate.py, chat.py and serve.py.

Works with three kinds of model paths:
- a Hub id or a full checkpoint directory (full fine-tuning output)
- a LoRA adapter directory (contains adapter_config.json); merged on load
"""
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from data_utils import CHAT_TEMPLATE_KWARGS

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class GenerationParams:
    # Qwen3 recommended sampling for non-thinking mode. temperature=0 means greedy.
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05

    def to_generate_kwargs(self):
        if self.temperature <= 0:
            return {"max_new_tokens": self.max_new_tokens, "do_sample": False,
                    "repetition_penalty": self.repetition_penalty}
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }


GREEDY = GenerationParams(temperature=0.0)


class _StopOnEvent(StoppingCriteria):
    """Lets the consumer of stream() abort generation early (e.g. Ctrl-C in chat.py)."""

    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def inference_dtype(device):
    # bf16 halves memory on GPU; on CPU bf16 matmuls are slow, so stay in fp32.
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def is_adapter_dir(model_path):
    return (Path(model_path) / "adapter_config.json").is_file()


def load_model_and_tokenizer(model_path, device=None):
    device = device or pick_device()
    dtype = inference_dtype(device)
    if is_adapter_dir(model_path):
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        model = model.merge_and_unload()  # plain model: faster generation, no peft overhead
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    # train.py saves the tokenizer next to the weights/adapter, so this works for all three cases.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device).eval()
    return model, tokenizer


class ChatModel:
    """Thin wrapper that applies the chat template consistently with training."""

    def __init__(self, model, tokenizer, name="model"):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        # One GPU, one model: serialize generate() calls coming from server threads.
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_path, device=None):
        model, tokenizer = load_model_and_tokenizer(model_path, device)
        return cls(model, tokenizer, name=str(model_path))

    @property
    def device(self):
        return self.model.device

    def build_inputs(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **CHAT_TEMPLATE_KWARGS,
        ).to(self.device)

    def _generate_kwargs(self, inputs, params):
        pad_id = self.tokenizer.pad_token_id
        return {
            **inputs,
            **params.to_generate_kwargs(),
            "pad_token_id": pad_id if pad_id is not None else self.tokenizer.eos_token_id,
        }

    @torch.inference_mode()
    def generate(self, messages, params=GenerationParams()):
        inputs = self.build_inputs(messages)
        with self._lock:
            output = self.model.generate(**self._generate_kwargs(inputs, params))
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(self, messages, params=GenerationParams()):
        """Yield text chunks as they are generated (used by chat.py)."""
        inputs = self.build_inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        stop = threading.Event()
        kwargs = {
            **self._generate_kwargs(inputs, params),
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([_StopOnEvent(stop)]),
        }

        def run():
            with self._lock, torch.inference_mode():
                self.model.generate(**kwargs)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            yield from streamer
        finally:
            # Runs on normal completion and when the caller stops iterating early.
            stop.set()
            thread.join()

    @torch.inference_mode()
    def completion_loss(self, prompt_messages, answer):
        """Mean per-token NLL of `answer` given the prompt. Lower = model finds the answer more likely."""
        prompt_ids = self.build_inputs(prompt_messages)["input_ids"]
        full_ids = self.tokenizer.apply_chat_template(
            prompt_messages + [{"role": "assistant", "content": answer}],
            return_tensors="pt",
            return_dict=True,
            **CHAT_TEMPLATE_KWARGS,
        )["input_ids"].to(self.device)
        labels = full_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100  # score only the answer tokens
        return self.model(input_ids=full_ids, labels=labels).loss.item()
