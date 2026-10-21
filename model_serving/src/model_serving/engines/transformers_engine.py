"""TransformersEngine: in-process generation with Hugging Face transformers.

Loads a full model or a LoRA adapter (merged on load). Model loading is deliberately
re-implemented here instead of importing the training code, so the serving image never
depends on it (ARCHITECTURE.md, "duplication accepted on purpose").
"""
import json
import threading
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from model_serving.config import ModelEntry
from model_serving.engines.base import Delta, Generation, GenerationRequest, cut_at_stop

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def pick_device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(setting: str, device: str):
    if setting != "auto":
        return _DTYPES[setting]
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32  # bf16 matmuls are slow on CPU


def template_kwargs_for(entry: ModelEntry, path: Path) -> dict:
    """models.yaml wins; otherwise the fine-tuned model's run_info.json (artifact contract)."""
    if entry.chat_template_kwargs is not None:
        return dict(entry.chat_template_kwargs)
    try:
        info = json.loads((Path(path) / "run_info.json").read_text(encoding="utf-8"))
        return dict(info.get("chat_template_kwargs") or {})
    except (OSError, ValueError):
        return {}


class _StopOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class TransformersEngine:
    def __init__(self, model, tokenizer, *, model_name: str, chat_template_kwargs: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.default_chat_template_kwargs = dict(chat_template_kwargs)
        self._lock = threading.Lock()  # one GPU: one generation at a time

    @classmethod
    def load(cls, entry: ModelEntry, path: Path, device=None) -> "TransformersEngine":
        device = pick_device(device)
        dtype = pick_dtype(entry.dtype, device)
        path = Path(path)
        if (path / "adapter_config.json").is_file():
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(path, dtype=dtype).merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model.to(device).eval()
        return cls(model, tokenizer, model_name=entry.name, chat_template_kwargs=template_kwargs_for(entry, path))

    # ------------------------------------------------------------------ internals
    def _inputs(self, req: GenerationRequest):
        return self.tokenizer.apply_chat_template(
            req.messages, tools=req.tools, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, **req.chat_template_kwargs,
        ).to(self.model.device)

    def _generate_kwargs(self, inputs, req: GenerationRequest):
        pad_id = self.tokenizer.pad_token_id
        kwargs = {
            **inputs,
            "max_new_tokens": req.max_tokens,
            "pad_token_id": pad_id if pad_id is not None else self.tokenizer.eos_token_id,
        }
        if req.temperature > 0:
            kwargs.update(do_sample=True, temperature=req.temperature, top_p=req.top_p)
        else:
            kwargs.update(do_sample=False)
        if req.stop:
            kwargs.update(stop_strings=req.stop, tokenizer=self.tokenizer)
        return kwargs

    def _finish(self, new_tokens, req, stopped: bool) -> str:
        if stopped:
            return "stop"
        ended_on_eos = len(new_tokens) > 0 and int(new_tokens[-1]) == self.tokenizer.eos_token_id
        return "length" if len(new_tokens) >= req.max_tokens and not ended_on_eos else "stop"

    # ---------------------------------------------------------------------- port
    @torch.inference_mode()
    def generate(self, req: GenerationRequest) -> Generation:
        inputs = self._inputs(req)
        prompt_len = inputs["input_ids"].shape[1]
        with self._lock:
            output = self.model.generate(**self._generate_kwargs(inputs, req))
        new_tokens = output[0, prompt_len:]
        text, stopped = cut_at_stop(self.tokenizer.decode(new_tokens, skip_special_tokens=True), req.stop)
        return Generation(text.strip(), prompt_len, len(new_tokens), self._finish(new_tokens, req, stopped))

    def stream(self, req: GenerationRequest):
        inputs = self._inputs(req)
        prompt_len = inputs["input_ids"].shape[1]
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancel = threading.Event()
        result = {}

        def run():
            with self._lock, torch.inference_mode():
                kwargs = self._generate_kwargs(inputs, req)
                kwargs.update(streamer=streamer, stopping_criteria=StoppingCriteriaList([_StopOnEvent(cancel)]))
                result["output"] = self.model.generate(**kwargs)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        # Hold back enough characters that a stop string split across chunks is never emitted.
        holdback = max((len(s) for s in req.stop), default=1) - 1
        text, sent, stopped = "", 0, False
        try:
            for chunk in streamer:
                text += chunk
                cut, stopped = cut_at_stop(text, req.stop)
                safe = len(cut) if stopped else max(len(text) - holdback, sent)
                if safe > sent:
                    yield Delta(text[sent:safe])
                    sent = safe
                if stopped:
                    break
            if not stopped and len(text) > sent:
                yield Delta(text[sent:])
        finally:
            cancel.set()
            thread.join()
        new_tokens = result["output"][0, prompt_len:] if "output" in result else []
        yield Delta("", self._finish(new_tokens, req, stopped), prompt_len, len(new_tokens))
