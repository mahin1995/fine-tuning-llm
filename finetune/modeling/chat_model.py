"""Chat generation that renders prompts exactly as training did."""
import threading

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from finetune.config import CHAT_TEMPLATE_KWARGS
from finetune.modeling.loading import load_model_and_tokenizer
from finetune.modeling.params import GenerationParams


class _StopOnEvent(StoppingCriteria):
    """Lets the consumer of stream() abort generation early (e.g. Ctrl-C in the chat CLI)."""

    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class ChatModel:
    """Thin wrapper that applies the chat template consistently with training.

    `tools` (optional, OpenAI-style function schemas) is passed through to the chat
    template, which is how tool-calling agents describe the available tools.
    """

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

    def build_inputs(self, messages, tools=None):
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
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
    def generate(self, messages, params=GenerationParams(), tools=None):
        inputs = self.build_inputs(messages, tools)
        with self._lock:
            output = self.model.generate(**self._generate_kwargs(inputs, params))
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(self, messages, params=GenerationParams(), tools=None):
        """Yield text chunks as they are generated."""
        inputs = self.build_inputs(messages, tools)
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
