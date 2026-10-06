"""Sampling settings. Deliberately free of torch so serving/evaluation can import it cheaply."""
from dataclasses import dataclass


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
