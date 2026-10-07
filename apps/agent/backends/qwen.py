"""ChatBackend for the fine-tuned Qwen models in this repo (finetune.modeling.ChatModel).

This adapter is the single bridge between `agent` and `finetune`; the agent loop
itself never imports finetune or torch.
"""
import warnings

from finetune.modeling.params import GenerationParams

# Greedy by default: tool calls must be exact JSON, and sampling makes small models sloppier.
AGENT_PARAMS = GenerationParams(temperature=0.0, max_new_tokens=512)


class QwenBackend:
    def __init__(self, chat_model, params: GenerationParams = AGENT_PARAMS):
        self.chat = chat_model
        self.params = params
        # ChatModel decodes with skip_special_tokens=True. If the tokenizer marks the
        # tool-call tags as special, they'd be stripped and no tool call could be parsed.
        if "<tool_call>" in getattr(chat_model.tokenizer, "all_special_tokens", []):
            warnings.warn("<tool_call> is a special token in this tokenizer; tool calls will be stripped "
                          "from decoded output and the agent will never see them", stacklevel=2)

    @classmethod
    def from_path(cls, model_path, params: GenerationParams = AGENT_PARAMS, device=None):
        from finetune.modeling.chat_model import ChatModel

        return cls(ChatModel.load(model_path, device=device), params)

    @property
    def name(self):
        return self.chat.name

    def complete(self, messages, tools):
        return self.chat.generate(messages, self.params, tools=tools or None)
