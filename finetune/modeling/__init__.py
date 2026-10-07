"""Model loading and chat generation.

    params      GenerationParams (pure dataclass, no torch import)
    loading     device/dtype selection, load full model or LoRA adapter
    chat_model  ChatModel: chat-template-consistent generate / stream / completion_loss

Import from the submodules directly so callers that only need GenerationParams don't pull in torch.
"""
