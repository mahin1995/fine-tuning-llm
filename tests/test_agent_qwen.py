"""The agent running on a real (tiny) Qwen3 model through QwenBackend."""
import warnings

from agent import Agent
from agent.backends.qwen import QwenBackend
from agent.builtin_tools import default_registry
from qwen_ft.modeling.chat_model import ChatModel
from qwen_ft.modeling.params import GenerationParams


def test_tools_are_rendered_into_the_prompt(tiny_model_dir):
    chat = ChatModel.load(tiny_model_dir, device="cpu")
    tools = default_registry().schemas()
    prompt = chat.tokenizer.decode(chat.build_inputs([{"role": "user", "content": "hi"}], tools)["input_ids"][0])
    assert "<tools>" in prompt and '"name": "calculator"' in prompt
    assert prompt.endswith("<think>\n\n</think>\n\n")  # still non-thinking mode, same as training


def test_tool_messages_render_like_qwen(tiny_model_dir):
    chat = ChatModel.load(tiny_model_dir, device="cpu")
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "calculator", "arguments": {"expression": "2+2"}}}]},
        {"role": "tool", "name": "calculator", "content": "4"},
    ]
    prompt = chat.tokenizer.decode(chat.build_inputs(messages, default_registry().schemas())["input_ids"][0])
    assert '<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call>' in prompt
    assert "<tool_response>\n4\n</tool_response>" in prompt


def test_agent_runs_end_to_end_on_tiny_model(tiny_model_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the special-token warning must not fire for Qwen-style tokenizers
        backend = QwenBackend.from_path(tiny_model_dir, GenerationParams(temperature=0.0, max_new_tokens=8),
                                        device="cpu")
    result = Agent(backend, default_registry(), max_steps=2).run("What is 2 + 2?")
    # A random tiny model won't produce a sensible answer; the loop must still terminate cleanly.
    assert result.stop_reason in ("answer", "max_steps")
    assert result.messages[0] == {"role": "user", "content": "What is 2 + 2?"}
