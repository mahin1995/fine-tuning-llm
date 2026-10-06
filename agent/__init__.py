"""A small, model-agnostic tool-calling agent loop.

    backend        ChatBackend protocol: complete(messages, tools) -> str
    parser         parse <tool_call>{json}</tool_call> blocks from model output
    tools          Tool, ToolRegistry (schema generation + argument validation)
    loop           Agent.run(question) -> AgentResult
    builtin_tools  calculator, current_time, search_knowledge_base
    backends/qwen  adapter for this repo's fine-tuned Qwen models (the only qwen_ft import)

The core modules import nothing from qwen_ft, torch or transformers, so the loop can
be tested with a scripted fake backend and reused with any model.

    python -m agent --model outputs/qwen3-ft "What is 17 * 23?"
"""
from agent.backend import ChatBackend
from agent.loop import Agent, AgentResult, Step
from agent.parser import ToolCall, parse_reply
from agent.tools import Tool, ToolError, ToolRegistry, make_tool

__all__ = ["Agent", "AgentResult", "ChatBackend", "Step", "Tool", "ToolCall", "ToolError", "ToolRegistry",
           "make_tool", "parse_reply"]
