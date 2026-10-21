from datetime import datetime, timezone

import pytest

from agent.builtin_tools import (
    KnowledgeBase,
    calculator,
    default_registry,
    make_clock_tool,
    make_search_tool,
)
from agent.tools import ToolError, ToolRegistry, make_tool, schema_from_signature
from conftest import TRAIN_DATA
from finetune.data.io import load_conversations


def test_schema_from_signature():
    def f(query: str, top_k: int = 3, exact: bool = False, weight: float = 1.0):
        """doc"""

    schema = schema_from_signature(f, {"query": "what to find"})
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"] == {"type": "string", "description": "what to find"}
    assert [schema["properties"][k]["type"] for k in ("top_k", "exact", "weight")] == ["integer", "boolean", "number"]


def test_schema_rejects_unsupported_types():
    def f(x: set):
        """doc"""

    with pytest.raises(TypeError):
        schema_from_signature(f)


def test_tool_needs_description():
    def f(x: str):
        pass

    with pytest.raises(ValueError):
        make_tool(f)


@pytest.fixture
def registry():
    reg = ToolRegistry()

    @reg.tool(params={"a": "first", "b": "second"})
    def add(a: int, b: int = 0) -> int:
        """Add two integers."""
        return a + b

    return reg


def test_registry_executes_and_serializes(registry):
    assert registry.execute("add", {"a": 2, "b": 3}) == "5"
    assert registry.schemas()[0]["function"]["name"] == "add"
    assert "add" in registry and len(registry) == 1


@pytest.mark.parametrize("name, args, error", [
    ("nope", {}, "unknown tool"),
    ("add", {}, "missing required"),
    ("add", {"a": 1, "c": 2}, "unknown argument"),
    ("add", {"a": "1"}, "type integer"),
    ("add", {"a": True}, "type integer"),  # bool is not an int for tool arguments
])
def test_registry_validates_before_running(registry, name, args, error):
    with pytest.raises(ToolError, match=error):
        registry.execute(name, args)


def test_duplicate_registration_rejected(registry):
    with pytest.raises(ValueError):
        registry.register(make_tool(calculator, name="add"))


@pytest.mark.parametrize("expr, expected", [
    ("2 + 3 * 4", "14"), ("(17 * 23) / 4", "97.75"), ("10 / 2", "5"), ("-2 ** 2", "-4"), ("7 // 2 + 7 % 2", "4"),
])
def test_calculator(expr, expected):
    assert calculator(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')", "x + 1", "len('a')", "1 if 1 else 2", "[1, 2]", "'a' * 3",
    "2 ** 1000", "((10 ** 100) ** 100) ** 100", "1 / 0", "1e308 * 1e308 ** 2", "(-8) ** 0.5", "1 +", "1" * 201,
])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError):
        calculator(expr)


def test_clock_uses_injected_time():
    tool = make_clock_tool(lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    assert tool.func() == "2026-01-02T03:04:05+00:00"


def test_knowledge_base_search_finds_relevant_entry():
    kb = KnowledgeBase.from_conversations(load_conversations(TRAIN_DATA))
    question, _ = kb.search("N+1 problem JPA", top_k=1)[0]
    assert "N+1" in question
    question, _ = kb.search("transactional propagation REQUIRES_NEW", top_k=1)[0]
    assert "REQUIRES_NEW" in question
    assert kb.search("the and of") == []


def test_search_tool_formats_results_and_validates_top_k():
    tool = make_search_tool(KnowledgeBase([("What is a bean?", "An object managed by Spring.")]))
    assert tool.func("spring bean") == "Q: What is a bean?\nA: An object managed by Spring."
    assert tool.func("kubernetes") == "no matching entries"
    with pytest.raises(ToolError):
        tool.func("bean", top_k=50)


def test_default_registry():
    assert default_registry().names == ["calculator", "current_time"]
    kb = KnowledgeBase([("q", "a")])
    assert default_registry(kb).names == ["calculator", "current_time", "search_knowledge_base"]
