"""Safe, side-effect-free tools that are useful out of the box.

    calculator              arithmetic via a whitelisted AST (no eval)
    current_time            local date/time (clock injectable for tests)
    search_knowledge_base   hybrid (BM25 + embeddings) search over Q&A pairs (e.g. the training data)

Each `make_*` function returns a Tool; `default_registry` wires them together.
"""
import ast
import operator
from datetime import datetime
from typing import Callable

from agent.tools import Tool, ToolError, ToolRegistry, make_tool
from memory import Embedder, HashingEmbedder, HybridRetriever, terms

# ---------------------------------------------------------------- calculator

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 100
MAX_RESULT_BITS = 10_000  # ~3000 digits; stops nested powers like ((10**100)**100)**100 from exhausting memory


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ToolError(f"exponent larger than {MAX_EXPONENT} is not allowed")
            if isinstance(left, int) and isinstance(right, int) and left.bit_length() * right > MAX_RESULT_BITS:
                raise ToolError("result too large")
        return _BIN_OPS[type(node.op)](left, right)
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression with + - * / // % ** and parentheses. Example: (17 * 23) / 4"""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ToolError(f"invalid expression: {e.msg}") from e
    try:
        result = _eval_node(tree)
    except ZeroDivisionError as e:
        raise ToolError("division by zero") from e
    except OverflowError as e:
        raise ToolError("result too large") from e
    if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
        raise ToolError("result too large")
    if isinstance(result, complex):  # e.g. (-8) ** 0.5
        raise ToolError("result is not a real number")
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def make_calculator_tool() -> Tool:
    return make_tool(calculator, params={"expression": "arithmetic expression, e.g. 2 * (3 + 4)"})


# -------------------------------------------------------------- current time

def make_clock_tool(now: Callable[[], datetime] = lambda: datetime.now().astimezone()) -> Tool:
    def current_time() -> str:
        """Return the current local date and time in ISO 8601 format."""
        return now().isoformat(timespec="seconds")

    return make_tool(current_time)


# ----------------------------------------------------------- knowledge base

class KnowledgeBase:
    """Q&A pairs searched with BM25 plus embeddings (memory.HybridRetriever, Reciprocal Rank Fusion).

    The default embedder is offline character hashing (finds rewordings that keep the key words);
    pass any memory.Embedder, e.g. an HttpEmbedder for Ollama / vLLM / OpenAI, for real semantic
    search. BM25 sees the question twice (field boost: the question states the topic) plus the
    answer; vectors are built from the question alone (long answers would dilute the topic)."""

    def __init__(self, entries: list[tuple[str, str]], embedder: Embedder | None = None,
                 min_similarity: float = 0.3):
        self.entries = entries
        self._retriever = HybridRetriever(
            [(str(i), f"{q}\n{q}\n{a}") for i, (q, a) in enumerate(entries)],
            embedder or HashingEmbedder(), fallback=HashingEmbedder(), min_similarity=min_similarity,
            vector_texts=[q for q, _ in entries],
        )

    @classmethod
    def from_conversations(cls, conversations, embedder: Embedder | None = None):
        """Build from chat conversations: the last user question and the final answer."""
        return cls([(m[-2]["content"], m[-1]["content"]) for m in conversations], embedder)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, str]]:
        if not terms(query):  # only stopwords: nothing to look for
            return []
        return [self.entries[int(hit.id)] for hit in self._retriever.search(query, top_k)]


def make_search_tool(kb: KnowledgeBase, max_top_k: int = 5) -> Tool:
    def search_knowledge_base(query: str, top_k: int = 3) -> str:
        """Search the project's Java / Spring Boot Q&A knowledge base. Returns the most relevant Q&A pairs."""
        if not 1 <= top_k <= max_top_k:
            raise ToolError(f"top_k must be between 1 and {max_top_k}")
        hits = kb.search(query, top_k)
        if not hits:
            return "no matching entries"
        return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in hits)

    return make_tool(search_knowledge_base, params={
        "query": "keywords describing the topic, e.g. 'transactional propagation'",
        "top_k": f"number of results, 1-{max_top_k}",
    })


# ----------------------------------------------------------------- registry

def default_registry(kb: KnowledgeBase | None = None, now=None) -> ToolRegistry:
    tools = [make_calculator_tool(), make_clock_tool(now) if now else make_clock_tool()]
    if kb is not None:
        tools.append(make_search_tool(kb))
    return ToolRegistry(tools)
