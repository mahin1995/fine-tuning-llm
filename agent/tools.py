"""Tool definitions and a registry that validates arguments before running anything.

Model output is untrusted input: every call is checked against the tool's JSON
schema (required/unknown keys, basic types) before the Python function is invoked.
"""
import inspect
import json
import typing
from dataclasses import dataclass
from typing import Any, Callable

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


class ToolError(Exception):
    """An error whose message is safe and useful to show to the model."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema: {"type": "object", "properties": {...}, "required": [...]}
    func: Callable[..., Any]

    def schema(self):
        """OpenAI-style function schema, the format chat templates expect."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def schema_from_signature(func, param_descriptions=None):
    """Build a JSON schema from type hints. Supports str/int/float/bool/list/dict parameters."""
    param_descriptions = param_descriptions or {}
    hints = typing.get_type_hints(func)
    properties, required = {}, []
    for name, param in inspect.signature(func).parameters.items():
        py_type = hints.get(name, str)
        if py_type not in _JSON_TYPES:
            raise TypeError(f"{func.__name__}.{name}: unsupported parameter type {py_type!r}")
        prop = {"type": _JSON_TYPES[py_type]}
        if name in param_descriptions:
            prop["description"] = param_descriptions[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def make_tool(func, name=None, description=None, params=None):
    """Create a Tool from a typed function; the docstring is the default description."""
    description = description or inspect.getdoc(func)
    if not description:
        raise ValueError(f"tool {func.__name__} needs a description (docstring or description=...)")
    return Tool(name or func.__name__, description, schema_from_signature(func, params), func)


def _type_ok(value, json_type):
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    expected = {"string": str, "boolean": bool, "array": list, "object": dict}.get(json_type)
    return expected is None or isinstance(value, expected)


def validate_arguments(tool, arguments):
    schema = tool.parameters
    properties = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in arguments]
    if missing:
        raise ToolError(f"{tool.name}: missing required argument(s): {', '.join(missing)}")
    unknown = [k for k in arguments if k not in properties]
    if unknown:
        raise ToolError(f"{tool.name}: unknown argument(s): {', '.join(unknown)}; "
                        f"expected: {', '.join(properties) or 'none'}")
    for key, value in arguments.items():
        json_type = properties[key].get("type")
        if not _type_ok(value, json_type):
            raise ToolError(f"{tool.name}: argument {key!r} must be of type {json_type}")


class ToolRegistry:
    def __init__(self, tools=()):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name=None, description=None, params=None):
        """Decorator: @registry.tool(params={"query": "what to search for"})"""
        def decorator(func):
            self.register(make_tool(func, name, description, params))
            return func
        return decorator

    @property
    def names(self):
        return list(self._tools)

    def __contains__(self, name):
        return name in self._tools

    def __len__(self):
        return len(self._tools)

    def schemas(self):
        return [t.schema() for t in self._tools.values()]

    def execute(self, name, arguments) -> str:
        """Run a tool and return its result as text. Raises ToolError for anything the model got wrong."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}; available tools: {', '.join(self.names) or 'none'}")
        validate_arguments(tool, arguments)
        result = tool.func(**arguments)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
