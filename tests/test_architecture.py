"""Enforce the package boundaries, so loose coupling doesn't erode over time.

Each check imports modules in a fresh interpreter and inspects sys.modules.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT


def imported_modules(*modules):
    code = (f"import sys; import {', '.join(modules)}; "
            "print('\\n'.join(sorted(m for m in sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def heavy(mods):
    return {m for m in mods if m.split(".")[0] in ("torch", "transformers", "trl", "peft", "datasets")}


def test_agent_core_is_independent_of_qwen_ft_and_ml_libraries():
    mods = imported_modules("agent", "agent.loop", "agent.tools", "agent.parser", "agent.builtin_tools")
    assert not {m for m in mods if m.startswith("qwen_ft")}
    assert not heavy(mods)


@pytest.mark.parametrize("module", [
    "qwen_ft.config", "qwen_ft.data.io", "qwen_ft.data.transforms", "qwen_ft.modeling.params",
    "qwen_ft.evaluation.evaluator", "qwen_ft.evaluation.report", "qwen_ft.serving.app",
    "qwen_ft.training.options", "qwen_ft.cli.train", "qwen_ft.__main__",
])
def test_light_modules_do_not_import_ml_libraries(module):
    assert not heavy(imported_modules(module)), f"{module} pulls in torch/transformers"


def _imports_of(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


# package -> qwen_ft packages it may import (cli is the composition root and may import anything)
ALLOWED = {
    "config": set(),
    "data": {"config", "data"},
    "modeling": {"config", "modeling"},
    "training": {"config", "data", "training"},
    "evaluation": {"config", "modeling", "evaluation"},
    "serving": {"data", "modeling", "serving"},
}


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_qwen_ft_layering(package):
    base = ROOT / "qwen_ft"
    files = [base / f"{package}.py"] if (base / f"{package}.py").exists() else list((base / package).rglob("*.py"))
    for path in files:
        for name in _imports_of(path):
            parts = name.split(".")
            if parts[0] == "agent":
                pytest.fail(f"{path.relative_to(ROOT)} imports the agent package")
            if parts[0] == "qwen_ft" and len(parts) > 1:
                assert parts[1] in ALLOWED[package], f"{path.relative_to(ROOT)} must not import {name}"


def test_only_backends_and_entrypoint_import_qwen_ft():
    for path in (ROOT / "agent").rglob("*.py"):
        rel = path.relative_to(ROOT / "agent").as_posix()
        if rel.startswith("backends/") or rel == "__main__.py":
            continue
        assert not any(n.split(".")[0] == "qwen_ft" for n in _imports_of(path)), f"agent/{rel} imports qwen_ft"
