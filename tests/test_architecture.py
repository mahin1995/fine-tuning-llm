"""Enforce the package boundaries, so loose coupling doesn't erode over time.

Each check imports modules in a fresh interpreter and inspects sys.modules.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT

APPS = ROOT / "apps"


def package_dir(name):
    """Packages live at the repo root (finetune) or under apps/ (agent, ops_crew, refiner)."""
    path = APPS / name if (APPS / name).is_dir() else ROOT / name
    assert path.is_dir(), f"package {name} not found"  # a moved package must not make a rule pass vacuously
    return path


def imported_modules(*modules):
    code = (f"import sys; import {', '.join(modules)}; "
            "print('\\n'.join(sorted(m for m in sys.modules)))")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(APPS)])}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def heavy(mods):
    return {m for m in mods if m.split(".")[0] in ("torch", "transformers", "trl", "peft", "datasets")}


def test_agent_core_is_independent_of_finetune_and_ml_libraries():
    mods = imported_modules("agent", "agent.loop", "agent.tools", "agent.parser", "agent.builtin_tools")
    assert not {m for m in mods if m.startswith("finetune")}
    assert not heavy(mods)


@pytest.mark.parametrize("module", [
    "finetune.config", "finetune.data.io", "finetune.data.transforms", "finetune.modeling.params",
    "finetune.evaluation.evaluator", "finetune.evaluation.report", "finetune.serving.app",
    "finetune.training.options", "finetune.cli.train", "finetune.__main__",
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


# package -> finetune packages it may import (cli is the composition root and may import anything)
ALLOWED = {
    "config": set(),
    "data": {"config", "data"},
    "modeling": {"config", "modeling"},
    "training": {"config", "data", "training"},
    "evaluation": {"config", "modeling", "evaluation"},
    "serving": {"data", "modeling", "serving"},
}


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_finetune_layering(package):
    base = ROOT / "finetune"
    files = [base / f"{package}.py"] if (base / f"{package}.py").exists() else list((base / package).rglob("*.py"))
    for path in files:
        for name in _imports_of(path):
            parts = name.split(".")
            if parts[0] == "agent":
                pytest.fail(f"{path.relative_to(ROOT)} imports the agent package")
            if parts[0] == "finetune" and len(parts) > 1:
                assert parts[1] in ALLOWED[package], f"{path.relative_to(ROOT)} must not import {name}"


def test_only_backends_and_entrypoint_import_finetune():
    for path in (APPS / "agent").rglob("*.py"):
        rel = path.relative_to(APPS / "agent").as_posix()
        if rel.startswith("backends/") or rel == "__main__.py":
            continue
        assert not any(n.split(".")[0] == "finetune" for n in _imports_of(path)), f"agent/{rel} imports finetune"


# ------------------------------------------------------------------ ops_crew

OPS_DOMAIN_MODULES = ["ops_crew.schemas", "ops_crew.settings"] + [
    f"ops_crew.domain.{p.stem}" for p in sorted((APPS / "ops_crew" / "domain").glob("*.py")) if p.stem != "__init__"
]


def test_ops_deterministic_layer_imports_no_crewai_or_ml_libraries():
    pytest.importorskip("pydantic_settings")
    mods = imported_modules(*OPS_DOMAIN_MODULES)
    assert not {m for m in mods if m.split(".")[0] in ("crewai", "litellm", "openai", "anthropic")}
    assert not heavy(mods)
    assert not {m for m in mods if m.startswith(("ops_crew.crew", "ops_crew.flow", "agent"))}


def test_only_crew_flow_evals_and_entrypoint_import_crewai():
    allowed = ("crew/", "flow.py", "evals/", "__main__.py")
    for path in (APPS / "ops_crew").rglob("*.py"):
        rel = path.relative_to(APPS / "ops_crew").as_posix()
        if rel.startswith(allowed):
            continue
        assert not any(n.split(".")[0] == "crewai" for n in _imports_of(path)), f"ops_crew/{rel} imports crewai"


def test_ops_domain_only_reuses_finetune_data():
    for path in (APPS / "ops_crew" / "domain").rglob("*.py"):
        for name in _imports_of(path):
            parts = name.split(".")
            if parts[0] == "finetune":
                assert parts[:2] == ["finetune", "data"], f"{path.name} imports {name}"
            assert not name.startswith(("ops_crew.crew", "ops_crew.flow", "ops_crew.evals"))


def test_packages_do_not_import_each_other_sideways():
    for package, forbidden in (("finetune", ("ops_crew", "agent")), ("agent", ("ops_crew",)),
                               ("ops_crew", ("agent",))):
        for path in package_dir(package).rglob("*.py"):
            bad = [n for n in _imports_of(path) if n.split(".")[0] in forbidden]
            assert not bad, f"{path.relative_to(ROOT)} imports {bad}"


# ------------------------------------------------------------------- refiner

THIRD_PARTY_OK_FOR_REFINER = {"pydantic", "pydantic_ai"}


def test_refiner_depends_only_on_pydantic_ai():
    import sys as _sys

    stdlib = set(_sys.stdlib_module_names)
    for path in (APPS / "refiner").rglob("*.py"):
        for name in _imports_of(path):
            top = name.split(".")[0]
            assert top in stdlib or top in THIRD_PARTY_OK_FOR_REFINER or top == "refiner", \
                f"{path.relative_to(ROOT)} imports {name}"


def test_only_the_ops_crew_adapter_imports_refiner():
    for package in ("finetune", "agent", "ops_crew"):
        for path in package_dir(package).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if rel == "apps/ops_crew/refinement.py":
                continue
            assert not any(n.split(".")[0] == "refiner" for n in _imports_of(path)), f"{rel} imports refiner"
