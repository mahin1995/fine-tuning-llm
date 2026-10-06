"""Turn raw crew output into validated Pydantic models, or fail with an exact reason."""
import json
import re

from pydantic import BaseModel, ValidationError

from ops_crew.schemas import ActionProposal, ClassifiedRequest, CrewRun, ResearchFindings, ValidatedRun

# Task name in tasks.yaml -> schema its output must satisfy.
TASK_SCHEMAS: dict[str, type[BaseModel]] = {
    "classify_request": ClassifiedRequest,
    "research_context": ResearchFindings,
    "review_proposal": ActionProposal,
}

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class OutputValidationError(ValueError):
    pass


def extract_json(raw: str) -> dict:
    """Accept a bare JSON object, optionally inside a ```json fence. Nothing else."""
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise OutputValidationError(f"not valid JSON: {e.msg} at position {e.pos}") from e
    if not isinstance(value, dict):
        raise OutputValidationError("JSON output must be an object")
    return value


def parse_output(raw: str, schema: type[BaseModel]):
    data = extract_json(raw)
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}" for err in e.errors())
        raise OutputValidationError(f"{schema.__name__} schema violation: {problems}") from e


def validate_run(run: CrewRun) -> ValidatedRun:
    missing = [name for name in TASK_SCHEMAS if name not in run.outputs]
    if missing:
        raise OutputValidationError(f"missing task output(s): {', '.join(missing)}")
    parsed = {name: parse_output(run.outputs[name], schema) for name, schema in TASK_SCHEMAS.items()}
    return ValidatedRun(
        classification=parsed["classify_request"],
        findings=parsed["research_context"],
        proposal=parsed["review_proposal"],
        tool_calls=run.tool_calls,
    )
