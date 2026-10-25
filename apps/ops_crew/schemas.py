"""Typed contracts between the probabilistic and deterministic layers.

Every crew output is parsed into one of these models; anything that doesn't
validate is rejected (and retried / escalated) before any decision is made.
"""
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_ID_PATTERN = r"^[0-9a-f]{12}$"
STEP_ID_PATTERN = r"^s[1-9][0-9]?$"  # s1 .. s99; ids are echoed in feedback, so they must stay this simple


class Intent(StrEnum):
    ASK = "ask"                        # answer a Java / Spring Boot question (no side effects)
    DATASET_STATS = "dataset_stats"    # report dataset statistics (no side effects)
    ADD_EXAMPLE = "add_example"        # append a Q&A pair to the training data
    REMOVE_EXAMPLE = "remove_example"  # delete a training example (destructive)
    UNKNOWN = "unknown"                # out of scope / can't tell


class Role(StrEnum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActionParams(_Strict):
    question: str | None = Field(default=None, max_length=2000)
    answer: str | None = Field(default=None, max_length=8000)
    example_id: str | None = Field(default=None, pattern=EXAMPLE_ID_PATTERN)


class ClassifiedRequest(_Strict):
    """Output of the Intent Classifier."""
    intent: Intent
    params: ActionParams = ActionParams()
    confidence: float = Field(ge=0.0, le=1.0)
    clarifying_question: str | None = None


class ResearchFindings(_Strict):
    """Output of the Researcher: context gathered with read-only tools."""
    summary: str
    related_example_ids: list[str] = Field(default_factory=list)
    duplicate_of: str | None = Field(default=None, pattern=EXAMPLE_ID_PATTERN)
    overlaps_eval_set: bool = False
    draft_answer: str | None = None


class ActionProposal(_Strict):
    """Output of the Reviewer: the crew's final, reviewed proposal. Never executed directly."""
    intent: Intent
    params: ActionParams = ActionParams()
    answer: str | None = Field(default=None, max_length=8000)  # for intent=ask
    rationale: str
    issues: list[str] = Field(default_factory=list)  # gaps / contradictions the reviewer found
    confidence: float = Field(ge=0.0, le=1.0)
    clarifying_question: str | None = None


StepId = Annotated[str, Field(pattern=STEP_ID_PATTERN)]


class PlanStep(_Strict):
    """One step of a plan: a part of the user's request, copied verbatim, and its intent."""
    id: StepId
    intent: Intent
    excerpt: str = Field(min_length=1, max_length=4000)  # verbatim from the request: the planner writes no content
    depends_on: list[StepId] = Field(default_factory=list)  # earlier steps that must complete first


class Plan(_Strict):
    """Output of the Planner: the request split into ordered steps. Never executed directly;
    domain/planning.py checks it and every step still goes through the normal pipeline."""
    steps: list[PlanStep] = Field(min_length=1, max_length=20)  # the real limit is a policy (max_plan_steps)
    rationale: str = ""


class ToolCallRecord(_Strict):
    agent: str
    tool: str
    arguments: dict


class CrewRun(BaseModel):
    """Raw result of one crew execution, before validation."""
    outputs: dict[str, str]  # task name -> raw text
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class ValidatedRun(BaseModel):
    classification: ClassifiedRequest
    findings: ResearchFindings
    proposal: ActionProposal
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
