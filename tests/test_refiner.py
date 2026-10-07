"""refiner: self-correction and reflection, offline with scripted PydanticAI FunctionModels."""
import json

import pytest

pytest.importorskip("pydantic_ai")

import pydantic_ai.models  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from refiner import (  # noqa: E402
    Critique,
    ModelSpec,
    RefinerError,
    build_model,
    make_critic,
    make_reviser,
    reflect,
    self_correct,
    text_function_model,
)
from refiner.models import FALLBACK_ON, to_chat_messages, with_fallbacks  # noqa: E402

pydantic_ai.models.ALLOW_MODEL_REQUESTS = False  # any accidental real request fails the test


class Answer(BaseModel):
    answer: str = Field(min_length=10)
    confidence: float = Field(ge=0, le=1)


class Scripted:
    """A FunctionModel that returns the given outputs in order, in the shape the output mode expects."""

    __name__ = "scripted"  # FunctionModel reads the callable's name

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []  # (messages, info) per model call

    def __call__(self, messages, info):
        self.calls.append((messages, info))
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        if isinstance(out, str):
            return ModelResponse(parts=[TextPart(out)])
        if info.output_tools:  # tool mode
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, out)])
        return ModelResponse(parts=[TextPart(json.dumps(out))])

    def model(self):
        return FunctionModel(self, model_name="scripted")

    def last_prompt_text(self):
        messages, _ = self.calls[-1]
        return json.dumps([str(p) for m in messages for p in m.parts])


GOOD = {"answer": "REQUIRES_NEW suspends the outer transaction.", "confidence": 0.9}


# ------------------------------------------------------------ self_correct

@pytest.mark.parametrize("mode", ["tool", "native", "prompted"])
def test_first_valid_output_is_accepted(mode):
    script = Scripted(GOOD)
    result = self_correct(script.model(), Answer, "q", output_mode=mode)
    assert result.output == Answer(**GOOD)
    assert (result.attempts, result.feedback) == (1, [])


@pytest.mark.parametrize("mode", ["tool", "prompted"])
def test_schema_error_is_fed_back_and_fixed(mode):
    script = Scripted({"answer": "short", "confidence": 0.9}, GOOD)
    result = self_correct(script.model(), Answer, "q", output_mode=mode)
    assert result.output.answer == GOOD["answer"]
    assert result.attempts == 2
    assert "String should have at least 10 characters" in result.feedback[0]
    assert "String should have at least 10 characters" in script.last_prompt_text()  # the model saw it


def test_check_problems_are_fed_back_verbatim():
    script = Scripted({"answer": "It is fine, trust me.", "confidence": 0.9}, GOOD)

    def no_hand_waving(a: Answer) -> list[str]:
        return ["answer contains no technical content; explain what REQUIRES_NEW does"] if "trust me" in a.answer else []

    result = self_correct(script.model(), Answer, "q", checks=[no_hand_waving])
    assert result.output.answer == GOOD["answer"]
    assert result.feedback[0].startswith(
        "Your previous output has these problems. Fix all of them and keep everything else unchanged:\n"
        "- answer contains no technical content; explain what REQUIRES_NEW does")


def test_every_check_runs_and_all_problems_are_reported_together():
    script = Scripted({"answer": "bad answer here", "confidence": 0.1}, GOOD)
    checks = [lambda a: ["problem A"] if "bad" in a.answer else [],
              lambda a: ["problem B"] if a.confidence < 0.5 else []]
    result = self_correct(script.model(), Answer, "q", checks=checks)
    assert "- problem A\n- problem B" in result.feedback[0]


def test_budget_exhaustion_raises_refiner_error():
    bad = {"answer": "x", "confidence": 2}
    script = Scripted(bad, bad, bad, GOOD)
    with pytest.raises(RefinerError, match="Exceeded maximum output retries \\(2\\)"):
        self_correct(script.model(), Answer, "q", max_retries=2)
    assert len(script.calls) == 3  # 1 attempt + 2 retries, the GOOD answer is never requested


def test_prose_instead_of_json_is_corrected_in_prompted_mode():
    script = Scripted("Sure! REQUIRES_NEW starts a new transaction.", GOOD)
    result = self_correct(script.model(), Answer, "q", output_mode="prompted")
    assert result.attempts == 2 and result.output.answer == GOOD["answer"]


def test_provider_failure_raises_refiner_error():
    script = Scripted(ConnectionError("connection refused"))
    with pytest.raises(RefinerError, match="ConnectionError: connection refused"):
        self_correct(script.model(), Answer, "q")


# ----------------------------------------------------------------- models

def test_fallback_model_switches_on_network_errors():
    down, up = Scripted(ConnectionError("down")), Scripted(GOOD)
    result = self_correct(with_fallbacks([down.model(), up.model()]), Answer, "q")
    assert result.output.answer == GOOD["answer"]
    assert len(down.calls) == 1 and len(up.calls) == 1
    assert ConnectionError in FALLBACK_ON


def test_local_model_specs_build_without_network():
    ollama = build_model(ModelSpec(provider="ollama", model="qwen2.5:7b-instruct"))
    vllm = build_model(ModelSpec(provider="openai_compatible", model="Qwen/Qwen2.5-7B-Instruct",
                                 base_url="http://localhost:8001/v1"))
    assert (ollama.model_name, ollama.system) == ("qwen2.5:7b-instruct", "ollama")
    assert vllm.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert ollama.settings["temperature"] == 0.0


def test_text_function_model_gets_schema_instructions_and_feedback():
    """A plain `messages -> text` model (like an in-process fine-tuned one) works in prompted mode."""
    seen = []
    replies = iter(['{"answer": "short", "confidence": 0.5}', json.dumps(GOOD)])

    def generate(chat):
        seen.append(chat)
        return next(replies)

    result = self_correct(text_function_model(generate), Answer, "What is REQUIRES_NEW?",
                          instructions="You are precise.", output_mode="prompted")
    assert result.output.answer == GOOD["answer"]
    first = seen[0]
    assert first[0]["role"] == "system" and first[0]["content"].startswith("You are precise.")
    assert "Always respond with a JSON object" in first[0]["content"]  # schema from PromptedOutput
    assert first[1] == {"role": "user", "content": "What is REQUIRES_NEW?"}
    assert [m["role"] for m in seen[1]] == ["system", "user", "assistant", "user"]
    assert "String should have at least 10 characters" in seen[1][-1]["content"]


def test_to_chat_messages_merges_system_prompts():
    assert to_chat_messages([], "only instructions") == [{"role": "system", "content": "only instructions"}]


# --------------------------------------------------------------- reflect

class Draft(BaseModel):
    text: str


def critic_from(scores, grounded=None):
    """Critic returning the given scores in order; records what it saw."""
    seen, scores, grounded = [], list(scores), list(grounded or [True] * len(scores))

    def critique(d):
        seen.append(d.text)
        return Critique(score=scores.pop(0), grounded=grounded.pop(0), issues=[f"issue with {d.text}"])

    critique.seen = seen
    return critique


def reviser_from(texts):
    texts = list(texts)

    def revise(d, verdict):
        return Draft(text=texts.pop(0))

    return revise


def test_good_draft_is_accepted_without_revision():
    result = reflect(Draft(text="v0"), critique=critic_from([0.9]), revise=reviser_from([]))
    assert (result.best.text, result.accepted, result.stop_reason, len(result.rounds)) == ("v0", True, "accepted", 1)


def test_revision_until_accepted():
    result = reflect(Draft(text="v0"), critique=critic_from([0.4, 0.6, 0.8]), revise=reviser_from(["v1", "v2"]))
    assert result.best.text == "v2" and result.accepted and result.stop_reason == "accepted"
    assert [(r.round, r.output.text, r.score) for r in result.rounds] == [(0, "v0", 0.4), (1, "v1", 0.6), (2, "v2", 0.8)]


def test_flip_flop_keeps_the_best_version():
    # v1 is better than v0, v2 is worse: v1 must win, and the score drop stops the loop
    result = reflect(Draft(text="v0"), critique=critic_from([0.5, 0.65, 0.3]), revise=reviser_from(["v1", "v2"]),
                     max_rounds=3)
    assert (result.best.text, result.best_score, result.accepted) == ("v1", 0.65, False)
    assert result.stop_reason == "no_improvement"


def test_max_rounds_and_repeated_output():
    result = reflect(Draft(text="v0"), critique=critic_from([0.3, 0.4, 0.5]), revise=reviser_from(["v1", "v2"]),
                     max_rounds=2)
    assert (result.best.text, result.stop_reason) == ("v2", "max_rounds")
    same = reflect(Draft(text="v0"), critique=critic_from([0.3]), revise=reviser_from(["v0"]))
    assert (same.best.text, same.stop_reason) == ("v0", "repeated_output")


def test_ungrounded_draft_is_never_accepted():
    result = reflect(Draft(text="v0"), critique=critic_from([0.95], grounded=[False]), revise=reviser_from(["v0"]))
    assert result.accepted is False
    assert result.rounds[0].score == 0.5


def test_critic_or_reviser_failure_returns_best_so_far():
    def broken_critic(d):
        raise RefinerError("critic down")

    result = reflect(Draft(text="v0"), critique=broken_critic, revise=reviser_from([]))
    assert (result.best.text, result.best_score, result.stop_reason) == ("v0", None, "critic_failed")

    def broken_reviser(d, v):
        raise RefinerError("reviser down")

    result = reflect(Draft(text="v0"), critique=critic_from([0.4]), revise=broken_reviser)
    assert (result.best.text, result.best_score, result.stop_reason) == ("v0", 0.4, "reviser_failed")


def test_make_critic_and_reviser_with_models():
    critic_script = Scripted({"score": 0.4, "grounded": True, "issues": ["too vague"]},
                             {"score": 0.85, "grounded": True, "issues": []})
    reviser_script = Scripted({"text": "short"}, {"text": "REQUIRES_NEW suspends the caller's transaction."})
    critic = make_critic(critic_script.model(), instructions="Critique.", render=lambda d: f"Draft: {d.text}")
    reviser = make_reviser(
        reviser_script.model(), Draft, instructions="Revise.",
        render=lambda d, v: f"Draft: {d.text}\nIssues: {v.issues}",
        checks=[lambda d: ["text is too short"] if len(d.text) < 10 else []],
    )
    result = reflect(Draft(text="it does stuff"), critique=critic, revise=reviser)
    assert result.accepted and result.best.text == "REQUIRES_NEW suspends the caller's transaction."
    assert "Issues: ['too vague']" in reviser_script.last_prompt_text()     # critique reached the reviser
    assert "text is too short" in reviser_script.last_prompt_text()          # reviser self-corrected
