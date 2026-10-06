import json

import pytest

from conftest import EVAL_DATA, TRAIN_DATA
from qwen_ft.config import CHAT_TEMPLATE_KWARGS
from qwen_ft.data.io import load_conversations
from qwen_ft.data.schema import DataError, validate_messages, validate_prompt
from qwen_ft.data.transforms import to_prompt_completion, train_eval_split

U = {"role": "user", "content": "q"}
A = {"role": "assistant", "content": "a"}
S = {"role": "system", "content": "s"}


def write_jsonl(path, rows):
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("messages", [[U, A], [S, U, A], [U, A, U, A], [S, U, A, U, A]])
def test_valid_conversations(messages):
    validate_messages(messages)


@pytest.mark.parametrize(
    "messages, error",
    [
        ([U], "at least"),
        ("not a list", "at least"),
        ([U, U], "alternate"),
        ([A, U], "alternate"),
        ([U, A, U], "last message"),
        ([U, S, A], "system message"),
        ([U, {"role": "assistant", "content": "  "}], "empty"),
        ([U, {"role": "bot", "content": "a"}], "invalid role"),
    ],
)
def test_invalid_conversations(messages, error):
    with pytest.raises(DataError, match=error):
        validate_messages(messages)


def test_load_reports_line_number(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [{"messages": [U, A]}, "{broken"])
    with pytest.raises(DataError, match=r"d.jsonl:2: invalid JSON"):
        load_conversations(path)


def test_load_skips_blank_lines(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [{"messages": [U, A]}, "", {"messages": [U, A]}])
    assert len(load_conversations(path)) == 2


def test_load_rejects_empty_file(tmp_path):
    with pytest.raises(DataError, match="no examples"):
        load_conversations(write_jsonl(tmp_path / "d.jsonl", []))


def test_repo_datasets_are_valid():
    assert len(load_conversations(TRAIN_DATA)) >= 1
    assert len(load_conversations(EVAL_DATA)) >= 1


def test_validate_prompt_requires_user_last():
    validate_prompt([U])
    validate_prompt([S, U, A, U])
    with pytest.raises(DataError):
        validate_prompt([U, A])


def test_to_prompt_completion_splits_last_turn():
    row = to_prompt_completion([S, U, A, U, A])
    assert row["prompt"] == [S, U, A, U]
    assert row["completion"] == [A]
    assert row["chat_template_kwargs"] == CHAT_TEMPLATE_KWARGS


def test_split_is_deterministic_and_disjoint():
    rows = [[{"role": "user", "content": str(i)}, A] for i in range(10)]
    train1, eval1 = train_eval_split(rows, 0.2, seed=1)
    train2, eval2 = train_eval_split(rows, 0.2, seed=1)
    assert (train1, eval1) == (train2, eval2)
    assert len(eval1) == 2 and len(train1) == 8
    assert not {r[0]["content"] for r in train1} & {r[0]["content"] for r in eval1}


def test_split_small_ratio_keeps_one_eval_example():
    rows = [[U, A]] * 5
    assert len(train_eval_split(rows, 0.01, seed=0)[1]) == 1
    assert train_eval_split(rows, 0.0, seed=0)[1] == []


def test_split_rejects_eval_taking_everything():
    with pytest.raises(DataError):
        train_eval_split([[U, A]], 0.5, seed=0)
