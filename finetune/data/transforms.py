"""Turn validated conversations into training examples."""
import random

from finetune.data.schema import DataError


def to_prompt_completion(messages, chat_template_kwargs: dict):
    """Split a conversation into TRL's conversational prompt-completion format.

    Only the final assistant turn becomes the completion, so the loss is computed
    on the answer, not on the question. `chat_template_kwargs` comes from the model's
    profile (finetune.profiles) so training renders prompts exactly like inference.
    """
    return {
        "prompt": messages[:-1],
        "completion": messages[-1:],
        "chat_template_kwargs": dict(chat_template_kwargs),
    }


def train_eval_split(conversations, eval_ratio, seed):
    """Shuffle and split. Returns (train, eval); eval is empty when eval_ratio == 0."""
    if not 0 <= eval_ratio < 1:
        raise ValueError("eval_ratio must be in [0, 1)")
    rows = list(conversations)
    random.Random(seed).shuffle(rows)
    n_eval = round(len(rows) * eval_ratio)
    if eval_ratio > 0 and n_eval == 0:
        n_eval = 1
    if n_eval >= len(rows):
        raise DataError(f"eval split of {n_eval} leaves no training data ({len(rows)} examples)")
    return rows[n_eval:], rows[:n_eval]


def question_of(messages):
    """Normalised last user question, used for duplicate / leakage checks."""
    return messages[-2]["content"].strip().lower()
