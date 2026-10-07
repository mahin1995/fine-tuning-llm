"""End-to-end: template consistency, train (full + LoRA), load, generate, evaluate."""
import json

import pytest

from conftest import EVAL_DATA, TRAIN_DATA
from finetune.cli import evaluate as evaluate_cli
from finetune.cli import train as train_cli
from finetune.cli.chat import trim_history
from finetune.profiles import resolve_profile
from finetune.data.io import load_conversations
from finetune.modeling.chat_model import ChatModel
from finetune.modeling.loading import is_adapter_dir
from finetune.modeling.params import GenerationParams
from finetune.training.options import LoraOptions, TrainOptions
from finetune.training.engine import run_training

GREEDY_SHORT = GenerationParams(temperature=0.0, max_new_tokens=8)


def test_inference_prompt_is_prefix_of_training_example(tokenizer, tiny_model_dir):
    """The core train/inference consistency guarantee, in text and token ids."""
    kwargs = resolve_profile(str(tiny_model_dir)).chat_template_kwargs
    assert kwargs == {"enable_thinking": False}  # resolved from config.json model_type=qwen3
    for messages in load_conversations(TRAIN_DATA):
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **kwargs)
        full = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        assert full.startswith(prompt)
        assert prompt.endswith("<think>\n\n</think>\n\n")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        assert full_ids[: len(prompt_ids)] == prompt_ids


def common_args(model_dir, out):
    return ["--model", str(model_dir), "--data", str(TRAIN_DATA), "--output", str(out), "--epochs", "1",
            "--batch-size", "2", "--grad-accum", "1", "--max-length", "256"]


@pytest.fixture(scope="module")
def full_ft_dir(tiny_model_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("full")
    assert train_cli.main(common_args(tiny_model_dir, out) + ["--eval-ratio", "0.2"]) == 0
    return out


@pytest.fixture(scope="module")
def lora_dir(tiny_model_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("lora")
    assert train_cli.main(common_args(tiny_model_dir, out) + ["--lora", "--lora-r", "4", "--merge"]) == 0
    return out


def test_full_finetune_saves_model_and_run_info(full_ft_dir, tiny_model_dir):
    assert (full_ft_dir / "model.safetensors").is_file()
    assert (full_ft_dir / "tokenizer_config.json").is_file()
    info = json.loads((full_ft_dir / "run_info.json").read_text())
    assert info["method"] == "full"
    assert (info["engine"], info["profile"]) == ("trl", "qwen3")
    assert info["chat_template_kwargs"] == {"enable_thinking": False}
    assert info["options"]["lora"] is None
    assert info["base_model"] == str(tiny_model_dir)
    assert "eval_loss" in info["metrics"] and "train_loss" in info["metrics"]


def test_lora_saves_adapter_and_merged_model(lora_dir):
    assert is_adapter_dir(lora_dir)
    assert not (lora_dir / "model.safetensors").exists()
    assert (lora_dir / "merged" / "model.safetensors").is_file()
    assert json.loads((lora_dir / "run_info.json").read_text())["method"] == "lora"


@pytest.mark.parametrize("which", ["full", "lora", "lora_merged"])
def test_trained_models_load_and_generate(which, full_ft_dir, lora_dir):
    path = {"full": full_ft_dir, "lora": lora_dir, "lora_merged": lora_dir / "merged"}[which]
    chat = ChatModel.load(path, device="cpu")
    messages = [{"role": "user", "content": "What is the N+1 problem?"}]
    assert isinstance(chat.generate(messages, GREEDY_SHORT), str)
    assert "".join(chat.stream(messages, GREEDY_SHORT)).strip() == chat.generate(messages, GREEDY_SHORT)
    loss = chat.completion_loss(messages, "A query per row.")
    assert loss > 0


def test_stream_can_be_stopped_early(tiny_model_dir):
    chat = ChatModel.load(tiny_model_dir, device="cpu")
    stream = chat.stream([{"role": "user", "content": "hi"}], GenerationParams(temperature=0.0, max_new_tokens=500))
    next(stream, None)
    stream.close()  # must not hang: the generation thread stops and is joined
    assert chat._lock.acquire(timeout=5)
    chat._lock.release()


def test_training_actually_reduces_answer_loss(tiny_model_dir, tmp_path):
    """Overfit the tiny model: loss on a training answer must go down."""
    conversation = load_conversations(TRAIN_DATA)[0]
    prompt, answer = conversation[:-1], conversation[-1]["content"]
    before = ChatModel.load(tiny_model_dir, device="cpu").completion_loss(prompt, answer)

    out = tmp_path / "overfit"
    args = common_args(tiny_model_dir, out)
    args[args.index("--epochs") + 1] = "5"
    assert train_cli.main(args + ["--lr", "5e-3", "--warmup-ratio", "0"]) == 0
    after = ChatModel.load(out, device="cpu").completion_loss(prompt, answer)
    assert after < before * 0.9


def test_evaluate_writes_report(full_ft_dir, tmp_path):
    report = tmp_path / "report.md"
    assert evaluate_cli.main(["--model", str(full_ft_dir), "--eval-data", str(EVAL_DATA),
                          "--report", str(report), "--max-new-tokens", "4"]) == 0
    text = report.read_text()
    assert "Mean answer loss" in text
    assert text.count("**Fine-tuned**") == len(load_conversations(EVAL_DATA))


def test_merge_requires_lora():
    with pytest.raises(SystemExit):
        train_cli.parse_options(["--merge"])


def test_cli_maps_to_options():
    opts = train_cli.parse_options(["--lora", "--lora-r", "8", "--merge", "--epochs", "2"])
    assert opts.lora == LoraOptions(r=8, merge=True)
    assert opts.epochs == 2 and opts.method == "lora"
    assert train_cli.parse_options([]).lora is None


def test_default_learning_rates():
    assert TrainOptions().lr == 2e-5
    assert TrainOptions(lora=LoraOptions()).lr == 2e-4
    assert TrainOptions(lr=1e-3).lr == 1e-3


@pytest.mark.parametrize("kwargs", [{"warmup_ratio": 1.0}, {"eval_ratio": -0.1}, {"batch_size": 0}])
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainOptions(**kwargs)


def test_run_training_can_be_called_without_cli(tiny_model_dir, tmp_path):
    """Training logic is usable from code (notebooks, other tools) with no argparse involved."""
    result = run_training(TrainOptions(model=str(tiny_model_dir), data=str(TRAIN_DATA), output=str(tmp_path),
                                       epochs=1, batch_size=4, grad_accum=1, max_length=256))
    assert result.output_dir == str(tmp_path)
    assert result.optimizer_steps == 3  # ceil(10 / 4)
    assert "train_loss" in result.metrics
    assert (result.engine, result.profile) == ("trl", "qwen3")


def test_trim_history_keeps_system_and_recent_pairs():
    sys_msg = {"role": "system", "content": "s"}
    pairs = [m for i in range(5) for m in ({"role": "user", "content": f"u{i}"},
                                           {"role": "assistant", "content": f"a{i}"})]
    trimmed = trim_history([sys_msg] + pairs, max_turns=2)
    assert [m["content"] for m in trimmed] == ["s", "u3", "a3", "u4", "a4"]
    assert trim_history(pairs, max_turns=0) == []
