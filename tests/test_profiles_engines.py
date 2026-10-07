"""Model profiles (per-model-family settings) and the training-engine port."""
import json
import sys
import types

import pytest

from finetune.profiles import (
    ProfileError,
    get_profile,
    load_profiles,
    resolve_profile,
)
from finetune.training import engine as engine_mod
from finetune.training.engine import (
    TrainResult,
    engine_names,
    get_engine,
    profile_for,
    run_training,
    write_run_info,
)
from finetune.training.options import LoraOptions, TrainOptions

QWEN3_KWARGS = {"enable_thinking": False}


# ------------------------------------------------------------------ profiles

@pytest.mark.parametrize("model, expected", [
    ("Qwen/Qwen3-0.6B", "qwen3"),
    ("qwen/qwen3-8b", "qwen3"),                      # case-insensitive
    ("Qwen/Qwen2.5-7B-Instruct-AWQ", "qwen2"),
    ("meta-llama/Llama-3.2-1B-Instruct", "llama"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "llama"),
    ("google/gemma-3-1b-it", "default"),
    ("outputs/does-not-exist", "default"),
])
def test_resolve_by_hub_id(model, expected):
    assert resolve_profile(model).name == expected


def test_repo_profiles_are_valid():
    profiles = load_profiles()
    assert profiles["qwen3"].chat_template_kwargs == QWEN3_KWARGS
    assert profiles["default"].chat_template_kwargs == {}
    assert profiles["default"].lora_targets() == "all-linear"
    assert profiles["qwen3"].lora_targets() == ["q_proj", "k_proj", "v_proj", "o_proj",
                                                "gate_proj", "up_proj", "down_proj"]


def write(path, name, data):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(json.dumps(data))
    return path


def test_resolve_local_dir_by_model_type(tmp_path, tiny_model_dir):
    assert resolve_profile(str(tiny_model_dir)).name == "qwen3"  # config.json model_type=qwen3
    assert resolve_profile(str(write(tmp_path / "m", "config.json", {"model_type": "llama"}))).name == "llama"
    assert resolve_profile(str(write(tmp_path / "x", "config.json", {"model_type": "gpt2"}))).name == "default"


def test_run_info_profile_wins_over_config(tmp_path):
    out = write(tmp_path / "out", "config.json", {"model_type": "llama"})
    write(out, "run_info.json", {"profile": "qwen3"})
    assert resolve_profile(str(out)).name == "qwen3"


def test_adapter_resolves_through_its_base_model(tmp_path):
    adapter = write(tmp_path / "adapter", "adapter_config.json", {"base_model_name_or_path": "Qwen/Qwen3-0.6B"})
    assert resolve_profile(str(adapter)).name == "qwen3"


def test_unknown_explicit_profile():
    with pytest.raises(ProfileError, match="unknown model profile 'gpt9'; known: qwen3, qwen2, llama, default"):
        get_profile("gpt9")


@pytest.mark.parametrize("yaml_text, error", [
    ("profiles: {qwen3: {}}", "needs a `profiles` mapping with a 'default' entry"),
    ("profiles: {default: {temperature: 0}}", "unknown key\\(s\\) \\['temperature'\\]"),
    ("profiles: {default: {lora_target_modules: everything}}", "must be a list or 'all-linear'"),
    ("profiles: {default: {lora_target_modules: []}}", "non-empty list"),
    ("profiles: {default: {chat_template_kwargs: [1]}}", "must be a mapping"),
])
def test_invalid_profiles_file(tmp_path, yaml_text, error):
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ProfileError, match=error):
        load_profiles(path)


def test_chat_model_uses_the_resolved_profile(tiny_model_dir):
    pytest.importorskip("torch")
    from finetune.modeling.chat_model import ChatModel

    assert ChatModel.load(tiny_model_dir, device="cpu").chat_template_kwargs == QWEN3_KWARGS
    plain = ChatModel.load(tiny_model_dir, device="cpu", profile=get_profile("default"))
    assert plain.chat_template_kwargs == {}
    prompt = plain.tokenizer.decode(plain.build_inputs([{"role": "user", "content": "hi"}])["input_ids"][0])
    assert "<think>" not in prompt  # without enable_thinking=False the template adds no empty think block


# ------------------------------------------------------------------- engines

def test_engine_registry():
    assert engine_names() == ["trl"]
    assert get_engine("trl").name == "trl"
    with pytest.raises(ValueError, match="unknown training engine 'unsloth'; available: trl"):
        get_engine("unsloth")


@pytest.fixture
def fake_engine(monkeypatch):
    """Register an in-memory engine: proves a new engine needs no change to callers."""
    calls = []

    class FakeEngine:
        name = "fake"

        def train(self, opts, profile):
            calls.append((opts, profile))
            return TrainResult(opts.output, 1, {}, engine=self.name, profile=profile.name)

    module = types.ModuleType("fake_engine_module")
    module.FakeEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "fake_engine_module", module)
    monkeypatch.setitem(engine_mod.ENGINES, "fake", "fake_engine_module:FakeEngine")
    return calls


def test_new_engine_plugs_in_without_caller_changes(fake_engine):
    result = run_training(TrainOptions(model="meta-llama/Llama-3.2-1B", output="o"), engine="fake")
    assert (result.engine, result.profile) == ("fake", "llama")
    result = run_training(TrainOptions(model="meta-llama/Llama-3.2-1B", output="o", profile="qwen3"), engine="fake")
    assert result.profile == "qwen3"  # explicit profile wins over the id
    assert [p.name for _, p in fake_engine] == ["llama", "qwen3"]


def test_profile_for():
    assert profile_for(TrainOptions(model="Qwen/Qwen3-1.7B")).name == "qwen3"
    with pytest.raises(ProfileError):
        profile_for(TrainOptions(profile="nope"))


def test_lora_targets_come_from_the_profile_unless_set():
    pytest.importorskip("trl")
    from finetune.training.trl_engine import lora_targets

    assert lora_targets(LoraOptions(), get_profile("qwen3"))[0] == "q_proj"
    assert lora_targets(LoraOptions(), get_profile("default")) == "all-linear"
    assert lora_targets(LoraOptions(target_modules=("c_attn",)), get_profile("qwen3")) == ["c_attn"]


def test_run_info_contract(tmp_path):
    opts = TrainOptions(model="Qwen/Qwen3-0.6B", output=str(tmp_path))
    path = write_run_info(opts, engine="trl", profile=get_profile("qwen3"), optimizer_steps=7,
                          metrics={"train_loss": 1.0}, packages=("pytest", "not-installed-pkg"))
    info = json.loads(path.read_text())
    assert {k: info[k] for k in ("base_model", "method", "engine", "profile", "chat_template_kwargs",
                                 "optimizer_steps")} == {
        "base_model": "Qwen/Qwen3-0.6B", "method": "full", "engine": "trl", "profile": "qwen3",
        "chat_template_kwargs": QWEN3_KWARGS, "optimizer_steps": 7}
    assert list(info["versions"]) == ["pytest"]  # missing packages are skipped, not a crash
    assert resolve_profile(str(tmp_path)).name == "qwen3"  # the artifact is self-describing


def test_cli_engine_and_profile_flags():
    from finetune.cli.train import parse

    opts, engine = parse([])
    assert (engine, opts.profile) == ("trl", None)
    opts, engine = parse(["--engine", "trl", "--profile", "llama"])
    assert opts.profile == "llama"
    with pytest.raises(SystemExit):
        parse(["--engine", "nope"])
    with pytest.raises(SystemExit):
        parse(["--profile", "nope"])
