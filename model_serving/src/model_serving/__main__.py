"""python -m model_serving <command>

    list                          models in models.yaml
    download <model>              fetch a model (registry name or Hugging Face id) into the HF cache
    serve --model <model>         OpenAI-compatible server (default port 8001)
    check --base-url URL --model M   contract check against any server (this one, vLLM, Ollama)
    vllm-command <model>          print the docker command to serve the same model with vLLM
    ollama-help                   steps to serve a model with Ollama
"""
import argparse
import os
import shlex
import sys

from model_serving.config import ConfigError, load_registry, resolve_entry


def cmd_list(args):
    for entry in load_registry().values():
        kwargs = entry.chat_template_kwargs if entry.chat_template_kwargs is not None else "(from run_info.json)"
        print(f"{entry.name:<26} {entry.source:<36} max_tokens_limit={entry.max_tokens_limit:<6} "
              f"chat_template_kwargs={kwargs}")
    return 0


def cmd_download(args):
    from model_serving.download import DownloadError, fetch, size_on_disk

    entry = resolve_entry(args.model)
    try:
        path = fetch(entry)
    except DownloadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{entry.name}: {path} ({size_on_disk(path) / 1e9:.2f} GB)")
    return 0


def build_engine(args, entry):
    if args.engine == "fake":
        from model_serving.engines.fake import FakeEngine

        return FakeEngine(model_name=entry.name, default_chat_template_kwargs=entry.chat_template_kwargs)
    from model_serving.download import fetch
    from model_serving.engines.transformers_engine import TransformersEngine

    return TransformersEngine.load(entry, fetch(entry), device=args.device)


def cmd_serve(args):
    import uvicorn

    from model_serving.api.app import create_app

    entry = resolve_entry(args.model)
    if args.served_model_name:
        entry = entry.model_copy(update={"name": args.served_model_name})
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    print(f"loading {entry.source} as {entry.name!r} ({args.engine}) ...", file=sys.stderr)
    engine = build_engine(args, entry)
    print(f"chat_template_kwargs={engine.default_chat_template_kwargs} max_tokens_limit={entry.max_tokens_limit} "
          f"auth={'on' if api_key else 'off'}", file=sys.stderr)
    uvicorn.run(create_app(engine, max_tokens_limit=entry.max_tokens_limit, api_key=api_key),
                host=args.host, port=args.port)
    return 0


def cmd_check(args):
    from model_serving.check import run_checks

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    kwargs = {"enable_thinking": False} if args.no_thinking else None
    report = run_checks(args.base_url, args.model, api_key=api_key, chat_template_kwargs=kwargs)
    print(report.format())
    return 0 if report.ok else 1


def cmd_vllm(args):
    from model_serving.backends import vllm_command

    print(shlex.join(vllm_command(resolve_entry(args.model), port=args.port)))
    return 0


def cmd_ollama(args):
    from model_serving.backends import OLLAMA_STEPS

    print(OLLAMA_STEPS)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m model_serving", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="models in models.yaml").set_defaults(fn=cmd_list)

    d = sub.add_parser("download", help="fetch a model into the HF cache")
    d.add_argument("model")
    d.set_defaults(fn=cmd_download)

    s = sub.add_parser("serve", help="OpenAI-compatible server")
    s.add_argument("--model", required=True, help="registry name, Hugging Face id or model directory")
    s.add_argument("--served-model-name", default=None, help="name clients use (default: registry name)")
    s.add_argument("--engine", choices=["transformers", "fake"], default="transformers")
    s.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8001)
    s.add_argument("--api-key-env", default="MODEL_SERVING_API_KEY",
                   help="env var holding the API key; requests need 'Authorization: Bearer <key>' if set")
    s.set_defaults(fn=cmd_serve)

    c = sub.add_parser("check", help="contract check against an OpenAI-compatible server")
    c.add_argument("--base-url", default="http://localhost:8001/v1")
    c.add_argument("--model", required=True)
    c.add_argument("--api-key-env", default="MODEL_SERVING_API_KEY")
    c.add_argument("--no-thinking", action="store_true",
                   help="send chat_template_kwargs={enable_thinking: false} (Qwen3 on vLLM)")
    c.set_defaults(fn=cmd_check)

    v = sub.add_parser("vllm-command", help="docker command to serve a model with vLLM")
    v.add_argument("model")
    v.add_argument("--port", type=int, default=8001)
    v.set_defaults(fn=cmd_vllm)

    sub.add_parser("ollama-help", help="how to serve a model with Ollama").set_defaults(fn=cmd_ollama)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
