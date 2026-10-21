# model_serving

Downloads a model and serves it behind an **OpenAI-compatible API**. This is a self-contained
component, with its own `requirements.txt`, `Dockerfile` and `models.yaml`. It never imports the
training code (`finetune/`) or the apps (`apps/`), and nothing imports it: clients talk to it over
HTTP only. The server can therefore be replaced by **vLLM** or **Ollama** by changing a URL
(ARCHITECTURE.md, section 3.3).

```
model_serving/
├── models.yaml              model registry: source, chat_template_kwargs, max_tokens_limit, dtype
├── requirements.txt         serving dependencies only (torch comes from the base image)
├── Dockerfile, run.sh       own image (`model-serving`), independent of the training image
├── src/model_serving/
│   ├── config.py            registry loading (light: no torch)
│   ├── download.py          Hugging Face Hub / local directory → path
│   ├── engines/             InferenceEngine port: base.py, transformers_engine.py, fake.py, toolcalls.py
│   ├── api/                 FastAPI app + OpenAI request/response schemas
│   ├── check.py             contract check, runs against any OpenAI-compatible server
│   ├── backends.py          vLLM docker command, Ollama steps
│   └── __main__.py          CLI
└── tests/                   unit tests (FakeEngine, no torch needed for most)
```

## Commands

| Docker (`model_serving/run.sh`) | Without Docker (`PYTHONPATH=model_serving/src`) |
|---|---|
| `./run.sh build` | |
| `./run.sh list` | `python -m model_serving list` |
| `./run.sh download qwen3-0.6b` | `python -m model_serving download qwen3-0.6b` |
| `./run.sh serve --model qwen3-0.6b` | `python -m model_serving serve --model qwen3-0.6b` |
| `./run.sh serve --model qwen3-ft` (reads `../outputs/qwen3-ft`) | `python -m model_serving serve --model outputs/qwen3-ft` |
| `./run.sh check --model qwen3-0.6b` | `python -m model_serving check --model qwen3-0.6b` |
| `./run.sh test` | `python -m pytest model_serving/tests` |

- The server listens on `127.0.0.1:8001` (`PORT=9000 ./run.sh serve ...` to change).
- Set `MODEL_SERVING_API_KEY` to require `Authorization: Bearer <key>`. `/health` stays open.
- `--engine fake` serves canned replies without a model, for wiring tests on any machine.

## API

```
GET  /health
GET  /v1/models
POST /v1/chat/completions   model, messages, max_tokens | max_completion_tokens, temperature, top_p,
                            stop, stream, stream_options.include_usage, tools, tool_choice,
                            chat_template_kwargs (extension, same name as vLLM)
```

- **Generation policy belongs to the caller.** `max_tokens`, temperature etc. come with every
  request. The server only caps `max_tokens` at the model's `max_tokens_limit`. It caps the value
  instead of rejecting the request, and reports the cap in the `X-Max-Tokens-Limited: true` header.
- **Chat template variables** (Qwen3 `enable_thinking: false`) are a per-model default: from
  `models.yaml`, or from a fine-tuned output's `run_info.json`. A request can override them.
- **Tool calling**: `tools` are rendered by the model's chat template. Hermes-style
  `<tool_call>{...}</tool_call>` output is returned as OpenAI `tool_calls` with
  `finish_reason: "tool_calls"`. If any block is malformed, the whole text is returned as
  `content`, so nothing is half-parsed.
- **Streaming**: Server-Sent Events. The first chunk carries the role, then content deltas, then a
  final chunk with `finish_reason`, an optional usage chunk, and `data: [DONE]`. Stop strings that
  are split across chunks are held back and never sent. Requests with `tools` are generated in
  full and sent as one chunk.
- Errors use the OpenAI format (`{"error": {"message", "type", "code"}}`). An unknown model returns
  404 `model_not_found`.

Any OpenAI client works:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8001/v1", api_key="unused")
client.chat.completions.create(model="qwen3-0.6b", max_tokens=256,
                               messages=[{"role": "user", "content": "hi"}])
```

## Moving to vLLM or Ollama

1. Start the other runtime under the **same model name**:
   - vLLM: `PYTHONPATH=model_serving/src python -m model_serving vllm-command qwen3-ft` prints the
     `docker run` command. Run it on the host, so the printed paths are host paths.
   - Ollama: `python -m model_serving ollama-help` (needs a GGUF conversion for fine-tuned models).
2. Run the contract check against it:
   `python -m model_serving check --base-url http://localhost:8001/v1 --model qwen3-ft --no-thinking`
   (`--no-thinking` sends `chat_template_kwargs`, which vLLM needs for Qwen3).
3. Change the clients' `*_BASE_URL`. Nothing else changes. vLLM has no per-model default for
   template variables, so clients send `chat_template_kwargs` themselves (OpenAI SDK: `extra_body`).

## Rules (enforced by `tests/test_architecture.py`)

- No imports from `finetune`, `agent`, `ops_crew` or `refiner`, and none of them import `model_serving`.
  Model loading is re-implemented here on purpose (ARCHITECTURE.md, "duplication accepted").
- Light modules (config, download, api, check, backends, engines.base/toolcalls/fake, CLI) don't
  import torch or transformers at module level.
- `models.yaml` `chat_template_kwargs` must match `finetune/model_profiles.yaml` for the same model
  family, so serving renders prompts exactly as training did.
