"""model_serving: download models and serve them behind an OpenAI-compatible API.

Self-contained component (see ARCHITECTURE.md): its own requirements.txt, Dockerfile and
models.yaml, and it imports nothing from the rest of the repository. Clients only know the
HTTP contract, so this server can be replaced by vLLM or Ollama by changing a URL.

    config     models.yaml -> ModelEntry
    download   fetch a model from the Hugging Face Hub (no torch needed)
    engines/   InferenceEngine port + transformers / fake implementations, tool-call parsing
    api/       OpenAI request/response schemas and the FastAPI app
    check      contract check against any OpenAI-compatible server (this one, vLLM, Ollama)
    backends   helpers for moving to vLLM / Ollama
"""
