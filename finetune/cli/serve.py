"""Run the chat API + browser UI.

    PORT=8000 ./run.sh python -m finetune serve --model outputs/qwen3-ft
    # then open http://localhost:8000
"""
import argparse
import sys

from finetune.config import DEFAULT_OUTPUT_DIR
from finetune.serving.app import create_app


def main(argv=None):
    import uvicorn

    p = argparse.ArgumentParser(prog="python -m finetune serve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_OUTPUT_DIR)
    # 0.0.0.0 is needed inside Docker; run.sh only publishes the port on the host's 127.0.0.1.
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    def load():
        from finetune.modeling.chat_model import ChatModel

        return ChatModel.load(args.model)

    uvicorn.run(create_app(load), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
