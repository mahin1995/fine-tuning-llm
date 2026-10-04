"""Interactive terminal chat with a base or fine-tuned model.

    python chat.py                         # outputs/qwen3-ft
    python chat.py --model Qwen/Qwen3-0.6B # compare with the base model
    python chat.py --system "You are a senior Java interviewer."

Commands: /reset clears history, /exit (or Ctrl-D) quits.
"""
import argparse
import sys

from inference import ChatModel, GenerationParams


def trim_history(history, max_turns):
    """Keep the system message plus the last `max_turns` complete user/assistant pairs."""
    system = [m for m in history[:1] if m["role"] == "system"]
    turns = history[len(system):]
    return system + (turns[-max_turns * 2:] if max_turns > 0 else [])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="outputs/qwen3-ft")
    p.add_argument("--system", default=None, help="optional system prompt")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7, help="0 = greedy")
    p.add_argument("--max-turns", type=int, default=8, help="history turns kept in the prompt")
    args = p.parse_args(argv)

    print(f"loading {args.model} ...")
    chat = ChatModel.load(args.model)
    params = GenerationParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    initial = [{"role": "system", "content": args.system}] if args.system else []
    history = list(initial)
    print("ready. /reset clears history, /exit quits.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user == "/exit":
            return 0
        if user == "/reset":
            history = list(initial)
            print("(history cleared)\n")
            continue

        # Trim before appending so the kept turns always start with a user message.
        history = trim_history(history, args.max_turns) + [{"role": "user", "content": user}]
        print("bot> ", end="", flush=True)
        reply = []
        try:
            for chunk in chat.stream(history, params):
                print(chunk, end="", flush=True)
                reply.append(chunk)
        except KeyboardInterrupt:
            print(" [interrupted]", end="")
        print("\n")
        answer = "".join(reply).strip()
        if answer:
            history.append({"role": "assistant", "content": answer})
        else:
            history.pop()  # nothing generated: drop the unanswered question


if __name__ == "__main__":
    sys.exit(main())
