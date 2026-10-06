"""Run the tool-calling agent with a Qwen model from this repo.

    python -m agent --model outputs/qwen3-ft "What is 17 * 23?"
    python -m agent --model Qwen/Qwen3-0.6B                       # interactive
    python -m agent --kb data/train.jsonl --max-steps 5 --quiet "Explain REQUIRES_NEW"

This module is the agent's composition root: it is the only place that picks the
backend (Qwen), the tools and the knowledge base and wires them into Agent.
"""
import argparse
import json
import sys

from agent.builtin_tools import KnowledgeBase, default_registry
from agent.loop import Agent, Step

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for Java and Spring Boot questions. "
    "Use a tool only when it helps: search_knowledge_base for questions the project's Q&A may cover, "
    "calculator for arithmetic, current_time for today's date or time. "
    "When you have enough information, answer directly without calling a tool."
)


def print_step(step: Step):
    if step.call is None:
        print(f"  ! {step.output}", file=sys.stderr)
        return
    args = json.dumps(step.call.arguments, ensure_ascii=False)
    output = step.output if len(step.output) <= 300 else step.output[:300] + " ..."
    print(f"  -> {step.call.name}({args})\n     {output}", file=sys.stderr)


def build_agent(args):
    from agent.backends.qwen import AGENT_PARAMS, QwenBackend
    from qwen_ft.data.io import load_conversations
    from qwen_ft.modeling.params import GenerationParams

    kb = KnowledgeBase.from_conversations(load_conversations(args.kb)) if args.kb else None
    params = GenerationParams(temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                              top_p=AGENT_PARAMS.top_p, top_k=AGENT_PARAMS.top_k)
    print(f"loading {args.model} ...", file=sys.stderr)
    backend = QwenBackend.from_path(args.model, params)
    return Agent(
        backend,
        default_registry(kb),
        system_prompt=args.system,
        max_steps=args.max_steps,
        on_step=None if args.quiet else print_step,
    )


def answer(agent, question, history=None):
    result = agent.run(question, history)
    if result.answer is None:
        print(f"(stopped after {agent.max_steps} steps without a final answer)")
    else:
        print(result.answer)
    return result.messages


def main(argv=None):
    from qwen_ft.config import DEFAULT_OUTPUT_DIR, DEFAULT_TRAIN_DATA

    p = argparse.ArgumentParser(prog="python -m agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?", help="ask one question and exit (omit for interactive mode)")
    p.add_argument("--model", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--kb", default=DEFAULT_TRAIN_DATA, help="JSONL Q&A file for search_knowledge_base ('' = off)")
    p.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (most reliable tool calls)")
    p.add_argument("--quiet", action="store_true", help="don't print tool calls")
    args = p.parse_args(argv)

    agent = build_agent(args)
    if args.question:
        answer(agent, args.question)
        return 0

    print("agent ready. /reset clears history, /exit quits.\n", file=sys.stderr)
    history = None
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question == "/exit":
            return 0
        if question == "/reset":
            history = None
            print("(history cleared)\n")
            continue
        history = answer(agent, question, history)
        print()


if __name__ == "__main__":
    sys.exit(main())
