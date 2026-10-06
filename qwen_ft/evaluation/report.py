"""Markdown report for a base vs fine-tuned comparison."""


def render_report(conversations, base, tuned):
    lines = [
        "# Evaluation report",
        "",
        "| Model | Mean answer loss (lower is better) |",
        "|---|---|",
        f"| base: `{base.model}` | {base.mean_loss:.4f} |",
        f"| fine-tuned: `{tuned.model}` | {tuned.mean_loss:.4f} |",
        "",
    ]
    for i, (messages, b, t) in enumerate(zip(conversations, base.results, tuned.results), start=1):
        lines += [
            f"## {i}. {messages[-2]['content']}",
            "",
            f"**Reference:** {messages[-1]['content']}",
            "",
            f"**Base** (loss {b.loss:.3f}):\n\n{b.answer}",
            "",
            f"**Fine-tuned** (loss {t.loss:.3f}):\n\n{t.answer}",
            "",
        ]
    return "\n".join(lines)
