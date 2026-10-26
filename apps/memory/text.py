"""Text normalisation shared by every index: word terms for BM25, character n-grams for hashing."""
import re

# Keeps Java / Spring tokens whole: @Transactional, N+1, REQUIRES_NEW, spring.jpa.open-in-view, C#.
_WORD = re.compile(r"[a-z0-9@_+#.\-]+")
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "difference", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "that", "the", "this", "to",
    "what", "when", "which", "why", "with", "you", "your",
})


def normalize(text: str) -> str:
    """Whitespace- and case-insensitive form, used for every "copied verbatim" comparison."""
    return " ".join(text.split()).lower()


_PARTS = re.compile(r"[.\-_@]+")


def terms(text: str, expand: bool = True) -> list[str]:
    """Lower-cased word tokens without stopwords (repeats kept: BM25 counts them).

    With `expand`, compound tokens also yield their parts, so "propagation" finds
    Propagation.REQUIRED and "transactional" finds @Transactional."""
    out = []
    for word in (w.strip(".-") for w in _WORD.findall(text.lower())):
        if len(word) < 2 or word in STOPWORDS:
            continue
        out.append(word)
        if expand:
            parts = [p for p in _PARTS.split(word) if len(p) > 1 and p not in STOPWORDS]
            if parts != [word]:
                out.extend(parts)
    return out


def char_ngrams(text: str, sizes: tuple[int, ...] = (3, 4, 5)) -> list[str]:
    padded = f" {normalize(text)} "
    return [padded[i:i + n] for n in sizes for i in range(len(padded) - n + 1)]
