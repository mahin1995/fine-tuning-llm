"""Consolidation helpers: turn many similar episodes into a few groups worth a human's look."""
from memory.retrieval import Embedder, cosine


def cluster(texts: list[str], embedder: Embedder, threshold: float) -> list[list[int]]:
    """Greedy single-pass clustering: each text joins the first cluster whose first member is at
    least `threshold` similar, otherwise it starts a new cluster. Returns lists of indices, in
    input order. O(n x clusters x d): meant for consolidation runs over hundreds of episodes."""
    vectors = embedder.embed(texts) if texts else []
    clusters: list[list[int]] = []
    for i, vector in enumerate(vectors):
        for members in clusters:
            if cosine(vectors[members[0]], vector) >= threshold:
                members.append(i)
                break
        else:
            clusters.append([i])
    return clusters
