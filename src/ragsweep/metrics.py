"""Retrieval metrics.

Every number this tool prints comes from here, so these functions are deliberately
small, pure and tested against values worked out by hand.

Relevance is recorded per *document*, not per chunk, because chunk ids change every
time the chunk size changes. ``retrieved`` is therefore a ranked list of document ids,
one entry per retrieved chunk, and the same document may appear more than once.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence

__all__ = [
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "summarise",
]


def _check(expected: Collection[str], k: int | None = None) -> set[str]:
    if not expected:
        raise ValueError("expected must list at least one relevant document")
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    return set(expected)


def recall_at_k(expected: Collection[str], retrieved: Sequence[str], k: int) -> float:
    """Share of the relevant documents that appear in the top ``k`` results.

    With one relevant document this is simply "did we find it": 1.0 or 0.0. With two,
    finding one of them scores 0.5, which is what makes multi-hop questions readable.
    """
    wanted = _check(expected, k)
    found = wanted & set(retrieved[:k])
    return len(found) / len(wanted)


def reciprocal_rank(expected: Collection[str], retrieved: Sequence[str]) -> float:
    """1 / position of the first relevant result. 0.0 if none was retrieved.

    First place scores 1.0, second 0.5, third 0.33. Averaged over every question this
    is MRR, and unlike recall it cares *where* the right answer landed.
    """
    wanted = _check(expected)
    for position, document_id in enumerate(retrieved, start=1):
        if document_id in wanted:
            return 1.0 / position
    return 0.0


def _dcg(gains: Iterable[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(expected: Collection[str], retrieved: Sequence[str], k: int) -> float:
    """Normalised discounted cumulative gain over binary relevance.

    Like MRR, but it accounts for every relevant document rather than only the first,
    and it rewards putting them higher. A document already counted scores nothing the
    second time, so repeated chunks from one document cannot inflate the score.
    """
    wanted = _check(expected, k)
    seen: set[str] = set()
    gains: list[float] = []
    for document_id in retrieved[:k]:
        if document_id in wanted and document_id not in seen:
            seen.add(document_id)
            gains.append(1.0)
        else:
            gains.append(0.0)
    ideal = _dcg([1.0] * min(len(wanted), k))
    return _dcg(gains) / ideal if ideal else 0.0


def summarise(
    cases: Iterable[tuple[Collection[str], Sequence[str]]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Average every metric across a set of (expected, retrieved) pairs.

    Returns an empty dict for no cases rather than raising, because a sweep over an
    empty label file should report nothing rather than blow up mid-run.
    """
    cases = list(cases)
    if not cases:
        return {}

    scores: dict[str, float] = {}
    for k in ks:
        scores[f"recall@{k}"] = sum(recall_at_k(e, r, k) for e, r in cases) / len(cases)
        scores[f"ndcg@{k}"] = sum(ndcg_at_k(e, r, k) for e, r in cases) / len(cases)
    scores["mrr"] = sum(reciprocal_rank(e, r) for e, r in cases) / len(cases)
    return scores
