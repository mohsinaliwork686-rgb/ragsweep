"""What every retriever has to look like."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

#: (chunk index, score), best first.
Hit = tuple[int, float]


@runtime_checkable
class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def search(self, query: str, k: int) -> list[Hit]:
        """Return at most ``k`` hits, ranked best first."""


def top_k(scores: np.ndarray, k: int, *, drop_zero: bool = False) -> list[Hit]:
    """Rank by score, descending, breaking ties by index so runs are reproducible.

    A full stable sort rather than argpartition: it costs a few milliseconds even at a
    hundred thousand chunks, and deterministic output is worth more here than the
    milliseconds, because the whole tool exists to compare runs against each other.
    """
    if k <= 0 or scores.size == 0:
        return []
    order = np.lexsort((np.arange(scores.size), -scores))[:k]
    hits = [(int(index), float(scores[index])) for index in order]
    return [hit for hit in hits if hit[1] > 0] if drop_zero else hits
