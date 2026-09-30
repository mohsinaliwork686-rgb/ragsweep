"""Vector similarity over embeddings.

Brute force: the chunk matrix is already L2-normalised, so one matrix-vector product
gives every cosine similarity at once. At a hundred thousand chunks that is a few
milliseconds, which is well past the size this tool is meant for, so an approximate
index would add a dependency and a tuning knob for no gain.
"""

from __future__ import annotations

import numpy as np

from ragsweep.embedding import Embedder, EmbeddingCache, embed_texts
from ragsweep.retrievers.base import Hit, top_k


class DenseRetriever:
    def __init__(
        self,
        vectors: np.ndarray,
        embedder: Embedder,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.vectors = vectors
        self.embedder = embedder
        # Questions repeat for every configuration in a sweep, so the same cache that
        # holds the chunk vectors makes re-encoding them free.
        self.cache = cache

    @property
    def name(self) -> str:
        return "dense"

    def search(self, query: str, k: int) -> list[Hit]:
        if self.vectors.size == 0:
            return []
        query_vector = embed_texts([query], self.embedder, self.cache)[0]
        return top_k(self.vectors @ query_vector, k)
