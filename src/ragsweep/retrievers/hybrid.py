"""Reciprocal rank fusion of a dense and a lexical retriever.

Blending raw scores does not work: BM25 scores are unbounded and cosine similarities
sit in [-1, 1], so whichever has the larger numbers wins regardless of quality. RRF
throws the scores away and combines *positions* instead, which needs no normalisation
and no per-corpus tuning.

    score(chunk) = sum over retrievers of  1 / (rrf_k + rank in that retriever)

``rrf_k`` of 60 is the value from the original paper and is not worth tuning here.
"""

from __future__ import annotations

from ragsweep.retrievers.base import Hit, Retriever

RRF_K = 60


class HybridRetriever:
    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        *,
        rrf_k: int = RRF_K,
        depth: int | None = None,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k
        # How deep to look in each retriever before fusing. Too shallow and a chunk
        # ranked 12th by both never gets the chance to win on combined evidence.
        self.depth = depth

    @property
    def name(self) -> str:
        return "hybrid"

    def search(self, query: str, k: int) -> list[Hit]:
        depth = self.depth or max(k * 5, 50)
        fused: dict[int, float] = {}
        for retriever in (self.dense, self.lexical):
            for rank, (index, _) in enumerate(retriever.search(query, depth), start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]
