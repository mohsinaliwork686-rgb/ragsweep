"""Okapi BM25, written out rather than pulled in as a dependency.

It is about sixty lines, and keeping it here means the base install is numpy and rich,
so `ragsweep` does something useful the moment it is installed. It is also the baseline
the dense retrievers have to beat, so it belongs in the repo rather than in a black box.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ragsweep.embedding import tokenise
from ragsweep.retrievers.base import Hit, top_k

K1 = 1.5
B = 0.75


class BM25Retriever:
    """Classic keyword ranking: term frequency, damped, weighted by rarity.

    Repeated terms in a query are counted once. For the short natural-language
    questions this tool is built around, letting one repeated word dominate the score
    does more harm than good.
    """

    def __init__(self, texts: list[str], *, k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.count = len(texts)

        tokenised = [tokenise(text) for text in texts]
        self.lengths = np.array([len(tokens) for tokens in tokenised], dtype=np.float32)
        self.average_length = float(self.lengths.mean()) if self.count else 0.0

        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, tokens in enumerate(tokenised):
            for term, frequency in Counter(tokens).items():
                self.postings.setdefault(term, []).append((index, frequency))

        self.idf = {
            term: math.log(1 + (self.count - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    @property
    def name(self) -> str:
        return "bm25"

    def search(self, query: str, k: int) -> list[Hit]:
        scores = np.zeros(self.count, dtype=np.float32)
        if not self.average_length:
            return []

        for term in set(tokenise(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for index, frequency in posting:
                length_penalty = 1 - self.b + self.b * self.lengths[index] / self.average_length
                scores[index] += idf * frequency * (self.k1 + 1) / (
                    frequency + self.k1 * length_penalty
                )

        return top_k(scores, k, drop_zero=True)
