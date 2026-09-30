"""The three retrievers.

The shared contract is tested against all three at once. A retriever that quietly
returned unranked results, or more hits than asked for, would still pass a spot check
on one implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragsweep.embedding import HashingEmbedder, embed_texts
from ragsweep.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, top_k

TEXTS = [
    "Refunds are processed within five business days of approval.",
    "Approval is required for refunds above five hundred dollars.",
    "The office coffee machine is serviced every quarter.",
    "Shipping is free on orders over fifty dollars.",
    "Annual leave must be requested two weeks in advance.",
]


def build(kind: str, texts: list[str] = TEXTS):
    embedder = HashingEmbedder(dimensions=512)
    if kind == "bm25":
        return BM25Retriever(texts)
    dense = DenseRetriever(embed_texts(texts, embedder), embedder)
    if kind == "dense":
        return dense
    return HybridRetriever(dense, BM25Retriever(texts))


@pytest.mark.parametrize("kind", ["bm25", "dense", "hybrid"])
class TestEveryRetrieverHonoursTheContract:
    def test_has_a_name(self, kind):
        assert build(kind).name == kind

    def test_never_returns_more_than_k(self, kind):
        assert len(build(kind).search("refund approval dollars", 2)) <= 2

    def test_results_are_ranked_best_first(self, kind):
        hits = build(kind).search("refund approval", 5)
        scores = [score for _, score in hits]
        assert scores == sorted(scores, reverse=True)

    def test_indexes_are_valid_and_unique(self, kind):
        hits = build(kind).search("refund approval dollars", 5)
        indexes = [index for index, _ in hits]
        assert len(set(indexes)) == len(indexes)
        assert all(0 <= index < len(TEXTS) for index in indexes)

    def test_finds_the_obviously_relevant_chunk(self, kind):
        top = build(kind).search("how long do refunds take", 1)
        assert top and top[0][0] in {0, 1}

    def test_k_of_zero_returns_nothing(self, kind):
        assert build(kind).search("refund", 0) == []

    def test_k_larger_than_the_corpus(self, kind):
        assert len(build(kind).search("refund approval dollars leave coffee", 99)) <= len(TEXTS)

    def test_repeated_searches_agree(self, kind):
        retriever = build(kind)
        assert retriever.search("refund", 3) == retriever.search("refund", 3)

    def test_empty_corpus_does_not_crash(self, kind):
        assert build(kind, []).search("refund", 3) == []


class TestBM25:
    def test_no_shared_words_means_no_results(self):
        assert BM25Retriever(TEXTS).search("zebra unicorn", 3) == []

    def test_rare_words_outrank_common_ones(self):
        texts = ["the cat sat", "the dog sat", "the bird sat", "a zebra appeared"]
        hits = BM25Retriever(texts).search("zebra the", 4)
        assert hits[0][0] == 3

    def test_shorter_documents_win_on_equal_term_frequency(self):
        texts = ["refund", "refund " + "padding " * 40]
        assert BM25Retriever(texts).search("refund", 2)[0][0] == 0

    def test_a_repeated_query_word_is_counted_once(self):
        retriever = BM25Retriever(TEXTS)
        assert retriever.search("refund refund refund", 3) == retriever.search("refund", 3)

    def test_case_is_ignored(self):
        retriever = BM25Retriever(TEXTS)
        assert retriever.search("REFUND", 3) == retriever.search("refund", 3)


class TestDense:
    def test_scores_are_cosine_similarities(self):
        embedder = HashingEmbedder(dimensions=256)
        vectors = embed_texts(TEXTS, embedder)
        retriever = DenseRetriever(vectors, embedder)
        index, score = retriever.search(TEXTS[0], 1)[0]
        assert index == 0
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_uses_the_cache_for_query_vectors(self, tmp_path):
        from ragsweep.embedding import EmbeddingCache

        cache = EmbeddingCache(tmp_path)
        embedder = HashingEmbedder(dimensions=128)
        retriever = DenseRetriever(embed_texts(TEXTS, embedder, cache), embedder, cache)
        retriever.search("refund", 3)
        before = cache.hits
        retriever.search("refund", 3)
        assert cache.hits > before


class TestHybrid:
    def test_fuses_by_rank_not_by_score(self):
        """BM25 scores are unbounded and cosine sits in [-1, 1]; only ranks are comparable."""
        retriever = build("hybrid")
        hits = retriever.search("refund approval", 3)
        assert all(0 < score < 1 for _, score in hits)

    def test_a_chunk_both_retrievers_like_beats_one_only_a_single_one_likes(self):
        retriever = build("hybrid")
        hits = dict(retriever.search("refunds above five hundred dollars approval", 5))
        assert hits[1] > hits[2]

    def test_ties_break_on_index_so_runs_are_reproducible(self):
        retriever = build("hybrid")
        assert retriever.search("dollars", 5) == retriever.search("dollars", 5)


class TestTopK:
    def test_ranks_descending(self):
        assert top_k(np.array([0.1, 0.9, 0.5]), 3) == [(1, 0.9), (2, 0.5), (0, 0.1)]

    def test_ties_break_on_index(self):
        assert top_k(np.array([0.5, 0.5, 0.5]), 2) == [(0, 0.5), (1, 0.5)]

    def test_drop_zero_removes_non_matches(self):
        assert top_k(np.array([0.0, 0.7, 0.0]), 3, drop_zero=True) == [(1, pytest.approx(0.7))]

    def test_keeps_zeros_by_default(self):
        assert len(top_k(np.array([0.0, 0.7, 0.0]), 3)) == 3

    def test_empty_scores(self):
        assert top_k(np.array([]), 5) == []
