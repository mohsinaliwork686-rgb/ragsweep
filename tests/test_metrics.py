"""Metrics, checked against values worked out by hand.

If these are wrong, every number ragsweep prints is wrong and nothing downstream
would reveal it. So the expected values here are calculated on paper, not by
running the code and pasting the output.
"""

from __future__ import annotations

import math

import pytest

from ragsweep.metrics import ndcg_at_k, recall_at_k, reciprocal_rank, summarise


class TestRecallAtK:
    def test_single_relevant_found_inside_k(self):
        assert recall_at_k({"c"}, ["a", "b", "c"], 3) == 1.0

    def test_single_relevant_outside_k(self):
        assert recall_at_k({"c"}, ["a", "b", "c"], 2) == 0.0

    def test_two_relevant_one_found_scores_half(self):
        assert recall_at_k({"a", "b"}, ["a", "x", "y"], 3) == 0.5

    def test_two_relevant_both_found(self):
        assert recall_at_k({"a", "b"}, ["a", "x", "b"], 3) == 1.0

    def test_k_larger_than_result_list(self):
        assert recall_at_k({"a"}, ["a"], 10) == 1.0

    def test_nothing_retrieved(self):
        assert recall_at_k({"a"}, [], 5) == 0.0

    def test_repeated_chunks_from_one_document_do_not_inflate(self):
        # three chunks, all from document "a", which is the only relevant one
        assert recall_at_k({"a"}, ["a", "a", "a"], 3) == 1.0

    def test_empty_expected_is_a_label_error(self):
        with pytest.raises(ValueError, match="at least one relevant"):
            recall_at_k(set(), ["a"], 3)

    @pytest.mark.parametrize("k", [0, -1])
    def test_k_must_be_positive(self, k):
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k({"a"}, ["a"], k)


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("retrieved", "expected_score"),
        [
            (["a", "b", "c"], 1.0),        # first place
            (["b", "a", "c"], 0.5),        # second place
            (["b", "c", "a"], 1 / 3),      # third place
            (["b", "c", "d"], 0.0),        # never found
            ([], 0.0),                     # nothing retrieved
        ],
    )
    def test_position_decides_the_score(self, retrieved, expected_score):
        assert reciprocal_rank({"a"}, retrieved) == pytest.approx(expected_score)

    def test_only_the_first_relevant_hit_counts(self):
        # "a" at position 2 and "b" at position 3; the score comes from position 2 only
        assert reciprocal_rank({"a", "b"}, ["x", "a", "b"]) == 0.5


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k({"a"}, ["a", "b", "c"], 3) == 1.0

    def test_relevant_document_in_third_place(self):
        # gains [0, 0, 1] -> DCG = 1 / log2(4) = 0.5;  IDCG = 1 / log2(2) = 1.0
        assert ndcg_at_k({"c"}, ["a", "b", "c"], 3) == pytest.approx(0.5)

    def test_two_relevant_split_apart(self):
        # gains [1, 0, 1] -> DCG = 1/log2(2) + 1/log2(4) = 1.5
        # ideal [1, 1]    -> IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
        expected = 1.5 / (1 + 1 / math.log2(3))
        assert ndcg_at_k({"a", "b"}, ["a", "x", "b"], 3) == pytest.approx(expected)
        assert expected == pytest.approx(0.91973, abs=1e-5)

    def test_a_repeated_document_scores_nothing_the_second_time(self):
        # ["a", "a", "b"] must not beat ["a", "b"] by counting "a" twice
        assert ndcg_at_k({"a", "b"}, ["a", "a", "b"], 3) == pytest.approx(
            1.5 / (1 + 1 / math.log2(3))
        )

    def test_nothing_relevant_retrieved(self):
        assert ndcg_at_k({"a"}, ["x", "y"], 2) == 0.0

    def test_ideal_is_capped_at_k(self):
        # three relevant documents but only room for one: finding one is a perfect score
        assert ndcg_at_k({"a", "b", "c"}, ["a"], 1) == 1.0


class TestSummarise:
    def test_averages_across_cases(self):
        cases = [
            ({"a"}, ["a", "x"]),   # recall@1 = 1.0, rr = 1.0
            ({"b"}, ["x", "b"]),   # recall@1 = 0.0, rr = 0.5
        ]
        scores = summarise(cases, ks=(1, 2))
        assert scores["recall@1"] == pytest.approx(0.5)
        assert scores["recall@2"] == pytest.approx(1.0)
        assert scores["mrr"] == pytest.approx(0.75)

    def test_no_cases_returns_nothing_rather_than_raising(self):
        assert summarise([]) == {}

    def test_reports_every_requested_k(self):
        scores = summarise([({"a"}, ["a"])], ks=(1, 5))
        assert set(scores) == {"recall@1", "recall@5", "ndcg@1", "ndcg@5", "mrr"}
