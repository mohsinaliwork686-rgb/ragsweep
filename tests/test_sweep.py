"""The sweep runner.

The point of these tests is the *sharing*. A sweep that produced correct numbers while
re-chunking and re-embedding for every combination would pass a results check and still
be unusable, because nobody re-runs a tool that takes an hour.
"""

from __future__ import annotations

import pytest

from conftest import DOCUMENTS, LABELS, SMALL
from ragsweep.embedding import EmbeddingCache
from ragsweep.labels import Label
from ragsweep.sweep import RunSpec, SweepConfig, plan_runs, run_sweep


class TestPlanRuns:
    def test_one_run_per_combination(self):
        config = SweepConfig(
            retrievers=("dense",), strategies=("fixed", "sentence"),
            chunk_sizes=(256,), overlaps=(0,), models=("hashing-64",),
        )
        assert len(plan_runs(config)) == 2

    def test_overlap_at_or_above_size_is_dropped(self):
        config = SweepConfig(
            retrievers=("bm25",), strategies=("fixed",),
            chunk_sizes=(128,), overlaps=(0, 64, 128, 256),
        )
        assert [spec.overlap for spec in plan_runs(config)] == [0, 64]

    def test_bm25_runs_once_per_chunking_not_once_per_model(self):
        config = SweepConfig(
            retrievers=("bm25", "dense"), strategies=("fixed",),
            chunk_sizes=(256,), overlaps=(0,), models=("hashing-64", "hashing-128"),
        )
        specs = plan_runs(config)
        assert sum(1 for spec in specs if spec.retriever == "bm25") == 1
        assert sum(1 for spec in specs if spec.retriever == "dense") == 2

    def test_lexical_runs_carry_no_model(self):
        config = SweepConfig(retrievers=("bm25",), strategies=("fixed",), chunk_sizes=(256,))
        assert all(spec.model is None for spec in plan_runs(config))

    def test_run_ids_are_unique_and_descriptive(self):
        specs = plan_runs(SMALL)
        assert len({spec.id for spec in specs}) == len(specs)
        assert RunSpec("dense", "fixed", 512, 64, "m").id == "dense-fixed-512-64-m"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"retrievers": ("magic",)}, "unknown retrievers"),
            ({"strategies": ("magic",)}, "unknown strategies"),
            ({"ks": (0,)}, "ks must all be positive"),
            ({"chunk_sizes": (-1,)}, "chunk_sizes must all be positive"),
            ({"models": ()}, "at least one model"),
        ],
    )
    def test_bad_config_is_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            plan_runs(SweepConfig(**kwargs))


class TestRunSweep:
    def test_one_result_per_planned_run(self):
        results = run_sweep(DOCUMENTS, LABELS, SMALL)
        assert [r.id for r in results] == [spec.id for spec in plan_runs(SMALL)]

    def test_every_requested_k_is_reported(self):
        (result, *_) = run_sweep(DOCUMENTS, LABELS, SMALL)
        assert set(result.metrics) == {"recall@1", "recall@3", "ndcg@1", "ndcg@3", "mrr"}

    def test_one_case_per_label(self):
        (result, *_) = run_sweep(DOCUMENTS, LABELS, SMALL)
        assert [case.id for case in result.cases] == ["q1", "q2", "q3"]

    def test_cases_carry_tags_and_expectations(self):
        (result, *_) = run_sweep(DOCUMENTS, LABELS, SMALL)
        case = result.cases[0]
        assert case.tags == ("policy",)
        assert case.expected == ("refunds.md",)

    def test_score_is_the_reciprocal_of_the_first_correct_rank(self):
        for result in run_sweep(DOCUMENTS, LABELS, SMALL):
            for case in result.cases:
                if case.first_correct_rank is None:
                    assert case.score == 0.0
                else:
                    assert case.score == pytest.approx(1.0 / case.first_correct_rank)

    def test_a_missed_question_scores_zero(self):
        labels = [Label("q9", "quantum zebra unicorn", frozenset({"refunds.md"}))]
        results = run_sweep(DOCUMENTS, labels, SweepConfig(
            retrievers=("bm25",), strategies=("fixed",), chunk_sizes=(120,),
            overlaps=(0,), ks=(1, 3),
        ))
        case = results[0].cases[0]
        assert case.first_correct_rank is None
        assert case.score == 0.0
        assert results[0].metrics["recall@3"] == 0.0

    def test_retrieval_actually_finds_the_right_document(self):
        results = run_sweep(DOCUMENTS, LABELS, SMALL)
        best = max(results, key=lambda r: r.metrics["recall@3"])
        assert best.metrics["recall@3"] == 1.0

    def test_config_is_recorded_on_every_run(self):
        (result, *_) = run_sweep(DOCUMENTS, LABELS, SMALL)
        assert result.config["strategy"] == "sentence"
        assert result.config["chunk_size"] == 120

    def test_timing_is_recorded(self):
        (result, *_) = run_sweep(DOCUMENTS, LABELS, SMALL)
        assert result.timing["chunks"] > 0
        assert set(result.timing) >= {"chunk_s", "embed_s", "index_s", "search_s"}

    def test_progress_callback_fires_once_per_run(self):
        seen = []
        results = run_sweep(DOCUMENTS, LABELS, SMALL, on_run=seen.append)
        assert [r.id for r in seen] == [r.id for r in results]

    def test_results_are_reproducible(self):
        first = run_sweep(DOCUMENTS, LABELS, SMALL)
        second = run_sweep(DOCUMENTS, LABELS, SMALL)
        assert [r.metrics for r in first] == [r.metrics for r in second]
        assert [r.cases for r in first] == [r.cases for r in second]


class TestWorkIsShared:
    def test_chunking_happens_once_per_chunking_not_once_per_run(self, monkeypatch):
        import ragsweep.sweep as sweep_module

        calls = []
        original = sweep_module.chunk_corpus

        def counting(documents, **kwargs):
            calls.append(kwargs)
            return original(documents, **kwargs)

        monkeypatch.setattr(sweep_module, "chunk_corpus", counting)

        config = SweepConfig(
            retrievers=("bm25", "dense", "hybrid"), strategies=("fixed",),
            chunk_sizes=(120, 240), overlaps=(0,),
            models=("hashing-64", "hashing-128"), ks=(3,),
        )
        results = run_sweep(DOCUMENTS, LABELS, config)

        assert len(results) == 10   # 2 chunkings x (1 bm25 + 2 models x 2 dense/hybrid)
        assert len(calls) == 2      # but only two chunkings

    def test_embedding_happens_once_per_chunking_and_model(self, tmp_path):
        cache = EmbeddingCache(tmp_path)
        config = SweepConfig(
            retrievers=("dense", "hybrid"), strategies=("fixed",),
            chunk_sizes=(120,), overlaps=(0,), models=("hashing-64",), ks=(3,),
        )
        run_sweep(DOCUMENTS, LABELS, config, cache=cache)
        # dense and hybrid share one embedding pass, so the second run of each is a hit
        assert cache.misses > 0
        before = cache.misses
        run_sweep(DOCUMENTS, LABELS, config, cache=cache)
        assert cache.misses == before

    def test_the_cache_makes_a_repeat_sweep_free(self, tmp_path):
        cache = EmbeddingCache(tmp_path)
        run_sweep(DOCUMENTS, LABELS, SMALL, cache=cache)
        first_pass_misses = cache.misses
        run_sweep(DOCUMENTS, LABELS, SMALL, cache=cache)
        assert cache.misses == first_pass_misses
        assert cache.hits > 0
