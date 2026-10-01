"""The ranked table, the CSV and the chart."""

from __future__ import annotations

import sys

import pytest

from conftest import DOCUMENTS, LABELS
from ragsweep.report import (
    ReportError,
    best,
    default_metric,
    recall_ks,
    render_table,
    sort_runs,
    to_csv,
    total_seconds,
    write_chart,
)
from ragsweep.sweep import RunResult, SweepConfig, run_sweep

WIDE = SweepConfig(
    retrievers=("bm25", "dense", "hybrid"),
    strategies=("fixed", "sentence"),
    chunk_sizes=(120, 240),
    overlaps=(0, 60),
    models=("hashing-512",),
    ks=(1, 3, 5),
)


@pytest.fixture
def runs():
    return run_sweep(DOCUMENTS, LABELS, WIDE)


def fake_run(run_id: str, score: float, seconds: float = 1.0) -> RunResult:
    return RunResult(
        id=run_id,
        config={"retriever": "bm25", "strategy": "fixed", "chunk_size": 256, "overlap": 0},
        metrics={"recall@1": score, "recall@5": score, "mrr": score},
        timing={"chunk_s": seconds, "embed_s": 0.0, "index_s": 0.0, "search_s": 0.0, "chunks": 4},
    )


class TestMetricDiscovery:
    def test_finds_every_k(self, runs):
        assert recall_ks(runs) == [1, 3, 5]

    def test_defaults_to_the_deepest_recall(self, runs):
        assert default_metric(runs) == "recall@5"

    def test_falls_back_to_mrr_when_there_is_no_recall(self):
        run = RunResult(id="x", config={}, metrics={"mrr": 1.0}, timing={})
        assert default_metric([run]) == "mrr"


class TestSorting:
    def test_best_score_first(self):
        ordered = sort_runs([fake_run("low", 0.2), fake_run("high", 0.9)], "recall@5")
        assert [run.id for run in ordered] == ["high", "low"]

    def test_ties_break_on_the_faster_run(self):
        ordered = sort_runs([fake_run("slow", 0.5, 9.0), fake_run("fast", 0.5, 0.1)], "recall@5")
        assert [run.id for run in ordered] == ["fast", "slow"]

    def test_remaining_ties_break_on_id_so_output_is_stable(self):
        ordered = sort_runs([fake_run("b", 0.5), fake_run("a", 0.5)], "recall@5")
        assert [run.id for run in ordered] == ["a", "b"]

    def test_unknown_metric_names_what_is_available(self, runs):
        with pytest.raises(ReportError, match="no metric 'recall@99'"):
            sort_runs(runs, "recall@99")

    def test_best_returns_the_top_run(self, runs):
        assert best(runs).id == sort_runs(runs)[0].id

    def test_best_of_nothing_is_none(self):
        assert best([]) is None


class TestTable:
    def test_has_one_column_per_k(self, runs):
        rendered = render_table(runs)
        for k in (1, 3, 5):
            assert f"recall@{k}" in rendered

    def test_marks_the_best_row(self, runs):
        lines = [line for line in render_table(runs).splitlines() if line.strip()]
        assert "<- best" in lines[1]
        assert sum("<- best" in line for line in lines) == 1

    def test_limit_shortens_the_table(self, runs):
        lines = [line for line in render_table(runs, limit=3).splitlines() if line.strip()]
        assert len(lines) == 4  # header plus three rows

    def test_model_column_is_hidden_for_a_single_model(self, runs):
        assert "model" not in render_table(runs).splitlines()[0]

    def test_model_column_appears_when_models_are_compared(self):
        config = SweepConfig(
            retrievers=("dense",), strategies=("fixed",), chunk_sizes=(120,),
            overlaps=(0,), models=("hashing-64", "hashing-128"), ks=(1,),
        )
        rendered = render_table(run_sweep(DOCUMENTS, LABELS, config))
        assert "model" in rendered.splitlines()[0]
        assert "hashing-64" in rendered

    def test_empty_runs_render_an_empty_table(self):
        assert render_table([]).strip() == ""


class TestCsv:
    def test_header_lists_every_metric(self, runs):
        header = to_csv(runs).splitlines()[0].split(",")
        assert {"id", "retriever", "chunk_size", "recall@5", "mrr", "total_s"} <= set(header)

    def test_one_row_per_run(self, runs):
        assert len(to_csv(runs).strip().splitlines()) == len(runs) + 1

    def test_rows_are_ordered_best_first(self, runs):
        first_id = to_csv(runs).splitlines()[1].split(",")[0]
        assert first_id == best(runs).id

    def test_missing_model_becomes_empty_not_none(self, runs):
        assert ",None," not in to_csv(runs)


class TestTiming:
    def test_total_adds_the_phases(self):
        run = RunResult(
            id="x", config={}, metrics={},
            timing={"chunk_s": 1.0, "embed_s": 2.0, "index_s": 0.5, "search_s": 0.25},
        )
        assert total_seconds(run) == pytest.approx(3.75)

    def test_missing_phases_count_as_zero(self):
        assert total_seconds(RunResult(id="x", config={}, metrics={}, timing={})) == 0.0


class TestChart:
    def test_writes_a_png(self, runs, tmp_path):
        path = write_chart(runs, tmp_path / "out" / "chart.png")
        assert path.is_file() and path.stat().st_size > 0

    def test_missing_matplotlib_is_actionable(self, runs, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        with pytest.raises(ReportError, match=r"pip install 'ragsweep\[chart\]'"):
            write_chart(runs, tmp_path / "chart.png")
