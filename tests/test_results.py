"""The results file.

Week 6 reads this, so the schema check and the per-case detail matter more than
anything else here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import DOCUMENTS, LABELS, SMALL
from ragsweep.results import (
    SCHEMA_VERSION,
    ResultsError,
    build_payload,
    corpus_fingerprint,
    read_results,
    runs_from_payload,
    write_results,
)
from ragsweep.sweep import run_sweep


@pytest.fixture
def payload():
    runs = run_sweep(DOCUMENTS, LABELS, SMALL)
    return build_payload(
        runs,
        documents=DOCUMENTS,
        labels=LABELS,
        corpus_path="./corpus",
        labels_path="./labels.jsonl",
        created_at=datetime(2026, 10, 1, 9, 30, tzinfo=UTC),
    )


class TestPayload:
    def test_carries_the_schema_version(self, payload):
        assert payload["schema"] == SCHEMA_VERSION
        assert payload["tool"] == "ragsweep"

    def test_timestamp_is_iso_utc(self, payload):
        assert payload["created_at"] == "2026-10-01T09:30:00Z"

    def test_records_the_inputs(self, payload):
        assert payload["corpus"]["documents"] == len(DOCUMENTS)
        assert payload["labels"]["questions"] == len(LABELS)
        assert payload["corpus"]["path"] == "./corpus"

    def test_one_entry_per_run(self, payload):
        assert len(payload["runs"]) == len(run_sweep(DOCUMENTS, LABELS, SMALL))

    def test_every_run_keeps_its_cases(self, payload):
        for run in payload["runs"]:
            assert len(run["cases"]) == len(LABELS)
            assert set(run["cases"][0]) == {
                "id", "tags", "expected", "retrieved", "first_correct_rank", "score",
            }

    def test_is_json_serialisable(self, payload):
        assert json.loads(json.dumps(payload))["schema"] == SCHEMA_VERSION


class TestFingerprint:
    def test_same_documents_give_the_same_fingerprint(self):
        assert corpus_fingerprint(DOCUMENTS) == corpus_fingerprint(list(reversed(DOCUMENTS)))

    def test_changed_text_changes_the_fingerprint(self):
        from ragsweep.corpus import Document

        altered = [Document(DOCUMENTS[0].id, DOCUMENTS[0].path, "different text"), *DOCUMENTS[1:]]
        assert corpus_fingerprint(altered) != corpus_fingerprint(DOCUMENTS)

    def test_changed_id_changes_the_fingerprint(self):
        from ragsweep.corpus import Document

        altered = [Document("renamed.md", Path("renamed.md"), DOCUMENTS[0].text), *DOCUMENTS[1:]]
        assert corpus_fingerprint(altered) != corpus_fingerprint(DOCUMENTS)


class TestRoundTrip:
    def test_write_then_read(self, tmp_path, payload):
        path = write_results(tmp_path / "results.json", payload)
        assert read_results(path)["schema"] == SCHEMA_VERSION

    def test_runs_survive_the_round_trip(self, tmp_path, payload):
        path = write_results(tmp_path / "out" / "results.json", payload)
        restored = runs_from_payload(read_results(path))
        original = run_sweep(DOCUMENTS, LABELS, SMALL)
        assert [r.id for r in restored] == [r.id for r in original]
        assert restored[0].metrics == pytest.approx(original[0].metrics, abs=1e-6)

    def test_cases_survive_the_round_trip(self, tmp_path, payload):
        path = write_results(tmp_path / "results.json", payload)
        restored = runs_from_payload(read_results(path))
        assert [c.id for c in restored[0].cases] == [label.id for label in LABELS]
        assert restored[0].cases[0].expected == ("refunds.md",)

    def test_missing_directories_are_created(self, tmp_path, payload):
        path = write_results(tmp_path / "deep" / "nested" / "results.json", payload)
        assert path.is_file()


class TestReadErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ResultsError, match="does not exist"):
            read_results(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ResultsError, match="not valid JSON"):
            read_results(path)

    def test_not_an_object(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ResultsError, match="expected a JSON object"):
            read_results(path)

    def test_missing_schema_field(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"runs": []}), encoding="utf-8")
        with pytest.raises(ResultsError, match="missing 'schema'"):
            read_results(path)

    def test_a_future_schema_says_so_plainly(self, tmp_path):
        path = tmp_path / "future.json"
        path.write_text(json.dumps({"schema": 2, "runs": []}), encoding="utf-8")
        with pytest.raises(ResultsError, match="is schema 2, but ragsweep .* schema 1"):
            read_results(path)

    def test_runs_must_be_a_list(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema": 1, "runs": {}}), encoding="utf-8")
        with pytest.raises(ResultsError, match="'runs' must be a list"):
            read_results(path)

    def test_a_malformed_run_names_its_position(self):
        with pytest.raises(ResultsError, match="run 1: missing field 'metrics'"):
            runs_from_payload({"runs": [
                {"id": "a", "config": {}, "metrics": {}},
                {"id": "b", "config": {}},
            ]})
