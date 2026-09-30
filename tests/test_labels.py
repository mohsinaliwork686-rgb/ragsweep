"""Reading the ground truth file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragsweep.corpus import Document
from ragsweep.labels import LabelError, load_labels, unknown_documents


@pytest.fixture
def labels_file(tmp_path):
    def _write(*rows: str) -> Path:
        path = tmp_path / "labels.jsonl"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    return _write


def row(**overrides) -> str:
    base = {"id": "q1", "question": "How long does a refund take?", "relevant": ["refunds.md"]}
    base.update(overrides)
    return json.dumps(base)


class TestLoading:
    def test_reads_one_label(self, labels_file):
        (label,) = load_labels(labels_file(row()))
        assert label.id == "q1"
        assert label.relevant == frozenset({"refunds.md"})
        assert label.tags == ()

    def test_reads_tags(self, labels_file):
        (label,) = load_labels(labels_file(row(tags=["policy", "multi-hop"])))
        assert label.tags == ("policy", "multi-hop")

    def test_multiple_relevant_documents(self, labels_file):
        (label,) = load_labels(labels_file(row(relevant=["a.md", "b.md"])))
        assert label.relevant == frozenset({"a.md", "b.md"})

    def test_blank_lines_and_comments_are_skipped(self, labels_file):
        path = labels_file(row(), "", "# a note", row(id="q2"))
        assert [label.id for label in load_labels(path)] == ["q1", "q2"]

    def test_whitespace_is_trimmed(self, labels_file):
        (label,) = load_labels(labels_file(row(id="  q1  ", question="  padded  ")))
        assert label.id == "q1"
        assert label.question == "padded"


class TestErrorsNameTheLine:
    def test_missing_file(self, tmp_path):
        with pytest.raises(LabelError, match="does not exist"):
            load_labels(tmp_path / "nope.jsonl")

    def test_empty_file(self, labels_file):
        with pytest.raises(LabelError, match="no labels found"):
            load_labels(labels_file("", "  "))

    def test_invalid_json_says_which_line(self, labels_file):
        with pytest.raises(LabelError, match="line 2: invalid JSON"):
            load_labels(labels_file(row(), "{not json"))

    def test_not_an_object(self, labels_file):
        with pytest.raises(LabelError, match="line 1: expected a JSON object"):
            load_labels(labels_file("[1, 2, 3]"))

    @pytest.mark.parametrize("field", ["id", "question", "relevant"])
    def test_missing_required_field(self, labels_file, field):
        payload = json.loads(row())
        del payload[field]
        with pytest.raises(LabelError, match=f"line 1: missing field '{field}'"):
            load_labels(labels_file(json.dumps(payload)))

    def test_empty_relevant_list(self, labels_file):
        with pytest.raises(LabelError, match="non-empty list"):
            load_labels(labels_file(row(relevant=[])))

    def test_relevant_must_hold_strings(self, labels_file):
        with pytest.raises(LabelError, match="must be a non-empty string"):
            load_labels(labels_file(row(relevant=[123])))

    def test_blank_question(self, labels_file):
        with pytest.raises(LabelError, match="'question' must be"):
            load_labels(labels_file(row(question="   ")))

    def test_tags_must_be_strings(self, labels_file):
        with pytest.raises(LabelError, match="'tags' must be a list of strings"):
            load_labels(labels_file(row(tags=[1, 2])))

    def test_duplicate_ids_are_rejected(self, labels_file):
        with pytest.raises(LabelError, match="line 2: duplicate id 'q1'"):
            load_labels(labels_file(row(), row()))


class TestUnknownDocuments:
    def documents(self, *ids):
        return [Document(id=i, path=Path(i), text="content") for i in ids]

    def test_reports_ids_the_corpus_does_not_have(self, labels_file):
        labels = load_labels(labels_file(row(relevant=["refunds.md", "typo.md"])))
        assert unknown_documents(labels, self.documents("refunds.md")) == {"q1": {"typo.md"}}

    def test_silent_when_everything_matches(self, labels_file):
        labels = load_labels(labels_file(row()))
        assert unknown_documents(labels, self.documents("refunds.md")) == {}
