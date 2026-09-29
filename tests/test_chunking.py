"""Chunking.

The shared invariants are tested against all three strategies at once, because a
strategy that quietly loses or duplicates text would still look fine in a spot check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragsweep.chunking import STRATEGIES, chunk_document, chunk_spans
from ragsweep.corpus import Document

PROSE = (
    "Refunds are processed within five business days. "
    "Approval is required for amounts above five hundred dollars.\n\n"
    "Shipping is free on orders over fifty dollars. "
    "Returns must be made within thirty days of delivery.\n\n"
    "Expenses are reimbursed monthly. Receipts are mandatory for every claim."
)


def document(text: str = PROSE) -> Document:
    return Document(id="handbook/policy.md", path=Path("handbook/policy.md"), text=text)


@pytest.mark.parametrize("strategy", STRATEGIES)
class TestInvariantsHoldForEveryStrategy:
    """Whatever the strategy, these must never be violated."""

    def test_chunk_text_matches_its_own_span(self, strategy):
        doc = document()
        for chunk in chunk_document(doc, strategy=strategy, size=120, overlap=20):
            assert chunk.text == doc.text[chunk.start : chunk.end]

    def test_every_character_is_covered(self, strategy):
        spans = chunk_spans(PROSE, strategy=strategy, size=120, overlap=20)
        covered = set()
        for start, end in spans:
            covered.update(range(start, end))
        assert covered == set(range(len(PROSE)))

    def test_spans_move_forward(self, strategy):
        spans = chunk_spans(PROSE, strategy=strategy, size=120, overlap=20)
        starts = [start for start, _ in spans]
        assert starts == sorted(starts)
        assert len(set(spans)) == len(spans)

    def test_empty_document_makes_no_chunks(self, strategy):
        assert chunk_spans("", strategy=strategy, size=100, overlap=10) == []

    def test_document_shorter_than_one_chunk_makes_exactly_one(self, strategy):
        assert chunk_spans("short text", strategy=strategy, size=500, overlap=50) == [(0, 10)]

    def test_every_chunk_adds_new_text(self, strategy):
        """No chunk may sit wholly inside the one before it."""
        spans = chunk_spans(PROSE, strategy=strategy, size=60, overlap=25)
        for (_, previous_end), (_, end) in zip(spans, spans[1:], strict=False):
            assert end > previous_end

    def test_no_chunk_is_empty(self, strategy):
        for start, end in chunk_spans(PROSE, strategy=strategy, size=60, overlap=10):
            assert end > start

    @pytest.mark.parametrize("overlap", [0, 10, 59])
    def test_terminates_at_every_overlap(self, strategy, overlap):
        spans = chunk_spans(PROSE * 3, strategy=strategy, size=60, overlap=overlap)
        assert 0 < len(spans) < 1000

    def test_unicode_survives_intact(self, strategy):
        text = "Café costs 5 € 🎉 naïve façade. " * 12
        spans = chunk_spans(text, strategy=strategy, size=80, overlap=16)
        rebuilt = ""
        for start, end in spans:
            rebuilt += text[max(start, len(rebuilt)) : end]
        assert rebuilt == text

    def test_overlap_at_or_above_size_is_rejected(self, strategy):
        with pytest.raises(ValueError, match="must be smaller than size"):
            chunk_spans(PROSE, strategy=strategy, size=100, overlap=100)

    @pytest.mark.parametrize("size", [0, -5])
    def test_size_must_be_positive(self, strategy, size):
        with pytest.raises(ValueError, match="size must be positive"):
            chunk_spans(PROSE, strategy=strategy, size=size, overlap=0)

    def test_negative_overlap_is_rejected(self, strategy):
        with pytest.raises(ValueError, match="overlap must not be negative"):
            chunk_spans(PROSE, strategy=strategy, size=100, overlap=-1)


class TestFixed:
    def test_splits_on_exact_character_counts(self):
        assert chunk_spans("abcdefghij", strategy="fixed", size=4, overlap=0) == [
            (0, 4),
            (4, 8),
            (8, 10),
        ]

    def test_overlap_repeats_exactly_that_many_characters(self):
        spans = chunk_spans("abcdefghij", strategy="fixed", size=4, overlap=2)
        assert spans == [(0, 4), (2, 6), (4, 8), (6, 10)]
        text = "abcdefghij"
        assert text[spans[0][0] : spans[0][1]][-2:] == text[spans[1][0] : spans[1][1]][:2]

    def test_last_chunk_is_not_padded(self):
        spans = chunk_spans("abcde", strategy="fixed", size=4, overlap=0)
        assert spans[-1] == (4, 5)


class TestSentence:
    def test_keeps_sentences_whole(self):
        text = "One two three. Four five six. Seven eight nine."
        spans = chunk_spans(text, strategy="sentence", size=30, overlap=0)
        for start, end in spans:
            assert text[start:end].strip().endswith(".")

    def test_packs_several_short_sentences_together(self):
        text = "A b. C d. E f. G h."
        spans = chunk_spans(text, strategy="sentence", size=100, overlap=0)
        assert spans == [(0, len(text))]

    def test_a_sentence_longer_than_the_size_is_cut(self):
        text = "x" * 250 + "."
        spans = chunk_spans(text, strategy="sentence", size=100, overlap=0)
        assert len(spans) == 3
        assert all(end - start <= 100 for start, end in spans)

    def test_overlap_repeats_a_whole_sentence(self):
        text = "Alpha one. Bravo two. Charlie three. Delta four."
        spans = chunk_spans(text, strategy="sentence", size=40, overlap=20)
        assert len(spans) > 1
        first, second = text[spans[0][0] : spans[0][1]], text[spans[1][0] : spans[1][1]]
        carried = second.split(".")[0]
        assert carried.strip()
        assert carried in first

    def test_overlap_is_dropped_rather_than_stalling(self):
        """When the size leaves no room to overlap and still advance, advance."""
        text = "Alpha one. Bravo two. Charlie three. Delta four."
        spans = chunk_spans(text, strategy="sentence", size=25, overlap=12)
        assert spans == [(0, 22), (22, 37), (37, 48)]


class TestRecursive:
    def test_prefers_paragraph_boundaries(self):
        text = "First para line.\n\nSecond para line.\n\nThird para line."
        spans = chunk_spans(text, strategy="recursive", size=25, overlap=0)
        assert [text[s:e].strip() for s, e in spans] == [
            "First para line.",
            "Second para line.",
            "Third para line.",
        ]

    def test_falls_back_to_lines_when_paragraphs_are_too_big(self):
        text = "line one\nline two\nline three\nline four"
        spans = chunk_spans(text, strategy="recursive", size=20, overlap=0)
        assert len(spans) > 1
        assert all(end - start <= 20 for start, end in spans)

    def test_falls_back_to_hard_cuts_when_nothing_else_works(self):
        text = "x" * 100
        spans = chunk_spans(text, strategy="recursive", size=30, overlap=0)
        assert [end - start for start, end in spans] == [30, 30, 30, 10]


class TestChunkDocument:
    def test_ids_and_indexes_are_sequential(self):
        chunks = chunk_document(document(), strategy="fixed", size=60, overlap=0)
        assert [c.index for c in chunks] == list(range(len(chunks)))
        assert [c.id for c in chunks] == [f"handbook/policy.md#{i}" for i in range(len(chunks))]

    def test_every_chunk_knows_its_document(self):
        chunks = chunk_document(document(), strategy="sentence", size=60, overlap=10)
        assert {c.document_id for c in chunks} == {"handbook/policy.md"}

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            chunk_spans(PROSE, strategy="magic", size=50, overlap=0)
