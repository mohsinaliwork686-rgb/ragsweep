"""Shared test data.

A tiny corpus with three clearly distinct documents, so retrieval tests have an
unambiguous right answer. The deliberately hard cases (near-duplicates, vocabulary
mismatch, multi-hop) belong in the real example corpus, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragsweep.corpus import Document
from ragsweep.labels import Label
from ragsweep.sweep import SweepConfig

DOCUMENTS = [
    Document(
        id="refunds.md",
        path=Path("refunds.md"),
        text=(
            "Refunds are processed within five business days of approval. "
            "Approval is required for any refund above five hundred dollars. "
            "Customers must request a refund within thirty days of delivery."
        ),
    ),
    Document(
        id="shipping.md",
        path=Path("shipping.md"),
        text=(
            "Shipping is free on orders over fifty dollars. "
            "Express delivery arrives the next working day. "
            "International orders take up to fourteen days."
        ),
    ),
    Document(
        id="leave.md",
        path=Path("leave.md"),
        text=(
            "Annual leave must be requested two weeks in advance. "
            "Unused leave does not carry over into the next year. "
            "Sick leave requires a note after three consecutive days."
        ),
    ),
]

LABELS = [
    Label("q1", "How long does a refund take?", frozenset({"refunds.md"}), ("policy",)),
    Label("q2", "When is shipping free?", frozenset({"shipping.md"}), ("policy",)),
    Label("q3", "How far ahead must leave be requested?", frozenset({"leave.md"}), ("hr",)),
]

#: Small enough to run in milliseconds, wide enough to exercise all three retrievers.
SMALL = SweepConfig(
    retrievers=("bm25", "dense", "hybrid"),
    strategies=("sentence",),
    chunk_sizes=(120,),
    overlaps=(0,),
    models=("hashing-256",),
    ks=(1, 3),
)


@pytest.fixture
def documents():
    return DOCUMENTS


@pytest.fixture
def labels():
    return LABELS


@pytest.fixture
def corpus_on_disk(tmp_path):
    """Write the shared corpus and labels to disk, for tests that go through the CLI."""
    import json

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for document in DOCUMENTS:
        (corpus / document.id).write_text(document.text, encoding="utf-8")

    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": label.id,
                    "question": label.question,
                    "relevant": sorted(label.relevant),
                    "tags": list(label.tags),
                }
            )
            for label in LABELS
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path, corpus, labels_path
