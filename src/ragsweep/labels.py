"""The labelled question set: the ground truth a sweep is scored against.

One JSON object per line. Relevance is recorded per *document*, never per chunk,
because chunk ids change with every chunk size and chunk-level labels would have to be
rewritten for every configuration in the sweep.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ragsweep.corpus import Document


class LabelError(Exception):
    """The labels file could not be read, with the line number that caused it."""


@dataclass(frozen=True)
class Label:
    id: str
    question: str
    relevant: frozenset[str]
    tags: tuple[str, ...] = ()


def _field(row: dict, name: str, line: int, *, required: bool = True):
    if name not in row:
        if required:
            raise LabelError(f"line {line}: missing field {name!r}")
        return None
    return row[name]


def _parse(row: dict, line: int) -> Label:
    if not isinstance(row, dict):
        raise LabelError(f"line {line}: expected a JSON object, got {type(row).__name__}")

    label_id = _field(row, "id", line)
    question = _field(row, "question", line)
    relevant = _field(row, "relevant", line)
    tags = _field(row, "tags", line, required=False) or []

    if not isinstance(label_id, str) or not label_id.strip():
        raise LabelError(f"line {line}: 'id' must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise LabelError(f"line {line}: 'question' must be a non-empty string")
    if not isinstance(relevant, list) or not relevant:
        raise LabelError(f"line {line}: 'relevant' must be a non-empty list of document ids")
    if not all(isinstance(item, str) and item.strip() for item in relevant):
        raise LabelError(f"line {line}: every entry in 'relevant' must be a non-empty string")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise LabelError(f"line {line}: 'tags' must be a list of strings")

    return Label(
        id=label_id.strip(),
        question=question.strip(),
        relevant=frozenset(relevant),
        tags=tuple(tags),
    )


def load_labels(path: Path) -> list[Label]:
    """Read a .jsonl labels file. Every error names the line that caused it."""
    path = Path(path)
    if not path.is_file():
        raise LabelError(f"labels file does not exist: {path}")

    labels: list[Label] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise LabelError(f"line {line_number}: invalid JSON ({error.msg})") from error

        label = _parse(row, line_number)
        if label.id in seen:
            raise LabelError(f"line {line_number}: duplicate id {label.id!r}")
        seen.add(label.id)
        labels.append(label)

    if not labels:
        raise LabelError(f"no labels found in {path}")
    return labels


def unknown_documents(
    labels: Iterable[Label], documents: Iterable[Document]
) -> dict[str, set[str]]:
    """Which labels point at document ids the corpus does not contain.

    A typo here silently caps every score in the sweep, and it looks exactly like bad
    retrieval, so it is worth catching before a run rather than after.
    """
    known = {document.id for document in documents}
    problems: dict[str, set[str]] = {}
    for label in labels:
        missing = set(label.relevant) - known
        if missing:
            problems[label.id] = missing
    return problems
