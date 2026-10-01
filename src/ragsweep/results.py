"""The results file: what a sweep writes, and what reads it back.

Week 6 (`evalboard`) consumes this file, so it is versioned from the first release and
carries per-case detail rather than only the aggregates. Averages are exactly what hides
a regression, so a file holding only averages would make `evalboard` pointless.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragsweep import __version__
from ragsweep.corpus import Document
from ragsweep.labels import Label
from ragsweep.sweep import CaseResult, RunResult

SCHEMA_VERSION = 1
TOOL = "ragsweep"


class ResultsError(Exception):
    """A results file could not be read."""


def corpus_fingerprint(documents: Sequence[Document]) -> str:
    """A short hash of the corpus content.

    Two results files are only comparable if they came from the same documents. This is
    what lets `evalboard` refuse to diff two runs over different corpora instead of
    quietly reporting nonsense.
    """
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda d: d.id):
        digest.update(document.id.encode())
        digest.update(b"\x00")
        digest.update(document.text.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def _case_to_dict(case: CaseResult) -> dict[str, Any]:
    return {
        "id": case.id,
        "tags": list(case.tags),
        "expected": list(case.expected),
        "retrieved": list(case.retrieved),
        "first_correct_rank": case.first_correct_rank,
        "score": round(case.score, 6),
    }


def _run_to_dict(run: RunResult) -> dict[str, Any]:
    return {
        "id": run.id,
        "config": run.config,
        "metrics": {name: round(value, 6) for name, value in run.metrics.items()},
        "timing": run.timing,
        "cases": [_case_to_dict(case) for case in run.cases],
    }


def build_payload(
    runs: Sequence[RunResult],
    *,
    documents: Sequence[Document],
    labels: Sequence[Label],
    corpus_path: Path | str,
    labels_path: Path | str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    stamp = (created_at or datetime.now(UTC)).replace(microsecond=0)
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL,
        "tool_version": __version__,
        "created_at": stamp.isoformat().replace("+00:00", "Z"),
        "corpus": {
            "path": str(corpus_path),
            "documents": len(documents),
            "fingerprint": corpus_fingerprint(documents),
        },
        "labels": {"path": str(labels_path), "questions": len(labels)},
        "runs": [_run_to_dict(run) for run in runs],
    }


def write_results(path: Path | str, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_results(path: Path | str) -> dict[str, Any]:
    """Read a results file, failing with a message that says what is wrong."""
    path = Path(path)
    if not path.is_file():
        raise ResultsError(f"results file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ResultsError(f"{path} is not valid JSON ({error.msg})") from error

    if not isinstance(payload, dict):
        raise ResultsError(f"{path}: expected a JSON object at the top level")

    version = payload.get("schema")
    if version is None:
        raise ResultsError(f"{path}: missing 'schema' field. Was this written by ragsweep?")
    if version != SCHEMA_VERSION:
        raise ResultsError(
            f"{path} is schema {version}, but ragsweep {__version__} understands "
            f"schema {SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("runs"), list):
        raise ResultsError(f"{path}: 'runs' must be a list")
    return payload


def runs_from_payload(payload: dict[str, Any]) -> list[RunResult]:
    """Rebuild run objects so `ragsweep report` can re-render a saved sweep."""
    runs: list[RunResult] = []
    for position, raw in enumerate(payload.get("runs", [])):
        if not isinstance(raw, dict):
            raise ResultsError(f"run {position}: expected an object")
        for required in ("id", "config", "metrics"):
            if required not in raw:
                raise ResultsError(f"run {position}: missing field {required!r}")
        runs.append(
            RunResult(
                id=raw["id"],
                config=raw["config"],
                metrics=raw["metrics"],
                timing=raw.get("timing", {}),
                cases=tuple(
                    CaseResult(
                        id=case["id"],
                        tags=tuple(case.get("tags", ())),
                        expected=tuple(case.get("expected", ())),
                        retrieved=tuple(case.get("retrieved", ())),
                        first_correct_rank=case.get("first_correct_rank"),
                        score=case.get("score", 0.0),
                    )
                    for case in raw.get("cases", [])
                ),
            )
        )
    return runs
