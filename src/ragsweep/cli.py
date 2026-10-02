"""Command line entry point.

Everything here is argument handling, file loading and printing. The logic lives in
the modules it calls, which is what lets the whole sweep be tested without a terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

from rich.console import Console

from ragsweep import __version__
from ragsweep.corpus import CorpusError, load_corpus
from ragsweep.embedding import EmbeddingCache, EmbeddingError
from ragsweep.labels import LabelError, load_labels, unknown_documents
from ragsweep.report import ReportError, best, build_table, to_csv, write_chart
from ragsweep.results import (
    ResultsError,
    build_payload,
    read_results,
    runs_from_payload,
    write_results,
)
from ragsweep.sweep import RunResult, SweepConfig, plan_runs, run_sweep

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

DEFAULT_CONFIG = "sweep.toml"

SWEEP_TEMPLATE = """\
# Which documents to search, and the questions to score against.
corpus = "./corpus"
labels = "./labels.jsonl"

# Everything below is swept: every valid combination is measured.
retrievers  = ["bm25", "dense", "hybrid"]
strategies  = ["fixed", "sentence", "recursive"]
chunk_sizes = [256, 512, 1024]
overlaps    = [0, 64, 128]

# "hashing-512" needs nothing extra and is a lexical baseline, not a real embedder.
# For semantic retrieval: pip install 'ragsweep[dense]', then name a real model here.
models = ["hashing-512"]

k = [1, 3, 5, 10]

# Embedding vectors are cached here, so a repeat sweep costs seconds.
cache = ".cache/ragsweep"
"""

#: Starter labels. The tags name the cases worth having: a question needing two
#: documents, and one whose wording shares no words with the document that answers it.
SAMPLE_LABELS = [
    {
        "id": "q1",
        "question": "How long does a refund take?",
        "relevant": ["refunds.md"],
        "tags": ["policy"],
    },
    {
        "id": "q2",
        "question": "Who approves large refunds?",
        "relevant": ["refunds.md", "approvals.md"],
        "tags": ["multi-hop"],
    },
    {
        "id": "q3",
        "question": "How do I get my money back?",
        "relevant": ["refunds.md"],
        "tags": ["vocabulary-mismatch"],
    },
]
LABELS_TEMPLATE = "\n".join(json.dumps(row) for row in SAMPLE_LABELS) + "\n"


class ConfigError(Exception):
    """The sweep file could not be used."""


def load_sweep_file(path: Path) -> tuple[SweepConfig, Path, Path, Path | None]:
    """Read sweep.toml into (config, corpus path, labels path, cache path)."""
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}. Run 'ragsweep init' to make one.")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error

    base = path.parent
    defaults = SweepConfig()
    config = SweepConfig(
        retrievers=tuple(data.get("retrievers", defaults.retrievers)),
        strategies=tuple(data.get("strategies", defaults.strategies)),
        chunk_sizes=tuple(data.get("chunk_sizes", defaults.chunk_sizes)),
        overlaps=tuple(data.get("overlaps", defaults.overlaps)),
        models=tuple(data.get("models", defaults.models)),
        ks=tuple(data.get("k", defaults.ks)),
    )
    cache = data.get("cache")
    return (
        config,
        base / data.get("corpus", "./corpus"),
        base / data.get("labels", "./labels.jsonl"),
        base / cache if cache else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragsweep",
        description="Measure which retrieval settings actually find the right passage.",
    )
    parser.add_argument("--version", action="version", version=f"ragsweep {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    initialise = commands.add_parser("init", help="write a starter sweep.toml and labels file")
    initialise.add_argument("--path", type=Path, default=Path("."), help="where to write them")
    initialise.add_argument("--force", action="store_true", help="overwrite existing files")

    run = commands.add_parser("run", help="run the sweep")
    run.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    run.add_argument("--corpus", type=Path, help="override the corpus folder")
    run.add_argument("--labels", type=Path, help="override the labels file")
    run.add_argument("--out", type=Path, default=Path("results.json"))
    run.add_argument(
        "--model",
        action="append",
        dest="models",
        help="override the models from the config. Repeat for several",
    )
    run.add_argument("--cache", type=Path, help="override the embedding cache folder")
    run.add_argument("--no-cache", action="store_true", help="re-embed everything")
    run.add_argument("--metric", help="rank the table on this metric")
    run.add_argument("--limit", type=int, help="only show this many rows")
    run.add_argument("--quiet", action="store_true", help="no progress output")

    report = commands.add_parser("report", help="re-print a saved results file")
    report.add_argument("results", type=Path)
    report.add_argument("--format", dest="fmt", choices=("table", "csv", "json"), default="table")
    report.add_argument("--chart", type=Path, help="also write a PNG chart here")
    report.add_argument("--metric", help="rank on this metric")
    report.add_argument("--limit", type=int)
    return parser


def _init(args, console: Console) -> int:
    args.path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, body in ((DEFAULT_CONFIG, SWEEP_TEMPLATE), ("labels.jsonl", LABELS_TEMPLATE)):
        target = args.path / name
        if target.exists() and not args.force:
            console.print(f"[yellow]exists, left alone:[/] {target} (--force to overwrite)")
            continue
        target.write_text(body, encoding="utf-8")
        written.append(target)

    corpus = args.path / "corpus"
    corpus.mkdir(exist_ok=True)
    for path in written:
        console.print(f"wrote {path}")
    console.print(f"\nPut your documents in {corpus}/, edit labels.jsonl, then: ragsweep run")
    return EXIT_OK


def _show(runs: list[RunResult], console: Console, metric: str | None, limit: int | None) -> None:
    if not runs:
        console.print("[yellow]no runs to report[/]")
        return
    # Print the table object, not a pre-rendered string: rich re-wraps strings at the
    # terminal width, which splits every row in half on a narrow terminal.
    console.print(build_table(runs, metric=metric, limit=limit))
    winner = best(runs, metric)
    config = winner.config
    console.print(
        f"Best: {config['retriever']} / {config['strategy']} / "
        f"chunk {config['chunk_size']} / overlap {config['overlap']}"
    )


def _run(args, console: Console) -> int:
    config, corpus_path, labels_path, cache_path = load_sweep_file(args.config)
    corpus_path = args.corpus or corpus_path
    labels_path = args.labels or labels_path
    cache_path = None if args.no_cache else (args.cache or cache_path)
    if args.models:
        config = replace(config, models=tuple(args.models))

    documents = load_corpus(corpus_path)
    labels = load_labels(labels_path)

    missing = unknown_documents(labels, documents)
    if missing:
        console.print(
            f"[yellow]warning:[/] {len(missing)} label(s) point at documents the corpus "
            "does not contain. Every score below is capped by this.",
            style=None,
        )
        for label_id, ids in sorted(missing.items())[:5]:
            console.print(f"  {label_id}: {', '.join(sorted(ids))}")

    total = len(plan_runs(config))
    console.print(
        f"{len(documents)} documents, {len(labels)} questions, {total} runs\n"
    )

    cache = EmbeddingCache(cache_path) if cache_path else None
    completed = 0

    def progress(result: RunResult) -> None:
        nonlocal completed
        completed += 1
        if not args.quiet:
            score = result.metrics.get("mrr", 0.0)
            console.print(f"  [{completed}/{total}] {result.id}  mrr={score:.2f}", highlight=False)

    runs = run_sweep(documents, labels, config, cache=cache, on_run=progress)

    payload = build_payload(
        runs,
        documents=documents,
        labels=labels,
        corpus_path=corpus_path,
        labels_path=labels_path,
    )
    write_results(args.out, payload)

    console.print()
    _show(runs, console, args.metric, args.limit)
    console.print(f"\nWritten to {args.out}")
    return EXIT_OK


def _report(args, console: Console) -> int:
    payload = read_results(args.results)
    runs = runs_from_payload(payload)

    if args.fmt == "json":
        print(json.dumps(payload, indent=2))
    elif args.fmt == "csv":
        print(to_csv(runs, args.metric), end="")
    else:
        _show(runs, console, args.metric, args.limit)

    if args.chart:
        console.print(f"chart written to {write_chart(runs, args.chart, args.metric)}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    handlers = {"init": _init, "run": _run, "report": _report}
    try:
        return handlers[args.command](args, console)
    except (
        ConfigError,
        CorpusError,
        LabelError,
        ResultsError,
        ReportError,
        EmbeddingError,
    ) as error:
        print(f"ragsweep: {error}", file=sys.stderr)
        return EXIT_FAILED
    except ValueError as error:
        print(f"ragsweep: {error}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
