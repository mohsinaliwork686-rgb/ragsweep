"""Turning runs into something a person can read.

The table is the product. Everything else in this tool exists so that one ranked table
answers the question "which settings should I use", in one screen, without a UI.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ragsweep.sweep import RunResult

_RECALL = re.compile(r"^recall@(\d+)$")
_AT_K = re.compile(r"^(.*)@(\d+)$")


class ReportError(Exception):
    """A report could not be produced."""


def metric_order(name: str) -> tuple[str, int]:
    """Sort metric names so recall@10 lands after recall@5, not after recall@1.

    Plain alphabetical ordering puts "@10" between "@1" and "@3", which makes a CSV
    opened in a spreadsheet quietly misleading.
    """
    match = _AT_K.match(name)
    return (match.group(1), int(match.group(2))) if match else (name, -1)


def recall_ks(runs: Sequence[RunResult]) -> list[int]:
    """Which k values the runs actually carry, smallest first."""
    found: set[int] = set()
    for run in runs:
        for name in run.metrics:
            match = _RECALL.match(name)
            if match:
                found.add(int(match.group(1)))
    return sorted(found)


def default_metric(runs: Sequence[RunResult]) -> str:
    """Rank on the deepest recall by default: it is the one that bounds answer quality."""
    ks = recall_ks(runs)
    return f"recall@{ks[-1]}" if ks else "mrr"


def total_seconds(run: RunResult) -> float:
    return sum(
        float(run.timing.get(name, 0.0))
        for name in ("chunk_s", "embed_s", "index_s", "search_s")
    )


def sort_runs(runs: Sequence[RunResult], metric: str | None = None) -> list[RunResult]:
    """Best first. Ties break on the faster run, then on id, so output is stable."""
    metric = metric or default_metric(runs)
    missing = [run.id for run in runs if metric not in run.metrics]
    if missing:
        raise ReportError(
            f"no metric {metric!r} on run {missing[0]!r}. "
            f"Available: {sorted(runs[0].metrics) if runs else 'none'}"
        )
    return sorted(runs, key=lambda run: (-run.metrics[metric], total_seconds(run), run.id))


def best(runs: Sequence[RunResult], metric: str | None = None) -> RunResult | None:
    return sort_runs(runs, metric)[0] if runs else None


def build_table(
    runs: Sequence[RunResult], *, metric: str | None = None, limit: int | None = None
) -> Table:
    if not runs:
        # A header with no rows under it reads as a bug. Let the caller say "no runs".
        return Table(box=None)

    metric = metric or default_metric(runs)
    ordered = sort_runs(runs, metric)
    shown = ordered[:limit] if limit else ordered
    models = {run.config.get("model") for run in runs} - {None}

    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("retriever")
    table.add_column("strategy")
    table.add_column("chunk", justify="right")
    table.add_column("ovl", justify="right")
    if len(models) > 1:
        table.add_column("model")
    # Short headers and no chunk count: the table has to fit in an 80 column terminal,
    # or rich truncates the columns and the numbers become unreadable. The full detail
    # is in the results file for anyone who wants it.
    for k in recall_ks(runs):
        table.add_column(f"r@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("time", justify="right")
    table.add_column("")

    for position, run in enumerate(shown):
        config = run.config
        row = [
            str(config.get("retriever", "")),
            str(config.get("strategy", "")),
            str(config.get("chunk_size", "")),
            str(config.get("overlap", "")),
        ]
        if len(models) > 1:
            row.append(str(config.get("model") or "-"))
        row += [f"{run.metrics.get(f'recall@{k}', 0.0):.2f}" for k in recall_ks(runs)]
        row += [
            f"{run.metrics.get('mrr', 0.0):.2f}",
            f"{total_seconds(run):.1f}s",
            "*" if position == 0 else "",
        ]
        table.add_row(*row)
    return table


def render_table(
    runs: Sequence[RunResult],
    *,
    metric: str | None = None,
    limit: int | None = None,
    width: int = 120,
) -> str:
    console = Console(file=io.StringIO(), width=width, no_color=True, highlight=False)
    console.print(build_table(runs, metric=metric, limit=limit))
    return console.file.getvalue()


def to_csv(runs: Sequence[RunResult], metric: str | None = None) -> str:
    """One row per run, flat, for a spreadsheet or someone else's tooling."""
    ordered = sort_runs(runs, metric)
    metric_names = sorted({name for run in runs for name in run.metrics}, key=metric_order)
    columns = [
        "id", "retriever", "strategy", "chunk_size", "overlap", "model",
        *metric_names, "chunks", "total_s",
    ]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for run in ordered:
        row = {
            "id": run.id,
            "retriever": run.config.get("retriever"),
            "strategy": run.config.get("strategy"),
            "chunk_size": run.config.get("chunk_size"),
            "overlap": run.config.get("overlap"),
            "model": run.config.get("model") or "",
            "chunks": run.timing.get("chunks", ""),
            "total_s": round(total_seconds(run), 4),
        }
        row.update({name: run.metrics.get(name, "") for name in metric_names})
        writer.writerow(row)
    return buffer.getvalue()


def write_chart(
    runs: Sequence[RunResult], path: Path | str, metric: str | None = None
) -> Path:
    """One line per retriever: how the score moves with chunk size.

    This is the picture the whole tool is for. If the line is flat, chunk size does not
    matter for this corpus and the choice can be made on speed instead.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ReportError(
            "charts need matplotlib. Install it with: pip install 'ragsweep[chart]'"
        ) from error

    metric = metric or default_metric(runs)
    grouped: dict[str, dict[int, list[float]]] = {}
    for run in runs:
        retriever = str(run.config.get("retriever", "?"))
        size = int(run.config.get("chunk_size", 0))
        grouped.setdefault(retriever, {}).setdefault(size, []).append(run.metrics.get(metric, 0.0))

    figure, axes = plt.subplots(figsize=(7, 4.5))
    for retriever in sorted(grouped):
        sizes = sorted(grouped[retriever])
        scores = [sum(grouped[retriever][s]) / len(grouped[retriever][s]) for s in sizes]
        axes.plot(sizes, scores, marker="o", label=retriever)

    axes.set_xlabel("chunk size (characters)")
    axes.set_ylabel(metric)
    axes.set_title(f"{metric} by chunk size")
    axes.set_ylim(0, 1.02)
    axes.grid(alpha=0.3)
    axes.legend(title="retriever")
    figure.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
