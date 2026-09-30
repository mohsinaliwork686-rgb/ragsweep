"""Running every combination, without doing the same work twice.

A naive nested loop re-chunks and re-embeds for every configuration. The work is shared
in layers instead, because each layer only depends on the ones above it:

    strategy + size + overlap  ->  chunk once
      + model                  ->  embed once (and the disk cache makes a repeat free)
        + retriever            ->  build the index once
          + k                  ->  free: retrieve max(k) once and slice for each k

That last line is why ``k`` is not part of a run. One run reports recall@1, @3, @5 and
@10 together, because computing them costs nothing once the top max(k) is in hand.

BM25 does not depend on the embedding model, so it runs once per chunking no matter how
many models are being swept. Without that, sweeping two models would silently double the
lexical work and put duplicate rows in the results table.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from ragsweep.chunking import STRATEGIES, chunk_corpus
from ragsweep.corpus import Document
from ragsweep.embedding import EmbeddingCache, embed_texts, get_embedder
from ragsweep.labels import Label
from ragsweep.metrics import summarise
from ragsweep.retrievers import (
    RETRIEVERS,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
)

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class SweepConfig:
    retrievers: tuple[str, ...] = RETRIEVERS
    strategies: tuple[str, ...] = STRATEGIES
    chunk_sizes: tuple[int, ...] = (256, 512, 1024)
    overlaps: tuple[int, ...] = (0, 64, 128)
    models: tuple[str, ...] = ("hashing-512",)
    ks: tuple[int, ...] = DEFAULT_KS

    def validate(self) -> None:
        unknown = set(self.retrievers) - set(RETRIEVERS)
        if unknown:
            raise ValueError(f"unknown retrievers: {sorted(unknown)}. Choose from {RETRIEVERS}")
        unknown = set(self.strategies) - set(STRATEGIES)
        if unknown:
            raise ValueError(f"unknown strategies: {sorted(unknown)}. Choose from {STRATEGIES}")
        if not self.ks or min(self.ks) <= 0:
            raise ValueError("ks must all be positive")
        if not self.chunk_sizes or min(self.chunk_sizes) <= 0:
            raise ValueError("chunk_sizes must all be positive")
        if not self.models:
            raise ValueError("at least one model is required")


@dataclass(frozen=True)
class RunSpec:
    retriever: str
    strategy: str
    chunk_size: int
    overlap: int
    #: None for retrievers that do not use embeddings.
    model: str | None

    @property
    def chunking(self) -> tuple[str, int, int]:
        return (self.strategy, self.chunk_size, self.overlap)

    @property
    def id(self) -> str:
        parts = [self.retriever, self.strategy, str(self.chunk_size), str(self.overlap)]
        if self.model:
            parts.append(self.model)
        return "-".join(parts)


@dataclass(frozen=True)
class CaseResult:
    id: str
    tags: tuple[str, ...]
    expected: tuple[str, ...]
    retrieved: tuple[str, ...]
    #: 1-based position of the first correct document, or None if it was never found.
    first_correct_rank: int | None
    #: Reciprocal rank. One number per case, which is what week 6 diffs.
    score: float


@dataclass(frozen=True)
class RunResult:
    id: str
    config: dict
    metrics: dict[str, float]
    timing: dict[str, float]
    cases: tuple[CaseResult, ...] = field(default_factory=tuple)


def plan_runs(config: SweepConfig) -> list[RunSpec]:
    """Every run the sweep will perform, in execution order.

    Combinations where overlap is at least the chunk size are dropped rather than
    raising: they are meaningless, and a sweep listing several sizes and several
    overlaps will always produce a few of them.
    """
    config.validate()
    specs: list[RunSpec] = []
    for strategy in config.strategies:
        for size in config.chunk_sizes:
            for overlap in config.overlaps:
                if overlap >= size:
                    continue
                if "bm25" in config.retrievers:
                    specs.append(RunSpec("bm25", strategy, size, overlap, None))
                for model in config.models:
                    for retriever in ("dense", "hybrid"):
                        if retriever in config.retrievers:
                            specs.append(RunSpec(retriever, strategy, size, overlap, model))
    return specs


def _evaluate(
    retriever: Retriever,
    document_ids: Sequence[str],
    labels: Sequence[Label],
    ks: Sequence[int],
) -> tuple[dict[str, float], tuple[CaseResult, ...], float]:
    deepest = max(ks)
    cases: list[CaseResult] = []
    pairs: list[tuple[frozenset[str], list[str]]] = []

    started = time.perf_counter()
    for label in labels:
        hits = retriever.search(label.question, deepest)
        retrieved = [document_ids[index] for index, _ in hits]
        pairs.append((label.relevant, retrieved))

        rank = next(
            (position for position, doc in enumerate(retrieved, start=1) if doc in label.relevant),
            None,
        )
        cases.append(
            CaseResult(
                id=label.id,
                tags=label.tags,
                expected=tuple(sorted(label.relevant)),
                retrieved=tuple(retrieved),
                first_correct_rank=rank,
                score=1.0 / rank if rank else 0.0,
            )
        )
    elapsed = time.perf_counter() - started

    return summarise(pairs, ks), tuple(cases), elapsed


def run_sweep(
    documents: Iterable[Document],
    labels: Sequence[Label],
    config: SweepConfig | None = None,
    *,
    cache: EmbeddingCache | None = None,
    on_run: Callable[[RunResult], None] | None = None,
) -> list[RunResult]:
    """Run every combination and return one result per run."""
    config = config or SweepConfig()
    documents = list(documents)
    specs = plan_runs(config)

    by_chunking: dict[tuple[str, int, int], list[RunSpec]] = {}
    for spec in specs:
        by_chunking.setdefault(spec.chunking, []).append(spec)

    results: list[RunResult] = []
    for (strategy, size, overlap), group in by_chunking.items():
        started = time.perf_counter()
        chunks = chunk_corpus(documents, strategy=strategy, size=size, overlap=overlap)
        chunk_seconds = time.perf_counter() - started

        texts = [chunk.text for chunk in chunks]
        document_ids = [chunk.document_id for chunk in chunks]

        lexical: BM25Retriever | None = None
        lexical_seconds = 0.0
        if any(spec.retriever in {"bm25", "hybrid"} for spec in group):
            started = time.perf_counter()
            lexical = BM25Retriever(texts)
            lexical_seconds = time.perf_counter() - started

        vectors_by_model: dict[str, tuple] = {}
        for spec in group:
            embed_seconds = 0.0
            index_seconds = 0.0

            if spec.retriever == "bm25":
                retriever: Retriever = lexical
                index_seconds = lexical_seconds
            else:
                if spec.model not in vectors_by_model:
                    embedder = get_embedder(spec.model)
                    started = time.perf_counter()
                    vectors = embed_texts(texts, embedder, cache)
                    vectors_by_model[spec.model] = (
                        vectors,
                        embedder,
                        time.perf_counter() - started,
                    )
                vectors, embedder, embed_seconds = vectors_by_model[spec.model]

                started = time.perf_counter()
                dense = DenseRetriever(vectors, embedder, cache)
                retriever = dense if spec.retriever == "dense" else HybridRetriever(dense, lexical)
                index_seconds = time.perf_counter() - started
                if spec.retriever == "hybrid":
                    index_seconds += lexical_seconds

            metrics, cases, search_seconds = _evaluate(retriever, document_ids, labels, config.ks)
            result = RunResult(
                id=spec.id,
                config={
                    "retriever": spec.retriever,
                    "strategy": spec.strategy,
                    "chunk_size": spec.chunk_size,
                    "overlap": spec.overlap,
                    "model": spec.model,
                },
                metrics=metrics,
                timing={
                    "chunk_s": round(chunk_seconds, 4),
                    "embed_s": round(embed_seconds, 4),
                    "index_s": round(index_seconds, 4),
                    "search_s": round(search_seconds, 4),
                    "chunks": len(chunks),
                },
                cases=cases,
            )
            results.append(result)
            if on_run is not None:
                on_run(result)

    return results
