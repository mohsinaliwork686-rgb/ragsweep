"""Retrieval strategies. Each one turns a query into ranked chunk indexes."""

from __future__ import annotations

from ragsweep.retrievers.base import Hit, Retriever, top_k
from ragsweep.retrievers.bm25 import BM25Retriever
from ragsweep.retrievers.dense import DenseRetriever
from ragsweep.retrievers.hybrid import HybridRetriever

RETRIEVERS: tuple[str, ...] = ("bm25", "dense", "hybrid")

__all__ = [
    "RETRIEVERS",
    "BM25Retriever",
    "DenseRetriever",
    "Hit",
    "HybridRetriever",
    "Retriever",
    "top_k",
]
