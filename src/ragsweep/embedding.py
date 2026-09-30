"""Turning text into vectors, and not doing it twice.

Every embedder returns L2-normalised rows, so cosine similarity is just a dot product
and the retrievers never have to think about normalisation again.

The disk cache is what makes iterating bearable. A sweep re-embeds the same chunks for
every retriever and every repeat run; without a cache each pass costs minutes and I
would stop re-running it, which is the point at which a measurement tool stops being
used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingError(Exception):
    """An embedding model was asked for but is not available."""


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a list of strings into a matrix of unit vectors."""

    @property
    def name(self) -> str:
        """Stable identifier. Part of the cache key, so it must change when the model does."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), dimensions), L2-normalised, float32."""


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric runs. Shared with BM25 so both see the same words."""
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Feature hashing. No model, no download, identical output on every machine.

    This exists so the test suite never touches the network and so `ragsweep` runs
    before anyone installs the extras. It is lexical, not semantic, so it behaves like
    a weaker BM25 — useful as a floor to compare against, not as a real embedder.
    """

    def __init__(self, dimensions: int = 256, seed: int = 0) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.seed = seed

    @property
    def name(self) -> str:
        return f"hashing-{self.dimensions}-{self.seed}"

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(f"{self.seed}:{token}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenise(text):
                matrix[row, self._bucket(token)] += 1.0
        return _normalise(matrix)


class SentenceTransformerEmbedder:
    """A real semantic embedder. Needs `pip install ragsweep[dense]`."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise EmbeddingError(
                    f"{self._model_name!r} needs sentence-transformers. "
                    "Install it with: pip install 'ragsweep[dense]'"
                ) from error
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._load().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalise(np.asarray(vectors, dtype=np.float32))


def get_embedder(name: str) -> Embedder:
    """Resolve a model name from the config to an embedder."""
    if name.startswith("hashing"):
        parts = name.split("-")
        dimensions = int(parts[1]) if len(parts) > 1 else 256
        seed = int(parts[2]) if len(parts) > 2 else 0
        return HashingEmbedder(dimensions=dimensions, seed=seed)
    return SentenceTransformerEmbedder(name)


# ----------------------------------------------------------------------------- cache


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "__", model)


class EmbeddingCache:
    """One folder per model: the vectors as .npy, the keys as .json.

    A corrupt or half-written store is treated as empty and rebuilt rather than raising.
    Losing a cache costs time; crashing on it costs the whole run.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def _folder(self, model: str) -> Path:
        return self.root / _slug(model)

    def read(self, model: str) -> tuple[dict[str, int], np.ndarray | None]:
        folder = self._folder(model)
        keys_path, vectors_path = folder / "keys.json", folder / "vectors.npy"
        if not keys_path.is_file() or not vectors_path.is_file():
            return {}, None
        try:
            keys = json.loads(keys_path.read_text(encoding="utf-8"))
            vectors = np.load(vectors_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}, None
        if not isinstance(keys, list) or len(keys) != len(vectors):
            return {}, None
        return {key: row for row, key in enumerate(keys)}, vectors

    def write(self, model: str, keys: list[str], vectors: np.ndarray) -> None:
        """Merge new rows into the store. Written to a temp file then renamed."""
        if not keys:
            return
        folder = self._folder(model)
        folder.mkdir(parents=True, exist_ok=True)

        existing_index, existing_vectors = self.read(model)
        fresh = [(key, row) for row, key in enumerate(keys) if key not in existing_index]
        if not fresh:
            return

        added = vectors[[row for _, row in fresh]]
        merged_keys = list(existing_index) + [key for key, _ in fresh]
        merged_vectors = added if existing_vectors is None else np.vstack([existing_vectors, added])

        self._atomic_write(folder / "vectors.npy", lambda p: np.save(p, merged_vectors))
        self._atomic_write(
            folder / "keys.json",
            lambda p: p.write_text(json.dumps(merged_keys), encoding="utf-8"),
        )

    @staticmethod
    def _atomic_write(target: Path, writer) -> None:
        temporary = target.with_suffix(target.suffix + ".tmp")
        writer(temporary)
        # np.save appends .npy when the name lacks it; find whatever was actually written
        if not temporary.exists() and temporary.with_suffix(temporary.suffix + ".npy").exists():
            temporary = temporary.with_suffix(temporary.suffix + ".npy")
        os.replace(temporary, target)


def embed_texts(
    texts: list[str], embedder: Embedder, cache: EmbeddingCache | None = None
) -> np.ndarray:
    """Embed every text, reusing whatever the cache already holds.

    Duplicate texts are encoded once. Order is preserved.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if cache is None:
        return embedder.encode(texts)

    keys = [cache_key(embedder.name, text) for text in texts]
    index, stored = cache.read(embedder.name)

    wanted: dict[str, str] = {}
    for key, text in zip(keys, texts, strict=True):
        if key not in index and key not in wanted:
            wanted[key] = text

    cache.hits += sum(1 for key in keys if key in index)
    cache.misses += len(keys) - sum(1 for key in keys if key in index)

    if wanted:
        fresh_keys = list(wanted)
        cache.write(embedder.name, fresh_keys, embedder.encode([wanted[k] for k in fresh_keys]))
        index, stored = cache.read(embedder.name)

    return np.stack([stored[index[key]] for key in keys])
