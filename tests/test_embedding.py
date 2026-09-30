"""Embedders and the disk cache.

Nothing here downloads a model or touches the network.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ragsweep.embedding import (
    EmbeddingCache,
    EmbeddingError,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cache_key,
    embed_texts,
    get_embedder,
    tokenise,
)


class CountingEmbedder:
    """Wraps an embedder and records how many texts it was actually asked to encode."""

    def __init__(self, inner=None):
        self.inner = inner or HashingEmbedder(dimensions=32)
        self.encoded: list[str] = []

    @property
    def name(self) -> str:
        return self.inner.name

    def encode(self, texts):
        self.encoded.extend(texts)
        return self.inner.encode(texts)


@pytest.fixture
def cache(tmp_path):
    return EmbeddingCache(tmp_path / "cache")


class TestHashingEmbedder:
    def test_rows_are_unit_length(self):
        vectors = HashingEmbedder(dimensions=32).encode(["hello world", "another one"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_shape_follows_dimensions(self):
        assert HashingEmbedder(dimensions=17).encode(["a", "b", "c"]).shape == (3, 17)

    def test_identical_across_instances(self):
        first = HashingEmbedder(dimensions=32).encode(["stable output"])
        second = HashingEmbedder(dimensions=32).encode(["stable output"])
        assert np.array_equal(first, second)

    def test_seed_changes_the_result(self):
        a = HashingEmbedder(dimensions=32, seed=0).encode(["text"])
        b = HashingEmbedder(dimensions=32, seed=1).encode(["text"])
        assert not np.array_equal(a, b)

    def test_shared_words_give_a_positive_score(self):
        vectors = HashingEmbedder(dimensions=512).encode(
            ["refund policy details", "refund policy timing", "unrelated coffee machine"]
        )
        assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])

    def test_empty_text_does_not_divide_by_zero(self):
        vectors = HashingEmbedder(dimensions=8).encode(["", "real text"])
        assert np.all(vectors[0] == 0.0)
        assert np.isfinite(vectors).all()

    def test_name_captures_the_settings(self):
        assert HashingEmbedder(dimensions=64, seed=2).name == "hashing-64-2"

    def test_dimensions_must_be_positive(self):
        with pytest.raises(ValueError, match="dimensions must be positive"):
            HashingEmbedder(dimensions=0)


class TestGetEmbedder:
    def test_hashing_by_name(self):
        assert isinstance(get_embedder("hashing"), HashingEmbedder)

    def test_hashing_with_settings(self):
        embedder = get_embedder("hashing-128-3")
        assert (embedder.dimensions, embedder.seed) == (128, 3)

    def test_anything_else_is_a_sentence_transformer(self):
        assert isinstance(get_embedder("all-MiniLM-L6-v2"), SentenceTransformerEmbedder)

    def test_missing_extra_gives_an_actionable_error(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)
        with pytest.raises(EmbeddingError, match=r"pip install 'ragsweep\[dense\]'"):
            get_embedder("all-MiniLM-L6-v2").encode(["text"])


class TestCache:
    def test_first_pass_encodes_everything(self, cache):
        embedder = CountingEmbedder()
        embed_texts(["a", "b", "c"], embedder, cache)
        assert embedder.encoded == ["a", "b", "c"]
        assert (cache.hits, cache.misses) == (0, 3)

    def test_second_pass_encodes_nothing(self, cache):
        embedder = CountingEmbedder()
        embed_texts(["a", "b"], embedder, cache)
        embedder.encoded.clear()
        embed_texts(["a", "b"], embedder, cache)
        assert embedder.encoded == []
        assert cache.hits == 2

    def test_only_the_new_texts_are_encoded(self, cache):
        embedder = CountingEmbedder()
        embed_texts(["a", "b"], embedder, cache)
        embedder.encoded.clear()
        embed_texts(["a", "b", "c"], embedder, cache)
        assert embedder.encoded == ["c"]

    def test_results_are_identical_with_and_without_the_cache(self, cache):
        embedder = HashingEmbedder(dimensions=32)
        direct = embed_texts(["a", "b"], embedder, None)
        cached = embed_texts(["a", "b"], embedder, cache)
        assert np.allclose(direct, cached)

    def test_order_is_preserved(self, cache):
        embedder = HashingEmbedder(dimensions=32)
        vectors = embed_texts(["zebra", "apple", "mango"], embedder, cache)
        assert np.allclose(vectors[1], embedder.encode(["apple"])[0])

    def test_duplicates_are_encoded_once(self, cache):
        embedder = CountingEmbedder()
        vectors = embed_texts(["same", "same", "same"], embedder, cache)
        assert embedder.encoded == ["same"]
        assert vectors.shape[0] == 3
        assert np.array_equal(vectors[0], vectors[2])

    def test_a_different_model_does_not_reuse_vectors(self, cache):
        small, large = CountingEmbedder(HashingEmbedder(32)), CountingEmbedder(HashingEmbedder(64))
        embed_texts(["text"], small, cache)
        embed_texts(["text"], large, cache)
        assert large.encoded == ["text"]

    def test_cache_key_depends_on_both_model_and_text(self):
        assert cache_key("a", "text") != cache_key("b", "text")
        assert cache_key("a", "one") != cache_key("a", "two")
        assert cache_key("a", "text") == cache_key("a", "text")

    def test_empty_input(self, cache):
        assert embed_texts([], HashingEmbedder(32), cache).shape == (0, 0)


class TestCorruptCacheIsRebuilt:
    def test_truncated_keys_file(self, cache):
        embedder = HashingEmbedder(dimensions=32)
        embed_texts(["a", "b"], embedder, cache)
        folder = cache.root / embedder.name
        (folder / "keys.json").write_text("not json", encoding="utf-8")
        assert embed_texts(["a", "b"], embedder, cache).shape == (2, 32)

    def test_key_count_disagrees_with_vector_count(self, cache):
        embedder = HashingEmbedder(dimensions=32)
        embed_texts(["a", "b"], embedder, cache)
        folder = cache.root / embedder.name
        (folder / "keys.json").write_text(json.dumps(["only-one"]), encoding="utf-8")
        assert cache.read(embedder.name) == ({}, None)
        assert embed_texts(["a", "b"], embedder, cache).shape == (2, 32)

    def test_missing_vectors_file(self, cache):
        embedder = HashingEmbedder(dimensions=32)
        embed_texts(["a"], embedder, cache)
        (cache.root / embedder.name / "vectors.npy").unlink()
        assert embed_texts(["a"], embedder, cache).shape == (1, 32)

    def test_model_names_with_slashes_are_made_safe_for_the_filesystem(self, cache):
        vectors = HashingEmbedder(dimensions=8).encode(["text"])
        cache.write("sentence-transformers/all-MiniLM-L6-v2", ["key1"], vectors)
        assert (cache.root / "sentence-transformers__all-MiniLM-L6-v2").is_dir()
        index, stored = cache.read("sentence-transformers/all-MiniLM-L6-v2")
        assert index == {"key1": 0}
        assert stored.shape == (1, 8)


class TestTokenise:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenise("Refund Policy: 5 days!") == ["refund", "policy", "5", "days"]

    def test_empty_text(self):
        assert tokenise("") == []
