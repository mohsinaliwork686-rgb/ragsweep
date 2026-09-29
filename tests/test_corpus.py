"""Reading a folder of documents."""

from __future__ import annotations

import pytest

from ragsweep.corpus import CorpusError, load_corpus, normalise, read_document


@pytest.fixture
def corpus(tmp_path):
    def _write(name: str, content: str):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    return tmp_path, _write


class TestLoadCorpus:
    def test_reads_markdown_and_text(self, corpus):
        root, write = corpus
        write("a.md", "first")
        write("b.txt", "second")
        assert [d.id for d in load_corpus(root)] == ["a.md", "b.txt"]

    def test_ids_are_relative_posix_paths(self, corpus):
        root, write = corpus
        write("handbook/refunds.md", "content")
        assert load_corpus(root)[0].id == "handbook/refunds.md"

    def test_ordering_is_stable(self, corpus):
        root, write = corpus
        for name in ["z.md", "a.md", "m/n.md"]:
            write(name, "content")
        assert [d.id for d in load_corpus(root)] == ["a.md", "m/n.md", "z.md"]

    def test_ignores_unknown_file_types(self, corpus):
        root, write = corpus
        write("keep.md", "content")
        write("skip.png", "content")
        write("skip.pdf", "content")
        assert [d.id for d in load_corpus(root)] == ["keep.md"]

    def test_skips_vendor_and_hidden_directories(self, corpus):
        root, write = corpus
        write("keep.md", "content")
        write("node_modules/pkg/readme.md", "content")
        write(".cache/notes.md", "content")
        assert [d.id for d in load_corpus(root)] == ["keep.md"]

    def test_empty_documents_are_dropped(self, corpus):
        root, write = corpus
        write("empty.md", "   \n\n  ")
        write("real.md", "content")
        assert [d.id for d in load_corpus(root)] == ["real.md"]

    def test_unreadable_file_is_skipped_not_fatal(self, corpus):
        root, write = corpus
        write("good.md", "content")
        (root / "bad.md").write_bytes(b"\xff\xfe\x00 invalid utf-8 \xff")
        assert [d.id for d in load_corpus(root)] == ["good.md"]

    def test_missing_path_gives_a_clear_error(self, tmp_path):
        with pytest.raises(CorpusError, match="does not exist"):
            load_corpus(tmp_path / "nope")

    def test_file_instead_of_directory(self, corpus):
        root, write = corpus
        path = write("a.md", "content")
        with pytest.raises(CorpusError, match="not a directory"):
            load_corpus(path)

    def test_empty_folder_says_what_it_looked_for(self, tmp_path):
        with pytest.raises(CorpusError, match="no readable documents"):
            load_corpus(tmp_path)


class TestHtml:
    def test_tags_are_stripped(self, corpus):
        root, write = corpus
        path = write("a.html", "<body><p>Hello</p><p>World</p></body>")
        assert read_document(path, root).text == "Hello\nWorld"

    def test_script_and_style_content_is_dropped(self, corpus):
        root, write = corpus
        path = write(
            "a.html",
            "<head><style>p{color:red}</style></head><body><script>alert(1)</script><p>Real</p></body>",
        )
        text = read_document(path, root).text
        assert text == "Real"

    def test_entities_are_decoded(self, corpus):
        root, write = corpus
        path = write("a.html", "<p>caf&eacute; &amp; bar</p>")
        assert read_document(path, root).text == "café & bar"


class TestNormalise:
    def test_strips_byte_order_mark(self):
        assert normalise("﻿hello") == "hello"

    def test_unifies_line_endings(self):
        assert normalise("a\r\nb\rc") == "a\nb\nc"

    def test_collapses_long_blank_runs(self):
        assert normalise("a\n\n\n\n\nb") == "a\n\nb"

    def test_keeps_a_single_blank_line(self):
        assert normalise("a\n\nb") == "a\n\nb"

    def test_preserves_unicode(self):
        assert normalise("café 🎉 naïve") == "café 🎉 naïve"
