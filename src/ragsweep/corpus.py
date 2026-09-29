"""Read a folder of documents into something the rest of the tool can work with.

Document ids are the path relative to the corpus root, in posix form. They are stable
across runs and across machines, which matters because the labels file refers to
documents by id and has to keep working when the chunk settings change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}
HTML_SUFFIXES = {".html", ".htm"}
READABLE_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES

SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".cache",
    ".pytest_cache", ".ruff_cache", "build", "dist", ".tox",
}

_BLANK_RUN = re.compile(r"\n{3,}")


class CorpusError(Exception):
    """The corpus could not be read at all."""


@dataclass(frozen=True)
class Document:
    id: str
    path: Path
    text: str

    def __len__(self) -> int:
        return len(self.text)


class _TextExtractor(HTMLParser):
    """Pull readable text out of HTML without pulling in a dependency."""

    _SKIP = {"script", "style", "head", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._depth:
            self._depth -= 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def normalise(text: str) -> str:
    """Strip the byte order mark, unify line endings, collapse long blank runs.

    Deliberately conservative. Chunking records character offsets into this text and
    slices it back out, so anything that rewrites the text must happen here, once,
    before offsets are ever taken.
    """
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    return _BLANK_RUN.sub("\n\n", text).strip()


def _html_to_text(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw)
    parser.close()
    return parser.text()


def read_document(path: Path, root: Path) -> Document | None:
    """Read one file. Returns None when it cannot be read, rather than raising.

    One unreadable file in a corpus of thousands should not end the run.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    text = _html_to_text(raw) if path.suffix.lower() in HTML_SUFFIXES else raw
    text = normalise(text)
    if not text:
        return None
    return Document(id=path.relative_to(root).as_posix(), path=path, text=text)


def load_corpus(root: Path) -> list[Document]:
    """Read every readable document under ``root``, sorted by id for stable ordering."""
    root = Path(root)
    if not root.exists():
        raise CorpusError(f"corpus path does not exist: {root}")
    if not root.is_dir():
        raise CorpusError(f"corpus path is not a directory: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in READABLE_SUFFIXES:
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in path.relative_to(root).parts):
            continue
        document = read_document(path, root)
        if document is not None:
            documents.append(document)

    if not documents:
        readable = ", ".join(sorted(READABLE_SUFFIXES))
        raise CorpusError(f"no readable documents under {root}. Looked for: {readable}")
    return documents
