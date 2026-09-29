"""Splitting documents into chunks.

Three strategies, all pure functions over spans rather than strings. A span is a
``(start, end)`` pair of character offsets into the document, and the invariant every
strategy holds is:

    document.text[chunk.start:chunk.end] == chunk.text

and the chunks together cover the whole document. Keeping offsets instead of copied
text is what lets a retrieved chunk point back at the exact passage it came from,
which weeks 6 and 9 both need.

A note on overlap. For ``fixed`` it is exact: each chunk repeats the previous
``overlap`` characters. For ``sentence`` and ``recursive`` it is approximate, because
backing up mid-sentence would defeat the point of those strategies. They step back over
whole units, taking as many as fit within ``overlap`` characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ragsweep.corpus import Document

Strategy = Literal["fixed", "sentence", "recursive"]
STRATEGIES: tuple[Strategy, ...] = ("fixed", "sentence", "recursive")

Span = tuple[int, int]

#: Tried in order. Earlier separators keep more meaning together.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

#: A sentence ends at ".!?" followed by whitespace, or at a blank line.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    index: int
    text: str
    start: int
    end: int

    def __len__(self) -> int:
        return len(self.text)


def _validate(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")


# --------------------------------------------------------------------------- fixed


def _fixed_spans(start: int, end: int, size: int, overlap: int) -> list[Span]:
    spans: list[Span] = []
    step = size - overlap
    cursor = start
    while cursor < end:
        stop = min(cursor + size, end)
        spans.append((cursor, stop))
        if stop >= end:
            break
        cursor += step
    return spans


# ----------------------------------------------------------------- shared packing


def _pack(units: list[Span], size: int, overlap: int) -> list[Span]:
    """Greedily merge units into chunks of at most ``size``, stepping back for overlap.

    Units are assumed contiguous and each no longer than ``size``.
    """
    if not units:
        return []

    chunks: list[Span] = []
    index = 0
    while index < len(units):
        start = units[index][0]
        end = units[index][1]
        cursor = index + 1
        while cursor < len(units) and units[cursor][1] - start <= size:
            end = units[cursor][1]
            cursor += 1
        chunks.append((start, end))
        if cursor >= len(units):
            break

        # Step back over whole units, taking as many as fit inside `overlap`, but
        # never so far that the next chunk could not reach past this one. Without
        # that second condition a chunk can come out wholly inside its predecessor,
        # adding no new text while still costing an embedding.
        back = cursor
        carried = 0
        while back > index + 1:
            width = units[back - 1][1] - units[back - 1][0]
            if carried + width > overlap:
                break
            if units[cursor][1] - units[back - 1][0] > size:
                break
            back -= 1
            carried += width
        index = back
    return chunks


# ------------------------------------------------------------------------ sentence


def _sentence_units(text: str, start: int, end: int, size: int) -> list[Span]:
    units: list[Span] = []
    cursor = start
    for match in _SENTENCE_BREAK.finditer(text[start:end]):
        stop = start + match.end()
        if stop > cursor:
            units.append((cursor, stop))
            cursor = stop
    if cursor < end:
        units.append((cursor, end))

    # A single sentence longer than the chunk size has to be cut anyway.
    sized: list[Span] = []
    for unit_start, unit_end in units:
        if unit_end - unit_start > size:
            sized.extend(_fixed_spans(unit_start, unit_end, size, 0))
        else:
            sized.append((unit_start, unit_end))
    return sized


# ----------------------------------------------------------------------- recursive


def _recursive_units(
    text: str, start: int, end: int, size: int, separators: tuple[str, ...]
) -> list[Span]:
    if end - start <= size:
        return [(start, end)]
    if not separators:
        return _fixed_spans(start, end, size, 0)

    separator, rest = separators[0], separators[1:]
    pieces: list[Span] = []
    cursor = start
    for match in re.finditer(re.escape(separator), text[start:end]):
        stop = start + match.end()
        if stop > cursor:
            pieces.append((cursor, stop))
            cursor = stop
    if cursor < end:
        pieces.append((cursor, end))

    if len(pieces) <= 1:
        return _recursive_units(text, start, end, size, rest)

    units: list[Span] = []
    for piece_start, piece_end in pieces:
        if piece_end - piece_start > size:
            units.extend(_recursive_units(text, piece_start, piece_end, size, rest))
        else:
            units.append((piece_start, piece_end))
    return units


# ---------------------------------------------------------------------- public api


def chunk_spans(text: str, *, strategy: Strategy, size: int, overlap: int) -> list[Span]:
    """Split ``text`` and return the spans, without copying any strings."""
    _validate(size, overlap)
    if not text:
        return []

    if strategy == "fixed":
        return _fixed_spans(0, len(text), size, overlap)
    if strategy == "sentence":
        return _pack(_sentence_units(text, 0, len(text), size), size, overlap)
    if strategy == "recursive":
        return _pack(_recursive_units(text, 0, len(text), size, _SEPARATORS), size, overlap)
    raise ValueError(f"unknown strategy: {strategy!r}. Expected one of {STRATEGIES}")


def chunk_document(
    document: Document, *, strategy: Strategy, size: int, overlap: int
) -> list[Chunk]:
    spans = chunk_spans(document.text, strategy=strategy, size=size, overlap=overlap)
    return [
        Chunk(
            id=f"{document.id}#{index}",
            document_id=document.id,
            index=index,
            text=document.text[start:end],
            start=start,
            end=end,
        )
        for index, (start, end) in enumerate(spans)
    ]


def chunk_corpus(
    documents: list[Document], *, strategy: Strategy, size: int, overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy=strategy, size=size, overlap=overlap))
    return chunks
