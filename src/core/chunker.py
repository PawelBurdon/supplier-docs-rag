"""Split documents into retrievable chunks that still know where they came from.

Design decisions, all of which cost something:

Size, ~1200 characters (~200 words). These documents are written in short
numbered sections of 60-120 words, so this usually lands one clause cluster per
chunk. Smaller separates a rule from its own qualifier -- "the penalty is 0.5
percent per day" ends up in a different chunk from "capped at 10 percent", and
an answer can then cite half a rule. Larger blurs the embedding: a vector
averaged over five topics is close to nothing in particular, and a citation that
says "section 5" is harder for a reader to check than one that says "clause
5.2".

Overlap, 15 percent (~180 characters), sentence-aligned. It rescues facts that
straddle a boundary. It also creates near-duplicate chunks that can occupy two
of five top-k slots, which inflates recall@k relative to a system with no
overlap. That is measured, not hidden -- see the README.

Overlap never crosses a section boundary. A chunk labelled "section 6" whose
first sentences belong to section 5 produces a citation that points at the wrong
place, and a wrong citation is worse than a missing one.

Tables are atomic. The lead-time table in the Delivery SLA is the single most
queried object in the corpus and half a table is worse than no table: rows
without a header row are numbers without meaning. The cost is a large chunk
whose embedding mixes every SKU category with every supplier tier, so a question
about one row may not pull it by vector similarity at all. Keyword search on
"FRM-CRB" is what recovers it, which is one of the concrete reasons this project
retrieves hybrid rather than dense-only.

Each chunk is indexed as "Document title > Section heading" followed by its
body. A chunk reading "5.2 The penalty is capped at 10 percent of the order
value" contains neither "late delivery" nor "SLA", and no realistic question
reaches it unprefixed. The cost is that every chunk from one document now shares
a prefix, so chunks from the same document look more alike to the embedder and
top-k diversifies less.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.loader import Document

TARGET_CHARS = 1200
MAX_CHARS = 1600
OVERLAP_CHARS = 180
MIN_CHARS = 200


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text plus the metadata a citation needs.

    Attributes:
        chunk_id: "<doc_id>::<ordinal>", stable for a given documents directory.
        doc_id: Document this came from, matched against the golden set.
        doc_title: Human-readable document title, shown in citations.
        source_path: File the chunk came from, so a citation names something a
            reader can open.
        section: Section heading, or "" for text before the first heading.
        text: The chunk body as it appears in the document. This is what a
            reader is shown with --verbose.
        index_text: What is embedded and what BM25 tokenises: the title and
            section prefix followed by the body.
        ordinal: Position within the document, from 0.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    source_path: str
    section: str
    text: str
    index_text: str
    ordinal: int

    @property
    def citation_label(self) -> str:
        """How this chunk names itself in an answer, e.g. "Delivery SLA, 5. Late Delivery Penalties"."""
        return f"{self.doc_title}, {self.section}" if self.section else self.doc_title


@dataclass(frozen=True)
class _Block:
    """A paragraph, list, or table. Atomic blocks are never split internally."""

    text: str
    atomic: bool


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Chunk every document, preserving input order so chunk ids stay stable."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks


def chunk_document(document: Document) -> list[Chunk]:
    """Split one document into chunks, section by section."""
    chunks: list[Chunk] = []
    ordinal = 0
    for heading, body in _split_sections(document):
        blocks = _split_blocks(body, document.fmt)
        for text in _pack(blocks):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{ordinal:03d}",
                    doc_id=document.doc_id,
                    doc_title=document.title,
                    source_path=document.source_path,
                    section=heading,
                    text=text,
                    index_text=_build_index_text(document.title, heading, text),
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    return chunks


def _build_index_text(title: str, heading: str, text: str) -> str:
    prefix = f"{title} > {heading}" if heading else title
    return f"{prefix}\n\n{text}"


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
# Numbered section headings in the plain-text documents: "3. SAMPLING PLAN".
# Three constraints, each earned by a false positive found while building the
# index: the number must carry a dot ("30 September 2026." is a wrapped date,
# not section 30); the line must be short, so a paragraph opening with a clause
# number stays a paragraph; and the line must be unindented, so a table row
# beginning with a year ("  2026 Q1  96.8% ...") is never a heading.
_TXT_HEADING = re.compile(r"^(\d+\.|\d+(?:\.\d+)+\.?)\s+([A-Z][^\n]{0,60})$")


def _split_sections(document: Document) -> list[tuple[str, str]]:
    """Split a document into (heading, body) pairs.

    Everything before the first heading -- reference numbers, version, owner --
    becomes a section with an empty heading rather than being discarded. In these
    documents that preamble carries the document reference (HCS-SLA-2024) that
    other documents cite, so it has to be retrievable.
    """
    pattern = _MD_HEADING if document.fmt == "markdown" else _TXT_HEADING
    sections: list[tuple[str, list[str]]] = [("", [])]

    for line in document.text.split("\n"):
        # Matched against the raw line, not a stripped copy: an indented line is
        # body text, never a heading.
        match = pattern.match(line)
        # A markdown level-1 heading is the document title. It is already held
        # in Document.title and prepended to every index_text, so it is dropped
        # here rather than becoming a section heading or a chunk of its own.
        is_title = document.fmt == "markdown" and match and len(match.group(1)) == 1
        if is_title:
            continue
        if match:
            heading = _heading_text(match, document.fmt)
            sections.append((heading, []))
        else:
            sections[-1][1].append(line)

    return [
        (heading, "\n".join(lines).strip("\n"))
        for heading, lines in sections
        if "\n".join(lines).strip()
    ]


def _heading_text(match: re.Match[str], fmt: str) -> str:
    if fmt == "markdown":
        return match.group(2).strip()
    return f"{match.group(1).rstrip('.')}. {match.group(2).strip()}"


def _is_table_line(line: str) -> bool:
    """Markdown table row, or a preformatted line in the plain-text documents.

    The plain-text documents lay out their tables with leading spaces and column
    alignment (the sampling plan, the quarterly performance summaries). Those
    lines are only meaningful together, so they are detected the same way.
    """
    stripped = line.strip()
    return stripped.startswith("|") or (line.startswith("  ") and bool(stripped))


def _split_blocks(body: str, fmt: str) -> list[_Block]:
    """Split a section body into paragraphs, keeping tabular runs atomic.

    A table is merged with the short paragraph immediately above it, so the
    sentence that says what the table measures travels with the numbers. Without
    that, the lead-time table retrieves as bare figures with no statement that
    lead time is counted in calendar days from order acknowledgement.
    """
    del fmt  # Table detection is format-agnostic; kept for call-site clarity.
    blocks: list[_Block] = []
    for raw in re.split(r"\n\s*\n", body):
        paragraph = raw.strip("\n")
        if not paragraph.strip():
            continue
        lines = paragraph.split("\n")
        is_table = sum(_is_table_line(line) for line in lines) >= max(2, len(lines) // 2)
        if is_table and blocks and not blocks[-1].atomic and len(blocks[-1].text) <= 400:
            intro = blocks.pop().text
            blocks.append(_Block(f"{intro}\n\n{paragraph}", atomic=True))
        else:
            blocks.append(_Block(paragraph, atomic=is_table))
    return blocks


_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z(\[])")


def _split_sentences(text: str) -> list[str]:
    """Heuristic sentence split.

    It only breaks on terminal punctuation followed by whitespace and a capital
    or a bracket, which leaves "0.5 percent" and "clause 5.2 of" intact. It will
    happily break after an abbreviation such as "B.V." -- acceptable, because the
    consequence is a slightly early chunk boundary, not a lost fact.
    """
    parts = [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]
    return parts or [text.strip()]


def _pack(blocks: list[_Block]) -> list[str]:
    """Greedily pack blocks into chunks of about TARGET_CHARS, with overlap."""
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        chunks.append("\n\n".join(current).strip())
        current.clear()

    def current_length() -> int:
        return sum(len(part) + 2 for part in current)

    for block in blocks:
        if block.atomic:
            if current_length() + len(block.text) > MAX_CHARS:
                flush()
                _seed_overlap(current, chunks, from_atomic=True)
            current.append(block.text)
            continue

        for piece in _fit(block.text):
            if current and current_length() + len(piece) > TARGET_CHARS:
                previous_was_atomic = current == [] or _looks_tabular(current[-1])
                flush()
                _seed_overlap(current, chunks, from_atomic=previous_was_atomic)
            current.append(piece)

    flush()
    return _merge_runt(chunks)


def _fit(text: str) -> list[str]:
    """Break an oversized paragraph on sentence boundaries; leave the rest alone."""
    if len(text) <= MAX_CHARS:
        return [text]
    pieces: list[str] = []
    buffer = ""
    for sentence in _split_sentences(text):
        if buffer and len(buffer) + len(sentence) > TARGET_CHARS:
            pieces.append(buffer.strip())
            buffer = ""
        buffer = f"{buffer} {sentence}".strip()
    if buffer:
        pieces.append(buffer)
    return pieces


def _seed_overlap(current: list[str], chunks: list[str], from_atomic: bool) -> None:
    """Start the next chunk with the tail of the previous one.

    Skipped after a table: repeating the last few rows of a table without its
    header produces a fragment that means nothing on its own and competes for a
    top-k slot with the intact table.
    """
    if from_atomic or not chunks:
        return
    tail = _tail_sentences(chunks[-1], OVERLAP_CHARS)
    if tail:
        current.append(tail)


def _tail_sentences(text: str, budget: int) -> str:
    """Take whole trailing sentences up to `budget` characters."""
    sentences = _split_sentences(text)
    taken: list[str] = []
    total = 0
    for sentence in reversed(sentences[:-1] or sentences):
        if total + len(sentence) > budget and taken:
            break
        taken.insert(0, sentence)
        total += len(sentence)
    return " ".join(taken)


def _looks_tabular(text: str) -> bool:
    lines = text.split("\n")
    return sum(_is_table_line(line) for line in lines) >= max(2, len(lines) // 2)


def _merge_runt(chunks: list[str]) -> list[str]:
    """Fold a too-small final chunk back into its predecessor where it fits.

    Sections often end with a one-line remark. On its own that line is a chunk
    with almost no lexical content, which is dead weight in the index and a
    useless citation target.
    """
    if len(chunks) < 2:
        return chunks
    if len(chunks[-1]) < MIN_CHARS and len(chunks[-2]) + len(chunks[-1]) <= MAX_CHARS:
        merged = f"{chunks[-2]}\n\n{chunks[-1]}"
        return chunks[:-2] + [merged]
    return chunks
