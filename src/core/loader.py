"""Read source documents from disk and keep the metadata a citation needs.

The loader is deliberately dumb: it does not chunk, clean, or interpret. Its one
job is to turn a directory of files into `Document` objects that carry enough
identity for a later citation to say "Delivery SLA, section 5" rather than
"chunk 47".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md": "markdown", ".txt": "text"}


@dataclass(frozen=True)
class Document:
    """One source file, normalised and identified.

    Attributes:
        doc_id: Stable identifier, the file stem. Used by the golden set to say
            which documents a question should retrieve, so it must not change
            when the file content changes.
        title: Human-readable title, taken from the document itself. This is
            what a citation shows to a reader.
        source_path: Path relative to the documents directory, kept so a reader
            can open the file the answer came from.
        fmt: "markdown" or "text". The chunker uses this to pick a heading
            pattern; nothing else depends on it.
        text: Normalised full text.
    """

    doc_id: str
    title: str
    source_path: str
    fmt: str
    text: str


def load_documents(directory: str | Path) -> list[Document]:
    """Load every supported file in `directory`, sorted by filename.

    Sorting matters: chunk ids are positional, and an index built twice from the
    same directory must produce the same ids, otherwise the evaluation harness
    compares runs that are not comparable.

    Raises:
        FileNotFoundError: if the directory does not exist.
        ValueError: if the directory contains no supported files.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {root}")

    documents: list[Document] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        fmt = SUPPORTED_SUFFIXES[path.suffix.lower()]
        text = _normalise(path.read_text(encoding="utf-8"))
        if not text.strip():
            continue
        documents.append(
            Document(
                doc_id=path.stem,
                title=_extract_title(text, fmt, fallback=path.stem),
                source_path=path.name,
                fmt=fmt,
                text=text,
            )
        )

    if not documents:
        raise ValueError(
            f"No .md or .txt files with content found in {root}. "
            "The index would be empty."
        )
    return documents


def _normalise(text: str) -> str:
    """Normalise line endings, strip a BOM, and trim trailing whitespace.

    This runs before anything else because the embedding cache is keyed on a
    hash of chunk text. A file checked out with CRLF endings on Windows would
    otherwise miss every cache entry written on a machine using LF, and the test
    suite would silently start needing the network.
    """
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


_MARKDOWN_TITLE = re.compile(r"^#\s+(.+)$")


def _extract_title(text: str, fmt: str, fallback: str) -> str:
    """Take the title from the document's own first heading or first line.

    A filename-derived title ("onboarding_velocore_components") is a worse
    citation label than the document's own heading, and these documents all
    start with one.
    """
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if fmt == "markdown":
            match = _MARKDOWN_TITLE.match(stripped)
            return match.group(1).strip() if match else stripped
        return stripped
    return fallback
