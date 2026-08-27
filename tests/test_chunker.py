"""Tests for chunking. No network, no API key.

These assert the properties the retrieval layer depends on: stable ids, intact
tables, section-accurate metadata, and no fact silently dropped between chunks.
"""

from pathlib import Path

import pytest

from src.core.chunker import MAX_CHARS, Chunk, chunk_document, chunk_documents
from src.core.loader import Document, load_documents

DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "documents"


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return chunk_documents(load_documents(DOCUMENTS_DIR))


def test_chunk_ids_are_unique(chunks: list[Chunk]):
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(set(ids)) == len(ids)


def test_every_chunk_carries_citable_metadata(chunks: list[Chunk]):
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.doc_title
        assert chunk.index_text.startswith(chunk.doc_title)
        assert chunk.text in chunk.index_text


def test_index_text_prefixes_the_section_heading(chunks: list[Chunk]):
    penalty_chunks = [
        chunk
        for chunk in chunks
        if chunk.doc_id == "delivery_sla" and "capped at 10 percent" in chunk.text
    ]
    assert penalty_chunks, "the penalty cap must survive chunking"
    chunk = penalty_chunks[0]
    assert "Late Delivery Penalties" in chunk.section
    # Without the prefix this chunk contains no term a user would search for.
    assert "Late Delivery Penalties" in chunk.index_text
    assert "Late Delivery Penalties" in chunk.citation_label


def test_lead_time_table_stays_in_one_chunk(chunks: list[Chunk]):
    table_chunks = [
        chunk for chunk in chunks if chunk.doc_id == "delivery_sla" and "FRM-CRB" in chunk.text
    ]
    assert len(table_chunks) == 1
    table = table_chunks[0].text
    for row in ("FRM-ALU", "FRM-CRB", "FRK-ALU", "WHL-*", "DRV-*", "CNS-*"):
        assert row in table
    assert "| Product category | SKU prefix |" in table, "header row must travel with the rows"
    assert "Lead time is measured in calendar days" in table, "intro sentence must travel too"


def test_only_tables_may_exceed_the_size_limit(chunks: list[Chunk]):
    for chunk in chunks:
        if len(chunk.text) > MAX_CHARS:
            assert chunk.text.count("|") > 6 or chunk.text.count("\n  ") > 2


def test_overlap_does_not_cross_a_section_boundary(chunks: list[Chunk]):
    for previous, current in zip(chunks, chunks[1:]):
        if previous.doc_id != current.doc_id or previous.section == current.section:
            continue
        # Compared on a character prefix rather than a sentence: many sections
        # open with a clause number, so splitting on "." would compare "3".
        opening = current.text[:60].strip()
        assert opening and opening not in previous.text


def test_facts_survive_chunking(chunks: list[Chunk]):
    corpus = "\n".join(chunk.text for chunk in chunks)
    for fact in (
        "0.5 percent of the order value",  # SLA penalty rate
        "capped at 10 percent",  # SLA penalty cap
        "1 percent of the order value per day late, capped at 15 percent",  # FAQ contradiction
        "net 45 days",  # framework agreement payment terms
        "restocking fee of 15 percent",  # returns policy
        "Restocking fee is 10 percent",  # FAQ contradiction
        "AQL 1.0",  # inspection thresholds
        "Supplier tier: Tier 3",  # onboarding, needed for multi-hop
    ):
        assert fact in corpus, f"lost during chunking: {fact}"


def test_a_long_section_splits_with_overlap():
    body = " ".join(f"Sentence number {index} states a distinct fact." for index in range(120))
    document = Document(
        doc_id="synthetic",
        title="Synthetic Document",
        source_path="synthetic.md",
        fmt="markdown",
        text=f"# Synthetic Document\n\n## 1. Long Section\n\n{body}\n",
    )
    produced = chunk_document(document)
    assert len(produced) > 1
    assert all(chunk.section == "1. Long Section" for chunk in produced)
    # Consecutive chunks inside one section share their boundary sentences.
    first_tail = produced[0].text[-180:]
    assert any(word in produced[1].text for word in first_tail.split(". ") if word)


def test_chunking_is_deterministic():
    documents = load_documents(DOCUMENTS_DIR)
    first = [(chunk.chunk_id, chunk.text) for chunk in chunk_documents(documents)]
    second = [(chunk.chunk_id, chunk.text) for chunk in chunk_documents(documents)]
    assert first == second
