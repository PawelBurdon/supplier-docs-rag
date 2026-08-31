"""Tests for document loading. No network, no API key."""

from pathlib import Path

import pytest

from src.core.loader import load_documents

DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "documents"


def test_loads_every_supported_file():
    documents = load_documents(DOCUMENTS_DIR)
    assert len(documents) == 10
    assert {document.fmt for document in documents} == {"markdown", "text"}


def test_doc_ids_are_unique_and_stable():
    documents = load_documents(DOCUMENTS_DIR)
    doc_ids = [document.doc_id for document in documents]
    assert len(set(doc_ids)) == len(doc_ids)
    # The golden set references documents by these ids, so a rename is a
    # breaking change and should fail here first.
    assert "delivery_sla" in doc_ids
    assert "internal_faq_procurement" in doc_ids


def test_loading_is_ordered():
    first = [document.doc_id for document in load_documents(DOCUMENTS_DIR)]
    second = [document.doc_id for document in load_documents(DOCUMENTS_DIR)]
    assert first == second == sorted(first)


def test_title_comes_from_the_document_not_the_filename():
    documents = {document.doc_id: document for document in load_documents(DOCUMENTS_DIR)}
    assert documents["delivery_sla"].title == "Delivery Service Level Agreement"
    assert documents["quality_inspection_procedure"].title == "QUALITY INSPECTION PROCEDURE"


def test_crlf_and_bom_are_normalised(tmp_path: Path):
    (tmp_path / "sample.md").write_bytes("﻿# Title\r\n\r\nBody line.\r\n".encode("utf-8"))
    document = load_documents(tmp_path)[0]
    assert "\r" not in document.text
    assert document.text.startswith("# Title")
    assert document.title == "Title"


def test_unsupported_and_empty_files_are_ignored(tmp_path: Path):
    (tmp_path / "keep.txt").write_text("Real content.", encoding="utf-8")
    (tmp_path / "skip.pdf").write_text("binary-ish", encoding="utf-8")
    (tmp_path / "blank.md").write_text("   \n\n", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert [document.doc_id for document in documents] == ["keep"]


def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        load_documents(DOCUMENTS_DIR / "does-not-exist")


def test_directory_without_documents_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        load_documents(tmp_path)
