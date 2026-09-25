"""Regression test for the stale-embedding bug found 2026-07-29 (pointed out
in review): src/retrieval/vector_store.py's skip_existing logic used to
check ONLY whether a key already had a non-null embedding in
disclosure_chunks - never whether the STORED TEXT still matched
chunks.jsonl. A corrected/restated disclosure page would rechunk correctly
upstream (text_splitter.py's content-hash incremental logic), but the DB
would silently keep the OLD chunk_text and OLD embedding forever under the
same key, because the row was never re-selected for embedding.

Fix: every chunk record now carries a chunk_hash (SHA-256 of its own text),
and disclosure_chunks stores chunk_hash + embedding_version per row.
_needs_embedding() decides "safe to skip" vs "must (re-)embed" using all
three: key, chunk_hash, embedding_version.

2026-07-30 update (Task #2, post-v3 methodology review): _needs_embedding()'s
key changed from (ticker, source_file, page, chunk_index) to
(pdf_document_id, page, chunk_index) (migration 010 locked this as the real
natural key - see vector_store.py's UPSERT_SQL v5 comment), and the function
now takes the already-resolved pdf_document_id as an explicit argument
rather than reading it out of the record dict. Test bodies updated to match
- the actual bug/fix being tested (content-hash vs key-only skip detection)
is unchanged.
"""
from src.retrieval.text_splitter import chunk_hash
from src.retrieval.vector_store import EMBEDDING_VERSION, _needs_embedding, _record_chunk_hash

PDF_DOC_ID = 42  # arbitrary fixed pdf_document_id used throughout this file

KEY_TEXT_OLD = "Doanh thu tang 15% so voi cung ky"
KEY_TEXT_NEW = "Doanh thu giam 15% so voi cung ky"  # corrected/restated


def _existing_row(text=KEY_TEXT_OLD, embedding_version=EMBEDDING_VERSION, pdf_document_id=PDF_DOC_ID):
    return {(pdf_document_id, 5, 0): (chunk_hash(text), embedding_version)}


def _record(text=KEY_TEXT_OLD, with_hash=True, page=5, chunk_index=0):
    r = {"ticker": "VNM", "source_file": "annual_report_2024.pdf", "page": page,
         "chunk_index": chunk_index, "chunk_text": text}
    if with_hash:
        r["chunk_hash"] = chunk_hash(text)
    return r


def test_unchanged_content_is_skipped():
    existing = _existing_row()
    assert _needs_embedding(_record(KEY_TEXT_OLD), PDF_DOC_ID, existing, EMBEDDING_VERSION) is False


def test_changed_content_same_key_forces_reembed():
    """This is the exact bug scenario: same (pdf_document_id, page,
    chunk_index), different chunk_text after a correction/restatement."""
    existing = _existing_row(text=KEY_TEXT_OLD)
    assert _needs_embedding(_record(KEY_TEXT_NEW), PDF_DOC_ID, existing, EMBEDDING_VERSION) is True


def test_new_key_always_needs_embedding():
    existing = _existing_row()
    new_page_record = _record(KEY_TEXT_OLD, page=6)
    assert _needs_embedding(new_page_record, PDF_DOC_ID, existing, EMBEDDING_VERSION) is True


def test_different_pdf_document_id_is_a_new_key():
    """Added 2026-07-30 - two DIFFERENT pdf_document_id values (even with
    identical page/chunk_index) must never be treated as the same row now
    that pdf_document_id (not ticker/source_file) is part of the locked key
    - this is exactly the "same ticker, multiple real PDF documents" case
    pdf_documents/pdf_document_id was introduced to disambiguate."""
    existing = _existing_row(pdf_document_id=PDF_DOC_ID)
    other_doc_record = _record(KEY_TEXT_OLD)
    assert _needs_embedding(other_doc_record, PDF_DOC_ID + 1, existing, EMBEDDING_VERSION) is True


def test_embedding_version_change_forces_reembed_even_if_text_unchanged():
    existing = _existing_row(text=KEY_TEXT_OLD, embedding_version="BAAI/bge-m3_normalized_v1")
    assert _needs_embedding(_record(KEY_TEXT_OLD), PDF_DOC_ID, existing, "BAAI/bge-m3_normalized_v2") is True


def test_backward_compat_record_without_chunk_hash_field_still_detects_unchanged():
    """chunks.jsonl / a precomputed file written before chunk_hash existed
    (e.g. the 2026-07-22 Colab embeddings) has no "chunk_hash" key at all -
    _record_chunk_hash() must compute it on the fly and still correctly
    identify unchanged content as skippable."""
    existing = _existing_row(text=KEY_TEXT_OLD)
    record_no_hash_key = _record(KEY_TEXT_OLD, with_hash=False)
    assert "chunk_hash" not in record_no_hash_key
    assert _needs_embedding(record_no_hash_key, PDF_DOC_ID, existing, EMBEDDING_VERSION) is False


def test_backward_compat_record_without_chunk_hash_field_still_detects_changed():
    existing = _existing_row(text=KEY_TEXT_OLD)
    record_no_hash_key = _record(KEY_TEXT_NEW, with_hash=False)
    assert _needs_embedding(record_no_hash_key, PDF_DOC_ID, existing, EMBEDDING_VERSION) is True


def test_record_chunk_hash_prefers_stored_value_over_recompute():
    """If a record's stored chunk_hash somehow disagrees with what
    recomputing from chunk_text would give (shouldn't normally happen,
    but _record_chunk_hash should trust the stored value first since it's
    the authoritative one text_splitter.py computed at chunk-creation time)."""
    r = {"chunk_text": "abc", "chunk_hash": "deadbeef"}
    assert _record_chunk_hash(r) == "deadbeef"


def test_record_chunk_hash_falls_back_to_recompute_when_missing():
    r = {"chunk_text": "abc"}
    assert _record_chunk_hash(r) == chunk_hash("abc")
