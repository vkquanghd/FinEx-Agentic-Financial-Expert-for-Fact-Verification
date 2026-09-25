"""Regression test for the 2026-07-29 incremental-chunking fix in
src/retrieval/text_splitter.py: re-running split_directory() on an unchanged
text_native/ folder must reuse every page's chunks from chunk_manifest.csv
instead of re-splitting them, and only newly-added/modified pages should be
freshly chunked.

Also covers the same-day follow-up fix: change detection is by SHA-256
content hash + CHUNKING_VERSION, not mtime/size -- a file whose mtime
changes but content doesn't (e.g. an iCloud re-sync) must NOT be treated
as changed, and a CHUNK_SIZE/CHUNK_OVERLAP/MIN_CHUNK_SIZE edit must force
a full re-chunk WITHOUT needing --force.
"""
import os
import time

import pytest

import src.retrieval.text_splitter as text_splitter_module
from src.retrieval.text_splitter import split_directory

TICKER_A_TEXT = "Doanh thu quy 1 tang truong manh " * 30
TICKER_B_TEXT = "Loi nhuan sau thue giam nhe " * 30
TICKER_C_TEXT = "Cong ty CCC vua cong bo bao cao " * 30


def _write(in_dir, name, text):
    (in_dir / name).write_text(text, encoding="utf-8")


@pytest.fixture
def text_native_dir(tmp_path):
    in_dir = tmp_path / "text_native"
    in_dir.mkdir()
    _write(in_dir, "AAA__annual_report__report1__p1.txt", TICKER_A_TEXT)
    _write(in_dir, "BBB__financial_statement__fs1__p1.txt", TICKER_B_TEXT)
    return in_dir


def test_first_run_chunks_everything(text_native_dir, tmp_path):
    out_path = tmp_path / "chunks.jsonl"
    summary = split_directory(text_native_dir, out_path, verbose=False)
    assert summary["n_rechunked_files"] == 2
    assert summary["n_reused_files"] == 0
    assert summary["n_chunks"] > 0
    assert (tmp_path / "chunk_manifest.csv").exists()


def test_rerun_with_no_changes_reuses_all_pages(text_native_dir, tmp_path):
    out_path = tmp_path / "chunks.jsonl"
    split_directory(text_native_dir, out_path, verbose=False)
    before = out_path.read_text(encoding="utf-8")

    summary = split_directory(text_native_dir, out_path, verbose=False)

    assert summary["n_reused_files"] == 2
    assert summary["n_rechunked_files"] == 0
    # Output content must be byte-identical -- reused chunks are carried
    # forward verbatim, not recomputed.
    assert out_path.read_text(encoding="utf-8") == before


def test_only_changed_or_new_files_get_rechunked(text_native_dir, tmp_path):
    out_path = tmp_path / "chunks.jsonl"
    split_directory(text_native_dir, out_path, verbose=False)

    time.sleep(1.1)  # ensure mtime resolution differs from the first write
    _write(text_native_dir, "AAA__annual_report__report1__p1.txt", "NOI DUNG MOI hoan toan khac " * 30)
    _write(text_native_dir, "CCC__annual_report__report2__p1.txt", TICKER_C_TEXT)

    summary = split_directory(text_native_dir, out_path, verbose=False)

    assert summary["n_reused_files"] == 1  # BBB unchanged
    assert summary["n_rechunked_files"] == 2  # AAA modified + CCC new


def test_mtime_touch_alone_does_not_force_rechunk(text_native_dir, tmp_path):
    """A file re-synced (e.g. iCloud) with identical bytes but a new mtime
    must still be reused -- this is exactly the case mtime/size-based
    detection would get wrong and content-hash detection gets right."""
    out_path = tmp_path / "chunks.jsonl"
    split_directory(text_native_dir, out_path, verbose=False)

    os.utime(text_native_dir / "AAA__annual_report__report1__p1.txt", None)
    summary = split_directory(text_native_dir, out_path, verbose=False)

    assert summary["n_reused_files"] == 2
    assert summary["n_rechunked_files"] == 0


def test_chunking_version_bump_forces_full_rechunk_without_force_flag(text_native_dir, tmp_path):
    """Simulates editing CHUNK_SIZE/CHUNK_OVERLAP/MIN_CHUNK_SIZE: the derived
    CHUNKING_VERSION string changes, which must invalidate every cached
    manifest entry on its own, without the caller needing to pass --force."""
    out_path = tmp_path / "chunks.jsonl"
    split_directory(text_native_dir, out_path, verbose=False)

    original_version = text_splitter_module.CHUNKING_VERSION
    try:
        text_splitter_module.CHUNKING_VERSION = "sliding_999_120_min80_v1"
        summary = text_splitter_module.split_directory(text_native_dir, out_path, verbose=False)
    finally:
        text_splitter_module.CHUNKING_VERSION = original_version

    assert summary["n_reused_files"] == 0
    assert summary["n_rechunked_files"] == 2


def test_force_ignores_manifest_and_rechunks_everything(text_native_dir, tmp_path):
    out_path = tmp_path / "chunks.jsonl"
    split_directory(text_native_dir, out_path, verbose=False)

    summary = split_directory(text_native_dir, out_path, verbose=False, force=True)

    assert summary["n_reused_files"] == 0
    assert summary["n_rechunked_files"] == 2


def test_bootstrap_run_with_no_manifest_reuses_existing_chunks_jsonl(text_native_dir, tmp_path):
    """Regression test for a bug in the first version of this fix: on the
    very first run after incremental chunking was added, chunk_manifest.csv
    doesn't exist yet, but chunks.jsonl from an OLDER pre-manifest run
    already has valid chunks. That must be reused, not force-rechunked --
    lacking a manifest is a bootstrapping gap, not evidence the file is new.
    """
    out_path = tmp_path / "chunks.jsonl"
    # Simulate pre-existing chunks.jsonl written by code that predates the
    # manifest (i.e. no chunk_manifest.csv on disk at all).
    split_directory(text_native_dir, out_path, verbose=False)
    (tmp_path / "chunk_manifest.csv").unlink()

    summary = split_directory(text_native_dir, out_path, verbose=False)

    assert summary["n_rechunked_files"] == 0
    assert summary["n_reused_files"] == 2


def test_removed_file_drops_out_of_output(text_native_dir, tmp_path):
    out_path = tmp_path / "chunks.jsonl"
    split_directory(text_native_dir, out_path, verbose=False)

    (text_native_dir / "BBB__financial_statement__fs1__p1.txt").unlink()
    summary = split_directory(text_native_dir, out_path, verbose=False)

    assert summary["n_files"] == 1
    assert "BBB" not in out_path.read_text(encoding="utf-8")
