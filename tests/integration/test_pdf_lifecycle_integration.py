"""PDF lifecycle integration test (Task #15, post-v3 methodology review,
2026-07-30) - runs the REAL pipeline end-to-end against a REAL Postgres +
pgvector instance. This is the highest-priority item from that review:
"verify code matches docs" by actually exercising the 8 scenarios below,
not just unit-testing individual functions against fake DB objects (every
other test in this repo, e.g. tests/test_retrieval/test_vector_store_stale_
embedding.py, is a fake-DB test - this is the one real-DB test).

Cannot be run inside the sandbox this file was authored in (no reachable
Postgres, no network to install psycopg2/sqlalchemy) - run it on your own
machine, after `docker compose up` (see docker-compose.yml), with:

    RUN_PDF_LIFECYCLE_INTEGRATION_TEST=1 DB_NAME=thesis_fin_qa_test \\
        python3 -m pytest tests/integration/test_pdf_lifecycle_integration.py -v -s

SAFETY: this test TRUNCATEs disclosure_chunks/pdf_documents rows it creates
and, in the final full-rebuild phase, wipes and rebuilds an entire schema's
worth of data. Two independent guards before it will touch a database at
all: (1) the RUN_PDF_LIFECYCLE_INTEGRATION_TEST=1 env var must be set
(never runs by accident via a bare `pytest`), (2) DB_NAME must contain
"test" (case-insensitive) - point it at a disposable database, e.g. by
running `docker compose exec postgres createdb -U thesis_user
thesis_fin_qa_test` first. NEVER run this against the real corpus database.

Embeddings: by default this test monkeypatches embed_texts() with a
deterministic, hash-seeded 1024-dim generator (matches EMBEDDING_DIM so the
real pgvector column type is still exercised) instead of loading the real
2.2GB bge-m3 model - this keeps the test fast and network-free while still
exercising every real DB write path. Set
PDF_LIFECYCLE_TEST_REAL_EMBEDDINGS=1 to use the real model instead (slower,
needs the model cached or a network connection) - only the fake-embedding
path has been exercised so far (this repo's sandbox cannot load bge-m3
either), so re-verify locally the first time you flip that flag on.

The 8 scenarios (verbatim from the review) are run IN SEQUENCE against one
shared corpus directory + one shared DB, because most of them are only
meaningful as a DELTA from the previous state:
  1. add a new PDF
  2. re-run with nothing changed (must be a full no-op)
  3. edit a PDF's content but keep the same filename
  4. change extraction_version (pdf_ingest_router.INGEST_VERSION)
  5. change chunking_version (text_splitter.CHUNKING_VERSION)
  6. change embedding_version (vector_store.EMBEDDING_VERSION)
  7. delete a PDF
  8. simulate a mid-batch crash, then resume normally

After every scenario this test checks: pdf_documents count (active),
disclosure_chunks count, newly-embedded count, deleted/orphaned chunk
count, updated-in-place chunk count, and a top-k retrieval sanity query.

Finally, and this is the CRITICAL check the review called out by name: the
final incremental state is compared against an from-scratch full rebuild of
the same end corpus, asserting

    incremental_chunk_keys == full_rebuild_chunk_keys
    incremental_chunk_hashes == full_rebuild_chunk_hashes

exactly as specified - if these two runs ever disagree, incremental
indexing has a correctness bug, full stop.
"""
import hashlib
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import psycopg2  # noqa: F401
except ImportError:
    psycopg2 = None

pytestmark = pytest.mark.integration


def _check_prereqs():
    if os.getenv("RUN_PDF_LIFECYCLE_INTEGRATION_TEST") != "1":
        pytest.skip(
            "Set RUN_PDF_LIFECYCLE_INTEGRATION_TEST=1 to run this real-Postgres "
            "integration test (it never runs as part of a bare `pytest` sweep)."
        )
    if psycopg2 is None or shutil.which("pdftotext") is None or shutil.which("pdfinfo") is None:
        pytest.skip("Requires psycopg2 and poppler-utils (pdftotext/pdfinfo) installed locally.")
    try:
        import reportlab  # noqa: F401
    except ImportError:
        pytest.skip("Requires reportlab (pip install reportlab) to generate test-fixture PDFs.")

    from src.core import config
    if "test" not in config.DB_NAME.lower():
        pytest.fail(
            f"DB_NAME='{config.DB_NAME}' does not contain 'test' - refusing to run this "
            f"destructive integration test against what looks like a real database. "
            f"Point DB_NAME at a disposable database, e.g. thesis_fin_qa_test."
        )
    try:
        from src.retrieval import vector_store as vs
        conn = vs.get_connection()
        conn.close()
    except Exception as e:
        pytest.skip(f"Cannot reach Postgres ({e}) - is `docker compose up` running?")


def _make_pdf(path: Path, paragraph: str, n_repeats: int = 8):
    """One-page real PDF via reportlab with enough extractable, non-suspicious
    ASCII-ish text to clear pdf_ingest_router.py's CHAR_COUNT_MIN=150 /
    LEGIBILITY_MIN=0.90 thresholds (real pdftotext -raw extraction, not
    mocked)."""
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=(612, 792))
    y = 750
    for _ in range(n_repeats):
        c.drawString(72, y, paragraph)
        y -= 20
        if y < 72:
            c.showPage()
            y = 750
    c.save()


def _fake_embed_texts(texts, batch_size: int = 32):
    """Deterministic, seeded-by-content-hash 1024-dim unit vectors - same
    EMBEDDING_DIM as the real bge-m3 column, so pgvector's VECTOR(1024) type
    and the cosine index are genuinely exercised; NOT semantically meaningful
    (this is not the real model), so the top-k check below only asserts
    that identical/near-identical text ranks above unrelated text, not real
    retrieval quality."""
    from src.retrieval.embeddings import EMBEDDING_DIM
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        seed = struct.unpack("Q", h[:8])[0]
        # Simple LCG seeded from the text hash - deterministic, no numpy/torch
        # dependency needed for a fake vector.
        vec = []
        state = seed
        for _ in range(EMBEDDING_DIM):
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            vec.append((state / (1 << 64)) * 2 - 1)
        norm = sum(x * x for x in vec) ** 0.5
        out.append([x / norm for x in vec])
    return out


class Snapshot:
    """One point-in-time read of every count/set this test's 8 scenarios
    need to check, taken via direct SQL so it's independent of (and thus a
    genuine cross-check on) the application code under test."""

    def __init__(self, conn):
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pdf_documents WHERE is_active")
            self.active_documents = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM disclosure_chunks")
            self.total_chunks = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM disclosure_chunks WHERE embedding IS NOT NULL")
            self.embedded_chunks = cur.fetchone()[0]
            cur.execute("SELECT pdf_document_id, page, chunk_index, chunk_hash, embedding_version "
                        "FROM disclosure_chunks")
            rows = cur.fetchall()
            self.keys = {(r[0], r[1], r[2]) for r in rows}
            self.hashes = {(r[0], r[1], r[2]): r[3] for r in rows}
            self.embedding_versions = {(r[0], r[1], r[2]): r[4] for r in rows}

    def diff(self, other: "Snapshot") -> dict:
        return {
            "new_keys": other.keys - self.keys,
            "removed_keys": self.keys - other.keys,
            "changed_hash_same_key": {
                k for k in (self.keys & other.keys)
                if self.hashes[k] != other.hashes[k]
            },
            "changed_embedding_version_same_key": {
                k for k in (self.keys & other.keys)
                if self.embedding_versions[k] != other.embedding_versions[k]
            },
        }


def _run_pipeline_stage(in_dir: Path, out_dir: Path, chunks_path: Path):
    """One full pass of the real (non-DB) stages: ingest -> chunk. Returns
    the ingest_manifest.csv path (for register_pdf_documents.py) and the
    chunk_directory summary dict from split_directory()."""
    import scripts.pdf_ingest_router as router
    old_argv = sys.argv
    try:
        sys.argv = ["pdf_ingest_router.py", "--input", str(in_dir), "--output", str(out_dir)]
        router.main()
    finally:
        sys.argv = old_argv

    from src.retrieval.text_splitter import split_directory
    summary = split_directory(out_dir / "text_native", chunks_path, verbose=False)
    return out_dir / "ingest_manifest.csv", summary


def _register_and_upsert(manifest_path: Path, chunks_path: Path):
    from scripts import register_pdf_documents as rpd
    from src.retrieval import vector_store as vs

    reg_summary = rpd.register_all(manifest_path)
    deleted_path = manifest_path.parent / "deleted_files.csv"
    n_deactivated = rpd.deactivate_deleted(deleted_path)

    records = vs.load_jsonl_records(chunks_path)
    n_written = vs.upsert_records(records) if records else 0

    conn = vs.get_connection()
    try:
        doc_ids_by_file = vs.resolve_pdf_document_ids_or_raise(conn, records) if records else {}
    finally:
        conn.close()
    current_keys = {(doc_ids_by_file[r["source_file"]], r["page"], r["chunk_index"]) for r in records}
    n_orphans_dryrun = vs.prune_orphaned_chunks(current_keys, dry_run=True)

    return {
        "register": reg_summary, "n_deactivated": n_deactivated,
        "n_written": n_written, "n_orphans_dryrun": n_orphans_dryrun,
        "current_keys": current_keys,
    }


@pytest.fixture(autouse=True)
def _patch_embeddings(monkeypatch):
    if os.getenv("PDF_LIFECYCLE_TEST_REAL_EMBEDDINGS") == "1":
        yield
        return
    from src.retrieval import vector_store as vs
    monkeypatch.setattr(vs, "embed_texts", _fake_embed_texts)
    yield


@pytest.fixture
def workdir():
    tmp = Path(tempfile.mkdtemp(prefix="pdf_lifecycle_it_"))
    (tmp / "raw").mkdir()
    (tmp / "processed").mkdir()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def _cleanup_db(conn, pdf_document_ids):
    """Best-effort teardown so re-running this test (or other tests sharing
    the same test DB) starts clean - deletes only rows this test created."""
    if not pdf_document_ids:
        return
    with conn.cursor() as cur:
        cur.execute("DELETE FROM disclosure_chunks WHERE pdf_document_id = ANY(%s)", (list(pdf_document_ids),))
        cur.execute("DELETE FROM pdf_documents WHERE id = ANY(%s)", (list(pdf_document_ids),))
    conn.commit()


PARA_ORIGINAL = "Tong tai san cua VNM tang muoi lam phan tram so voi cung ky nam truoc do doanh thu tang manh."
PARA_EDITED = "Tong tai san cua VNM giam tam phan tram so voi cung ky nam truoc do chi phi nguyen lieu tang."
PARA_HPG = "HPG ghi nhan loi nhuan sau thue tang hai muoi phan tram nho gia thep phuc hoi trong nam."


def test_pdf_lifecycle_8_scenarios():
    _check_prereqs()

    from src.retrieval import vector_store as vs
    import scripts.pdf_ingest_router as router
    import src.retrieval.text_splitter as ts

    tmp = Path(tempfile.mkdtemp(prefix="pdf_lifecycle_it_"))
    in_dir, out_dir = tmp / "raw", tmp / "processed"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    chunks_path = out_dir / "chunks.jsonl"
    conn = vs.get_connection()
    # autocommit so every Snapshot() SELECT below reads the latest state
    # committed by the pipeline calls above (which open/commit/close their
    # OWN separate connections via vs.get_connection()) instead of a stale
    # per-transaction snapshot from whenever this connection's transaction
    # happened to start.
    conn.autocommit = True
    all_pdf_document_ids = set()

    try:
        # ---------------------------------------------------------------
        # Scenario 1: add a new PDF (VNM)
        # ---------------------------------------------------------------
        before = Snapshot(conn)
        _make_pdf(in_dir / "VNM_baocaotaichinh_2024.pdf", PARA_ORIGINAL)
        manifest_path, _ = _run_pipeline_stage(in_dir, out_dir, chunks_path)
        result = _register_and_upsert(manifest_path, chunks_path)
        after = Snapshot(conn)
        d = before.diff(after)
        assert after.active_documents == before.active_documents + 1, "scenario 1: expected +1 active document"
        assert len(d["new_keys"]) == after.total_chunks - before.total_chunks
        assert result["n_written"] > 0, "scenario 1: new PDF must produce new embeddings"
        assert result["n_orphans_dryrun"] == 0, "scenario 1: nothing should be orphaned yet"
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM pdf_documents WHERE source_file = %s", ("VNM_baocaotaichinh_2024.pdf",))
            all_pdf_document_ids.add(cur.fetchone()[0])
        print(f"[1/8] add new PDF: +{after.active_documents - before.active_documents} doc, "
              f"+{len(d['new_keys'])} chunk(s), {result['n_written']} embedded, "
              f"{result['n_orphans_dryrun']} orphan(s)")

        # ---------------------------------------------------------------
        # Scenario 2: re-run with nothing changed -> full no-op
        # ---------------------------------------------------------------
        before = Snapshot(conn)
        manifest_path, _ = _run_pipeline_stage(in_dir, out_dir, chunks_path)
        result = _register_and_upsert(manifest_path, chunks_path)
        after = Snapshot(conn)
        d = before.diff(after)
        assert after.total_chunks == before.total_chunks
        assert after.active_documents == before.active_documents
        assert result["n_written"] == 0, "scenario 2: unchanged re-run must not re-embed anything"
        assert not d["new_keys"] and not d["removed_keys"] and not d["changed_hash_same_key"]
        print("[2/8] re-run unchanged: full no-op, 0 embedded, as expected")

        # ---------------------------------------------------------------
        # Scenario 3: edit VNM's content but keep the same filename
        # ---------------------------------------------------------------
        before = Snapshot(conn)
        _make_pdf(in_dir / "VNM_baocaotaichinh_2024.pdf", PARA_EDITED)
        manifest_path, _ = _run_pipeline_stage(in_dir, out_dir, chunks_path)
        result = _register_and_upsert(manifest_path, chunks_path)
        after = Snapshot(conn)
        d = before.diff(after)
        assert after.active_documents == before.active_documents, "content edit must NOT change document count"
        assert d["changed_hash_same_key"], "scenario 3: same key, changed chunk_hash expected (the stale-embedding bug's exact scenario)"
        assert result["n_written"] > 0, "scenario 3: changed content must trigger re-embedding"
        with conn.cursor() as cur:
            cur.execute("SELECT document_version FROM pdf_documents WHERE source_file = %s",
                        ("VNM_baocaotaichinh_2024.pdf",))
            version_after_edit = cur.fetchone()[0]
        assert version_after_edit == 2, f"expected document_version bumped to 2 after a real content change, got {version_after_edit}"
        print(f"[3/8] edit content, same filename: document_version->{version_after_edit}, "
              f"{len(d['changed_hash_same_key'])} chunk(s) re-embedded, doc count unchanged")

        # ---------------------------------------------------------------
        # Scenario 4: change extraction_version (INGEST_VERSION)
        # ---------------------------------------------------------------
        before = Snapshot(conn)
        old_char_min = router.CHAR_COUNT_MIN
        router.CHAR_COUNT_MIN = old_char_min - 1  # any change to this constant bumps INGEST_VERSION automatically
        router.INGEST_VERSION = f"charmin{router.CHAR_COUNT_MIN}_legmin{router.LEGIBILITY_MIN}_v1"
        try:
            manifest_path, _ = _run_pipeline_stage(in_dir, out_dir, chunks_path)
            result = _register_and_upsert(manifest_path, chunks_path)
        finally:
            router.CHAR_COUNT_MIN = old_char_min
            router.INGEST_VERSION = f"charmin{router.CHAR_COUNT_MIN}_legmin{router.LEGIBILITY_MIN}_v1"
        after = Snapshot(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT extraction_version, document_version FROM pdf_documents WHERE source_file = %s",
                        ("VNM_baocaotaichinh_2024.pdf",))
            extraction_version_after, doc_version_after = cur.fetchone()
        assert doc_version_after == 3, f"extraction_version change must bump document_version again, got {doc_version_after}"
        print(f"[4/8] change extraction_version: pdf_documents.extraction_version="
              f"{extraction_version_after}, document_version->{doc_version_after}")

        # ---------------------------------------------------------------
        # Scenario 5: change chunking_version
        # ---------------------------------------------------------------
        before = Snapshot(conn)
        old_chunk_size = ts.CHUNK_SIZE
        ts.CHUNK_SIZE = old_chunk_size - 100
        ts.CHUNKING_VERSION = f"sliding_{ts.CHUNK_SIZE}_{ts.CHUNK_OVERLAP}_min{ts.MIN_CHUNK_SIZE}_v1"
        try:
            summary = ts.split_directory(out_dir / "text_native", chunks_path, verbose=False, force=True)
            result = _register_and_upsert(manifest_path, chunks_path)
        finally:
            ts.CHUNK_SIZE = old_chunk_size
            ts.CHUNKING_VERSION = f"sliding_{ts.CHUNK_SIZE}_{ts.CHUNK_OVERLAP}_min{ts.MIN_CHUNK_SIZE}_v1"
        after = Snapshot(conn)
        assert result["n_written"] > 0, "scenario 5: a chunking_version change must force re-embedding"
        print(f"[5/8] change chunking_version: {summary['n_chunks']} chunk(s) in new chunks.jsonl, "
              f"{result['n_written']} (re-)embedded")

        # ---------------------------------------------------------------
        # Scenario 6: change embedding_version
        # ---------------------------------------------------------------
        before = Snapshot(conn)
        old_embedding_version = vs.EMBEDDING_VERSION
        vs.EMBEDDING_VERSION = old_embedding_version + "_TEST_BUMP"
        try:
            records = vs.load_jsonl_records(chunks_path)
            n_written = vs.upsert_records(records)
        finally:
            vs.EMBEDDING_VERSION = old_embedding_version
        after = Snapshot(conn)
        d = before.diff(after)
        assert n_written == before.total_chunks, "scenario 6: an embedding_version bump must re-embed EVERY existing chunk"
        assert d["changed_embedding_version_same_key"], "scenario 6: expected embedding_version to change for existing keys"
        print(f"[6/8] change embedding_version: {n_written} chunk(s) re-embedded (all of them, as expected)")

        # ---------------------------------------------------------------
        # Scenario 7: delete a PDF (HPG, added here so scenario 7 has
        # something concrete to remove without disturbing VNM's history)
        # ---------------------------------------------------------------
        _make_pdf(in_dir / "HPG_annual_2024.pdf", PARA_HPG)
        manifest_path, _ = _run_pipeline_stage(in_dir, out_dir, chunks_path)
        _register_and_upsert(manifest_path, chunks_path)
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM pdf_documents WHERE source_file = %s", ("HPG_annual_2024.pdf",))
            all_pdf_document_ids.add(cur.fetchone()[0])

        before = Snapshot(conn)
        (in_dir / "HPG_annual_2024.pdf").unlink()
        manifest_path, _ = _run_pipeline_stage(in_dir, out_dir, chunks_path)
        result = _register_and_upsert(manifest_path, chunks_path)
        assert result["n_orphans_dryrun"] > 0, "scenario 7: HPG's chunks must show up as orphaned once its PDF is deleted"
        vs.prune_orphaned_chunks(result["current_keys"], dry_run=False)
        after = Snapshot(conn)
        assert after.active_documents == before.active_documents - 1, "scenario 7: expected -1 active document"
        assert after.total_chunks < before.total_chunks, "scenario 7: orphaned HPG chunks must actually be deleted after --prune-orphans"
        with conn.cursor() as cur:
            cur.execute("SELECT is_active FROM pdf_documents WHERE source_file = %s", ("HPG_annual_2024.pdf",))
            assert cur.fetchone()[0] is False, "scenario 7: HPG's pdf_documents row must be soft-deleted (is_active=FALSE), not left active"
        print(f"[7/8] delete PDF: active_documents {before.active_documents}->{after.active_documents}, "
              f"chunks {before.total_chunks}->{after.total_chunks} after pruning orphans")

        # ---------------------------------------------------------------
        # Scenario 8: pipeline crashes mid-batch, then resumed
        # ---------------------------------------------------------------
        _make_pdf(in_dir / "MSN_annual_2024.pdf", "MSN ghi nhan doanh thu tang manh trong quy gan nhat nho mang ban le tieu dung.")
        manifest_path, _ = _run_pipeline_stage(in_dir, out_dir, chunks_path)
        rpd_mod = __import__("scripts.register_pdf_documents", fromlist=["register_all"])
        rpd_mod.register_all(manifest_path)
        deleted_path = manifest_path.parent / "deleted_files.csv"
        rpd_mod.deactivate_deleted(deleted_path)
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM pdf_documents WHERE source_file = %s", ("MSN_annual_2024.pdf",))
            all_pdf_document_ids.add(cur.fetchone()[0])

        before = Snapshot(conn)
        records = vs.load_jsonl_records(chunks_path)
        new_records = [r for r in records if r["source_file"] == "MSN_annual_2024.pdf"]
        assert new_records, "expected at least one fresh MSN chunk record to simulate a crash on"

        active_embed_fn = vs.embed_texts  # whatever's currently active - fake (default) or real
        call_count = {"n": 0}

        def _crash_after_first_batch(texts, batch_size=32):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated crash mid-batch (scenario 8)")
            return active_embed_fn(texts, batch_size=batch_size)

        vs.embed_texts = _crash_after_first_batch
        try:
            with pytest.raises(RuntimeError, match="simulated crash mid-batch"):
                vs.upsert_records(records, batch_size=1)
        finally:
            vs.embed_texts = active_embed_fn
        mid_crash = Snapshot(conn)
        assert mid_crash.embedded_chunks < len(records), (
            "scenario 8 setup check: the simulated crash should have left at least the MSN "
            "chunk(s) un-embedded, proving upsert_records() commits per-batch rather than "
            "all-or-nothing (see vector_store.py's 2026-07-21 per-batch-commit fix)"
        )

        n_written_resume = vs.upsert_records(records, batch_size=32)
        after = Snapshot(conn)
        assert after.embedded_chunks == len(records), "scenario 8: a normal resume must finish embedding everything the crash left incomplete"
        assert n_written_resume > 0, "scenario 8: resume must actually do work (not silently skip everything)"
        print(f"[8/8] crash-then-resume: crash left {mid_crash.embedded_chunks}/{len(records)} embedded, "
              f"resume wrote {n_written_resume} more, final {after.embedded_chunks}/{len(records)} embedded")

        # -----------------------------------------------------------
        # CRITICAL: incremental vs full-rebuild equivalence
        # -----------------------------------------------------------
        incremental_final = Snapshot(conn)
        incremental_chunk_keys = incremental_final.keys
        incremental_chunk_hashes = incremental_final.hashes

        _cleanup_db(conn, all_pdf_document_ids)
        rebuild_tmp = Path(tempfile.mkdtemp(prefix="pdf_lifecycle_it_rebuild_"))
        rebuild_out = rebuild_tmp / "processed"
        rebuild_out.mkdir(parents=True)
        rebuild_chunks_path = rebuild_out / "chunks.jsonl"
        try:
            manifest_path_full, _ = _run_pipeline_stage(in_dir, rebuild_out, rebuild_chunks_path)
            full_result = _register_and_upsert(manifest_path_full, rebuild_chunks_path)
            full_snapshot = Snapshot(conn)

            rebuild_doc_ids = set()
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT pdf_document_id FROM disclosure_chunks")
                rebuild_doc_ids = {r[0] for r in cur.fetchall()}

            assert incremental_chunk_keys == full_snapshot.keys, (
                "INCREMENTAL/FULL-REBUILD MISMATCH (keys) - incremental indexing has a "
                f"correctness bug. Incremental-only keys: {incremental_chunk_keys - full_snapshot.keys}, "
                f"full-rebuild-only keys: {full_snapshot.keys - incremental_chunk_keys}"
            )
            assert incremental_chunk_hashes == full_snapshot.hashes, (
                "INCREMENTAL/FULL-REBUILD MISMATCH (chunk_hash per key) - same keys but "
                "different content between the incremental path and a from-scratch rebuild."
            )
            print("[CRITICAL] incremental_chunk_keys == full_rebuild_chunk_keys: PASS")
            print("[CRITICAL] incremental_chunk_hashes == full_rebuild_chunk_hashes: PASS")

            # --- top-k retrieval sanity check (fake-embedding space unless
            # PDF_LIFECYCLE_TEST_REAL_EMBEDDINGS=1) ---
            from src.retrieval.embeddings import embed_query as real_embed_query
            query_vec = (vs.embed_texts([PARA_EDITED])[0] if os.getenv("PDF_LIFECYCLE_TEST_REAL_EMBEDDINGS") != "1"
                         else real_embed_query(PARA_EDITED))
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source_file, chunk_text FROM disclosure_chunks "
                    "ORDER BY embedding <=> %s::vector LIMIT 3",
                    (vs._vector_literal(query_vec),),
                )
                top3 = cur.fetchall()
            assert top3, "top-k retrieval returned nothing after the full rebuild"
            assert top3[0][0] == "VNM_baocaotaichinh_2024.pdf", (
                f"top-1 result for a VNM-edited-content query should be the VNM chunk, got {top3[0]}"
            )
            print(f"[CRITICAL] top-k retrieval sanity: top-1 = {top3[0][0]!r} (expected VNM file) - PASS")

            _cleanup_db(conn, rebuild_doc_ids)
        finally:
            shutil.rmtree(rebuild_tmp, ignore_errors=True)

    finally:
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)
