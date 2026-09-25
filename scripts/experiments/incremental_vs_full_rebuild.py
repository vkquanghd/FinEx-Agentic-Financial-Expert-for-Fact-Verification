"""Ablation experiment (Task #17 / roadmap item 9, 2026-07-30): measures
incremental indexing vs a from-scratch full rebuild on the REAL PDF corpus -
wall time per stage, embedding count, and retrieval equivalence - and writes
a JSON report for direct use in the thesis's ablation section.

This is DIFFERENT from tests/integration/test_pdf_lifecycle_integration.py:
that test asserts correctness (pass/fail) on a handful of tiny synthetic
PDFs; this script measures performance/cost at real corpus scale and
produces numbers to report, not a pass/fail test. Both share the same
underlying equivalence check (incremental_chunk_keys == full_rebuild_
chunk_keys, same for hashes) because that correctness property is exactly
what makes the timing comparison meaningful - a faster incremental run that
produced WRONG output would not be a real speedup.

Cannot be run inside the sandbox this file was authored in (no reachable
Postgres, no network for the real bge-m3 model). Run it on your own machine
against a DISPOSABLE test database (never your real corpus DB - Phase A
below wipes disclosure_chunks/pdf_documents entirely):

    docker compose exec postgres createdb -U thesis_user thesis_fin_qa_test
    DB_NAME=thesis_fin_qa_test RUN_INCREMENTAL_ABLATION=1 \\
        python3 -m scripts.experiments.incremental_vs_full_rebuild \\
            --raw-dir data/raw/pdf_disclosures --holdout 5 \\
            --report reports/incremental_vs_full_rebuild.json

--holdout N: N real PDFs are excluded from the initial ("base") corpus and
added back in Phase B to simulate the realistic "add N new PDFs to an
already-large corpus" incremental scenario (rather than a no-op re-run,
which would trivially show ~0 cost and isn't an interesting ablation
point). Use --holdout 0 to instead measure the pure "re-run with nothing
changed" no-op cost.

Embeddings: real bge-m3 by default (this script is about measuring REAL
performance, so a fake embedding function would defeat its own purpose,
unlike the correctness-focused integration test) - requires the model
downloaded/cached and a GPU/CPU willing to embed the full corpus once for
Phase A. Pass --fake-embeddings to use the deterministic stub instead (only
useful for a quick dry-run of this script's plumbing, NOT for real
ablation numbers - the JSON report is stamped with which mode was used so
a fake-embeddings run can never be mistaken for real numbers).
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require_test_db():
    import os
    from src.core import config
    if os.getenv("RUN_INCREMENTAL_ABLATION") != "1":
        raise SystemExit(
            "Set RUN_INCREMENTAL_ABLATION=1 to run this script - it WIPES "
            "disclosure_chunks/pdf_documents in DB_NAME (Phase A needs an "
            "empty starting state), and never runs by accident."
        )
    if "test" not in config.DB_NAME.lower():
        raise SystemExit(
            f"DB_NAME='{config.DB_NAME}' does not contain 'test' - refusing to run this "
            f"experiment against what looks like the real corpus database. Create a "
            f"disposable one (e.g. thesis_fin_qa_test) and point DB_NAME at it."
        )


def _wipe_db(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM disclosure_chunks")
        cur.execute("DELETE FROM pdf_documents")
    conn.commit()


def _run_full_pipeline(pdf_dir: Path, work_dir: Path, stage_times: dict) -> dict:
    """One complete pass: ingest -> chunk -> register -> embed+upsert.
    Returns a dict of counts; timings are appended into stage_times in
    place (keys prefixed by the caller so Phase A/B don't collide)."""
    import scripts.pdf_ingest_router as router
    from src.retrieval.text_splitter import split_directory
    from scripts import register_pdf_documents as rpd
    from src.retrieval import vector_store as vs

    out_dir = work_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"

    old_argv = sys.argv
    t0 = time.monotonic()
    try:
        sys.argv = ["pdf_ingest_router.py", "--input", str(pdf_dir), "--output", str(out_dir)]
        router.main()
    finally:
        sys.argv = old_argv
    stage_times["ingest_s"] = time.monotonic() - t0

    t0 = time.monotonic()
    chunk_summary = split_directory(out_dir / "text_native", chunks_path, verbose=False)
    stage_times["chunk_s"] = time.monotonic() - t0

    t0 = time.monotonic()
    reg_summary = rpd.register_all(out_dir / "ingest_manifest.csv")
    deleted_path = out_dir / "deleted_files.csv"
    rpd.deactivate_deleted(deleted_path)
    stage_times["register_s"] = time.monotonic() - t0

    t0 = time.monotonic()
    records = vs.load_jsonl_records(chunks_path)
    n_written = vs.upsert_records(records) if records else 0
    stage_times["embed_upsert_s"] = time.monotonic() - t0

    conn = vs.get_connection()
    try:
        doc_ids_by_file = vs.resolve_pdf_document_ids_or_raise(conn, records) if records else {}
    finally:
        conn.close()
    current_keys = {(doc_ids_by_file[r["source_file"]], r["page"], r["chunk_index"]) for r in records}

    return {
        "n_files_in_manifest": chunk_summary["n_files"],
        "n_chunks": chunk_summary["n_chunks"],
        "n_reused_files": chunk_summary["n_reused_files"],
        "n_rechunked_files": chunk_summary["n_rechunked_files"],
        "n_embedded_this_run": n_written,
        "register_summary": reg_summary,
        "current_keys": current_keys,
        "chunks_path": chunks_path,
    }


def _snapshot_db(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM pdf_documents WHERE is_active")
        n_active_docs = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM disclosure_chunks")
        n_chunks = cur.fetchone()[0]
        cur.execute("SELECT pdf_document_id, page, chunk_index, chunk_hash FROM disclosure_chunks")
        rows = cur.fetchall()
    return {
        "n_active_documents": n_active_docs,
        "n_chunks": n_chunks,
        "keys": {(r[0], r[1], r[2]) for r in rows},
        "hashes": {(r[0], r[1], r[2]): r[3] for r in rows},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdf_disclosures"),
                     help="Real PDF corpus directory (read-only - never modified by this script)")
    ap.add_argument("--holdout", type=int, default=5,
                     help="Number of real PDFs held out of the Phase-A base corpus, then added back "
                          "in Phase B to simulate an 'add N new PDFs' incremental run. 0 = pure "
                          "no-op re-run instead.")
    ap.add_argument("--report", type=Path, default=Path("reports/incremental_vs_full_rebuild.json"))
    ap.add_argument("--fake-embeddings", action="store_true",
                     help="Use a deterministic stub instead of the real bge-m3 model - dry-run "
                          "this script's plumbing only, NOT for real ablation numbers.")
    args = ap.parse_args()

    _require_test_db()

    from src.retrieval import vector_store as vs
    if args.fake_embeddings:
        import hashlib
        import struct
        from src.retrieval.embeddings import EMBEDDING_DIM

        def _fake_embed_texts(texts, batch_size=32):
            out = []
            for t in texts:
                h = hashlib.sha256(t.encode("utf-8")).digest()
                state = struct.unpack("Q", h[:8])[0]
                vec = []
                for _ in range(EMBEDDING_DIM):
                    state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
                    vec.append((state / (1 << 64)) * 2 - 1)
                norm = sum(x * x for x in vec) ** 0.5
                out.append([x / norm for x in vec])
            return out

        vs.embed_texts = _fake_embed_texts

    conn = vs.get_connection()
    conn.autocommit = True

    all_pdfs = sorted(args.raw_dir.glob("*.pdf"))
    if not all_pdfs:
        raise SystemExit(f"No PDFs found in {args.raw_dir}")

    holdout_n = min(args.holdout, len(all_pdfs) - 1) if args.holdout > 0 else 0
    holdout_files = all_pdfs[-holdout_n:] if holdout_n else []
    base_files = all_pdfs[: len(all_pdfs) - holdout_n] if holdout_n else all_pdfs

    work_root = Path("data/experiments/incremental_vs_full_rebuild")
    if work_root.exists():
        shutil.rmtree(work_root)
    base_dir = work_root / "base_input"
    base_dir.mkdir(parents=True)
    for p in base_files:
        (base_dir / p.name).symlink_to(p.resolve())

    print(f"Base corpus: {len(base_files)} file(s), holdout: {len(holdout_files)} file(s) "
          f"({'no-op re-run' if holdout_n == 0 else 'added in Phase B'})")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_dir": str(args.raw_dir),
        "n_total_pdfs": len(all_pdfs),
        "n_base_pdfs": len(base_files),
        "n_holdout_pdfs": len(holdout_files),
        "embeddings_mode": "fake_stub_DRY_RUN_ONLY" if args.fake_embeddings else "real_bge_m3",
    }

    # ---- Phase A: full rebuild of the BASE corpus (empty DB to start) ----
    print("\n=== Phase A: full rebuild (base corpus, empty DB) ===")
    _wipe_db(conn)
    phase_a_times = {}
    phase_a_work = work_root / "phase_a"
    phase_a_result = _run_full_pipeline(base_dir, phase_a_work, phase_a_times)
    phase_a_snapshot = _snapshot_db(conn)
    report["phase_a_full_rebuild_base"] = {
        "stage_times_s": phase_a_times,
        "total_s": sum(phase_a_times.values()),
        "n_chunks": phase_a_snapshot["n_chunks"],
        "n_embedded_this_run": phase_a_result["n_embedded_this_run"],
        "n_active_documents": phase_a_snapshot["n_active_documents"],
    }
    print(f"  {phase_a_snapshot['n_active_documents']} doc(s), {phase_a_snapshot['n_chunks']} chunk(s), "
          f"{phase_a_result['n_embedded_this_run']} embedded, "
          f"{sum(phase_a_times.values()):.1f}s total")

    # ---- Phase B: incremental - add the holdout files (or re-run as-is) ----
    label = "no-op re-run" if holdout_n == 0 else f"add {holdout_n} new PDF(s)"
    print(f"\n=== Phase B: incremental ({label}) ===")
    incremental_dir = work_root / "incremental_input"
    shutil.copytree(base_dir, incremental_dir, symlinks=True)
    for p in holdout_files:
        (incremental_dir / p.name).symlink_to(p.resolve())

    phase_b_times = {}
    phase_b_work = work_root / "phase_b"
    phase_b_result = _run_full_pipeline(incremental_dir, phase_b_work, phase_b_times)
    phase_b_snapshot = _snapshot_db(conn)
    report["phase_b_incremental"] = {
        "stage_times_s": phase_b_times,
        "total_s": sum(phase_b_times.values()),
        "n_chunks": phase_b_snapshot["n_chunks"],
        "n_embedded_this_run": phase_b_result["n_embedded_this_run"],
        "n_active_documents": phase_b_snapshot["n_active_documents"],
    }
    savings_pct = (
        100.0 * (1 - phase_b_result["n_embedded_this_run"] / max(phase_a_result["n_embedded_this_run"], 1))
    )
    report["embedding_savings_pct_vs_phase_a"] = round(savings_pct, 1)
    print(f"  {phase_b_snapshot['n_active_documents']} doc(s), {phase_b_snapshot['n_chunks']} chunk(s), "
          f"{phase_b_result['n_embedded_this_run']} embedded (vs {phase_a_result['n_embedded_this_run']} "
          f"in Phase A -> {savings_pct:.1f}% fewer embedding calls), "
          f"{sum(phase_b_times.values()):.1f}s total")

    incremental_final = _snapshot_db(conn)

    # ---- Phase C: true full rebuild of the FULL (base+holdout) corpus, from
    #      an empty DB, for the equivalence check + a second timing point ----
    print("\n=== Phase C: full rebuild (full corpus, empty DB) - equivalence baseline ===")
    _wipe_db(conn)
    phase_c_times = {}
    phase_c_work = work_root / "phase_c"
    phase_c_result = _run_full_pipeline(incremental_dir, phase_c_work, phase_c_times)
    phase_c_snapshot = _snapshot_db(conn)
    report["phase_c_full_rebuild_final"] = {
        "stage_times_s": phase_c_times,
        "total_s": sum(phase_c_times.values()),
        "n_chunks": phase_c_snapshot["n_chunks"],
        "n_embedded_this_run": phase_c_result["n_embedded_this_run"],
        "n_active_documents": phase_c_snapshot["n_active_documents"],
    }
    print(f"  {phase_c_snapshot['n_active_documents']} doc(s), {phase_c_snapshot['n_chunks']} chunk(s), "
          f"{phase_c_result['n_embedded_this_run']} embedded, "
          f"{sum(phase_c_times.values()):.1f}s total")

    keys_match = incremental_final["keys"] == phase_c_snapshot["keys"]
    hashes_match = incremental_final["hashes"] == phase_c_snapshot["hashes"]
    report["equivalence"] = {
        "incremental_chunk_keys_eq_full_rebuild_chunk_keys": keys_match,
        "incremental_chunk_hashes_eq_full_rebuild_chunk_hashes": hashes_match,
        "n_keys_only_in_incremental": len(incremental_final["keys"] - phase_c_snapshot["keys"]),
        "n_keys_only_in_full_rebuild": len(phase_c_snapshot["keys"] - incremental_final["keys"]),
    }
    report["total_time_incremental_path_s"] = round(sum(phase_a_times.values()) + sum(phase_b_times.values()), 1)
    report["total_time_full_rebuild_only_s"] = round(sum(phase_c_times.values()), 1)

    print(f"\nEquivalence: keys {'MATCH' if keys_match else 'MISMATCH'}, "
          f"hashes {'MATCH' if hashes_match else 'MISMATCH'}")
    if not (keys_match and hashes_match):
        print("  *** INCREMENTAL/FULL-REBUILD MISMATCH - this is a correctness bug, report it, "
              "do not just report the timing numbers as if everything were fine. ***")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written: {args.report}")

    conn.close()
    shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
