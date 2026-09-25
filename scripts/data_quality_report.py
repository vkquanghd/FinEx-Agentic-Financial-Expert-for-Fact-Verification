"""Task #23, roadmap item 8 (2026-07-30): automated Data Quality checks run
on demand (intended to be run after every real ETL/embedding/feature-build
pass) - "no duplicate natural keys; no chunk missing pdf_document_id; no
embedding with wrong dimension; no chunk_hash mismatched against its text;
no macro feature dated after information_cutoff; no target dated before
target_observed_at; no ABSA llm_unreviewed rows in dev/test; no single
pdf_document_id split across multiple ABSA splits." Outputs one JSON report
in (and extending) the roadmap's own example shape.

Every check here is DEFENSE IN DEPTH, not a replacement for the DB-level
constraints/application-level guards already in place elsewhere in this
project (UNIQUE constraints, migration 010's NOT NULL lock, feature_
builder.py's per-feature macro leakage nulling, annotation_schema.py's
validate_annotation() refusing llm_unreviewed+dev/test at write time) -
this script's whole point is "verify code matches docs" (the same
motivation behind Task #15's integration test): confirm those guards
actually held, on the REAL current state of the data, rather than trusting
that they did.

No new DB table (per the roadmap's "stop expanding the architecture") -
JSON report file only.

Usage (needs a live Postgres connection - cannot be exercised in the
sandbox that wrote this):
    python3 -m scripts.data_quality_report
    python3 -m scripts.data_quality_report --report reports/data_quality_2026-07-30.json
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

DUPLICATE_KEY_CHECKS = {
    # table -> natural key columns (matching each table's real UNIQUE
    # constraint in infra/postgres/migrations/001_init_schema.sql, quoted
    # in each check's own docstring below) - checked independently of the
    # DB constraint as defense-in-depth (catches legacy pre-constraint rows
    # or any direct-SQL bypass, not just future inserts).
    "disclosures": ["ticker", "fiscal_year", "period", "consolidation"],
    "pdf_documents": ["source_file"],
    "disclosure_chunks": ["pdf_document_id", "page", "chunk_index"],
    "predictive_features": ["feature_version", "ticker", "fiscal_year", "period", "consolidation", "target_col", "horizon"],
    "causal_panel_snapshot": ["snapshot_version", "ticker", "fiscal_year", "period", "consolidation"],
}


def check_duplicate_keys(conn) -> dict:
    """Re-derives what infra/postgres/migrations/001_init_schema.sql's own
    UNIQUE constraints already should have prevented, table by table -
    GROUP BY the natural key, count groups with > 1 row.

    Rows with ANY NULL natural-key column are excluded (WHERE col IS NOT
    NULL for every col), not just left in - fixed 2026-07-30, found running
    this for real mid-backfill: Postgres' GROUP BY treats NULL as equal to
    NULL, so while disclosure_chunks.pdf_document_id is still NULL for the
    whole table (pre-backfill transitional state, see
    scripts/backfill_pdf_document_id.py), every row sharing the same
    (page, chunk_index) across UNRELATED documents got grouped together as
    a spurious "duplicate" (this produced the misleading duplicate_keys=5475
    in the user's first real report). A row with a NULL key column isn't a
    duplicate in any meaningful sense - it's an INCOMPLETENESS problem,
    already tracked separately and honestly by check_null_pdf_document_id()."""
    by_table = {}
    by_table_null_key_rows = {}
    total = 0
    with conn.cursor() as cur:
        for table, cols in DUPLICATE_KEY_CHECKS.items():
            col_list = ", ".join(cols)
            not_null_where = " AND ".join(f"{c} IS NOT NULL" for c in cols)
            cur.execute(
                f"SELECT count(*) FROM (SELECT {col_list} FROM {table} "
                f"WHERE {not_null_where} "
                f"GROUP BY {col_list} HAVING COUNT(*) > 1) dup"
            )
            n = cur.fetchone()[0]
            by_table[table] = n
            total += n

            cur.execute(
                f"SELECT count(*) FROM {table} WHERE NOT ({not_null_where})"
            )
            by_table_null_key_rows[table] = cur.fetchone()[0]
    return {"total": total, "by_table": by_table, "by_table_null_key_rows_excluded": by_table_null_key_rows}


def check_placeholder_pdf_hash(conn) -> dict:
    """Counts pdf_documents rows registered with scripts/register_pdf_
    documents.py's MISSING_SOURCE_PDF_HASH_SENTINEL (added 2026-07-30,
    found running this for real: ~2890 of the real corpus's original PDF
    files are permanently gone from --pdf-dir - confirmed not recoverable
    from any on-disk backup, including data/raw/tinix_ocr's OCR-text-only
    archive - so --allow-missing-pdf-placeholder was used to unblock
    migration 010 rather than leave the DB stuck indefinitely).

    Informational, NOT counted in "ok" (same treatment as stale_embeddings/
    chunk_hash_not_yet_computed) - a nonzero count here is an ACKNOWLEDGED,
    deliberate gap (source PDF unavailable for content-hash verification),
    not a bug, but it must stay visible in every report, never silently
    disappear - this is the whole point of giving it its own tracked
    check instead of just letting it blend into "wrong_dim_embeddings=0,
    everything else is fine"."""
    from scripts.register_pdf_documents import MISSING_SOURCE_PDF_HASH_SENTINEL

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM pdf_documents WHERE pdf_hash = %s",
                     (MISSING_SOURCE_PDF_HASH_SENTINEL,))
        n_placeholder = cur.fetchone()[0]

        cur.execute("SELECT source_file FROM pdf_documents WHERE pdf_hash = %s ORDER BY source_file LIMIT 20",
                     (MISSING_SOURCE_PDF_HASH_SENTINEL,))
        sample = [r[0] for r in cur.fetchall()]
    return {"n_placeholder": n_placeholder, "sample_source_files": sample}


def check_null_pdf_document_id(conn) -> int:
    """Should always be 0 once migration 010_lock_pdf_document_id.sql has
    run (NOT NULL constraint) - checked anyway for an install that hasn't
    run that migration yet, or a direct-SQL bypass."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM disclosure_chunks WHERE pdf_document_id IS NULL")
        return cur.fetchone()[0]


def check_wrong_dim_embeddings(conn, expected_dim: int = 1024) -> int:
    """pgvector's VECTOR(1024) column type already refuses an insert of the
    wrong dimension at write time - this is a paranoid re-check, not
    expected to ever find anything, but costs one cheap query."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM disclosure_chunks WHERE embedding IS NOT NULL "
            f"AND vector_dims(embedding) != {expected_dim}"
        )
        return cur.fetchone()[0]


def check_orphan_chunks(conn) -> int:
    """Chunks whose PARENT pdf_documents row is soft-deleted (is_active =
    FALSE, see scripts/register_pdf_documents.py's deactivate_deleted(),
    Task #15) but haven't been pruned via `vector_store.py --prune-orphans`
    yet - a real, expected-to-sometimes-be-nonzero count (pruning is a
    manual, explicit step, not automatic), not a correctness bug by
    itself, but worth surfacing so it isn't forgotten indefinitely."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM disclosure_chunks dc "
            "JOIN pdf_documents pd ON dc.pdf_document_id = pd.id "
            "WHERE pd.is_active = FALSE"
        )
        return cur.fetchone()[0]


def check_stale_embeddings(conn, current_embedding_version: str) -> int:
    """Chunks embedded under a DIFFERENT embedding_version than the one
    src/retrieval/vector_store.py currently computes - these are stale, not
    wrong (they were valid under an older model/config), but a large count
    here means a large fraction of the corpus hasn't been re-embedded since
    the last EMBEDDING_VERSION bump."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM disclosure_chunks WHERE embedding IS NOT NULL "
            "AND embedding_version != %s",
            (current_embedding_version,),
        )
        return cur.fetchone()[0]


def check_chunk_hash_mismatches(conn) -> dict:
    """Recomputes text_splitter.chunk_hash()'s exact SHA-256-of-normalized-
    text for every stored chunk and compares against the stored chunk_hash
    - catches a hand-edited row, a partial/corrupted write, or a hashing
    logic drift between what's stored and what the CURRENT chunk_hash()
    would compute. Fetches chunk_text for every row - documented as
    potentially expensive on a very large corpus (this is an explicit
    trade-off: the check IS the recomputation, there's no cheaper proxy for
    "does chunk_hash still match this exact text").

    A NULL stored_hash is counted separately as "not yet computed", NOT as
    a mismatch - fixed 2026-07-30, found running this for real: a chunk
    embedded before the chunk_hash column existed legitimately has
    stored_hash=NULL, and compute_chunk_hash(text) can never equal None, so
    the old `!=` comparison silently counted every such row as "mismatched"
    (inflating the user's first real report to chunk_hash_mismatches=4739).
    "Not yet computed" and "genuinely corrupted/tampered" call for very
    different responses (re-run the hash backfill vs. investigate data
    corruption), so they must not be conflated into one number."""
    from src.retrieval.text_splitter import chunk_hash as compute_chunk_hash

    n_checked = n_mismatched = n_null_hash = 0
    sample_mismatches = []
    sample_null_hash = []
    with conn.cursor() as cur:
        cur.execute("SELECT pdf_document_id, page, chunk_index, chunk_text, chunk_hash FROM disclosure_chunks")
        for pdf_document_id, page, chunk_index, text, stored_hash in cur.fetchall():
            n_checked += 1
            if stored_hash is None:
                n_null_hash += 1
                if len(sample_null_hash) < 20:
                    sample_null_hash.append({"pdf_document_id": pdf_document_id, "page": page, "chunk_index": chunk_index})
                continue
            if compute_chunk_hash(text) != stored_hash:
                n_mismatched += 1
                if len(sample_mismatches) < 20:
                    sample_mismatches.append({"pdf_document_id": pdf_document_id, "page": page, "chunk_index": chunk_index})
    return {
        "n_checked": n_checked,
        "n_mismatched": n_mismatched,
        "sample_mismatches": sample_mismatches,
        "n_null_hash": n_null_hash,
        "sample_null_hash": sample_null_hash,
    }


def check_macro_leakage(session) -> dict:
    """Re-derives feature_builder.py's OWN per-feature macro leakage guard
    (module docstring point 4: a macro covariate is nulled out per-row if
    the latest release_date feeding that year's aggregate is AFTER this
    row's information_cutoff) directly against the STORED predictive_
    features rows, using the exact same _max_release_date()/_MACRO_
    RELEASE_SOURCES the builder itself uses (not a re-derived
    approximation) - confirms the guard actually held for every row
    currently in the table, not just that the code path exists."""
    from sqlalchemy import text
    from src.predictive.feature_builder import MACRO_COLUMNS, _MACRO_RELEASE_SOURCES, _max_release_date

    rows = session.execute(text(
        f"SELECT id, ticker, fiscal_year, information_cutoff, {', '.join(MACRO_COLUMNS)} FROM predictive_features"
    )).fetchall()

    release_cache = {}
    n_checked = n_violations = 0
    sample_violations = []
    for row in rows:
        row_id, ticker, fiscal_year, information_cutoff, *macro_values = row
        for col, value in zip(MACRO_COLUMNS, macro_values):
            if value is None or information_cutoff is None:
                continue
            n_checked += 1
            key = (col, int(fiscal_year))
            if key not in release_cache:
                category, pattern, use_like = _MACRO_RELEASE_SOURCES[col]
                release_cache[key] = _max_release_date(session, category, pattern, int(fiscal_year), use_like)
            max_release = release_cache[key]
            if max_release is not None and max_release > information_cutoff:
                n_violations += 1
                if len(sample_violations) < 20:
                    sample_violations.append({"predictive_features_id": row_id, "ticker": ticker, "macro_column": col})
    return {"n_checked": n_checked, "n_violations": n_violations, "sample_violations": sample_violations}


def check_target_causality(session) -> int:
    """target_observed_at is the information_cutoff of the FUTURE row that
    supplied target_value (see 001_init_schema.sql's comment) - it must
    always be AFTER this row's own information_cutoff (you cannot observe
    a future target before the present features it's predicted from)."""
    from sqlalchemy import text
    result = session.execute(text(
        "SELECT count(*) FROM predictive_features "
        "WHERE target_observed_at IS NOT NULL AND information_cutoff IS NOT NULL "
        "AND target_observed_at <= information_cutoff"
    )).fetchone()
    return result[0]


def check_absa_quality(annotations_path: Path) -> dict:
    """File-based checks (no DB) over data/eval/absa_annotations.jsonl:
    (1) llm_unreviewed rows in split=dev/test - already refused at WRITE
    time by annotation_schema.validate_annotation(), this re-checks the
    file as it actually sits on disk (catches a hand-edited/bypassed row);
    (2) a single real PDF document (proxied by (ticker, source_file), since
    annotation rows don't store pdf_document_id directly) whose rows span
    MORE THAN ONE split value - a train/test leakage risk the roadmap's
    item 5 explicitly warned about ("split by pdf_document_id, not
    randomly by row")."""
    from src.absa.annotation_schema import load_annotations

    records = load_annotations(annotations_path)
    llm_unreviewed_in_dev_test = [
        r for r in records if r.get("label_source") == "llm_unreviewed" and r.get("split") in ("dev", "test")
    ]

    by_document = {}
    for r in records:
        if r.get("source_file") is None:
            continue
        doc_key = (r.get("ticker"), r["source_file"])
        by_document.setdefault(doc_key, set()).add(r.get("split"))
    split_leakage_documents = [
        {"ticker": t, "source_file": sf, "splits": sorted(splits)}
        for (t, sf), splits in by_document.items() if len(splits) > 1
    ]

    return {
        "n_records": len(records),
        "n_llm_unreviewed_in_dev_test": len(llm_unreviewed_in_dev_test),
        "sample_llm_unreviewed_in_dev_test": [r.get("id") for r in llm_unreviewed_in_dev_test[:20]],
        "n_documents_split_across_multiple_splits": len(split_leakage_documents),
        "sample_split_leakage_documents": split_leakage_documents[:20],
    }


def build_report() -> dict:
    from src.core.db import get_session
    from src.retrieval.vector_store import EMBEDDING_VERSION, get_connection
    from src.absa.annotation_schema import ANNOTATIONS_PATH

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pdf_documents WHERE is_active")
            n_documents = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM disclosure_chunks")
            n_chunks = cur.fetchone()[0]

        duplicate_keys = check_duplicate_keys(conn)
        null_pdf_document_id = check_null_pdf_document_id(conn)
        wrong_dim_embeddings = check_wrong_dim_embeddings(conn)
        orphan_chunks = check_orphan_chunks(conn)
        stale_embeddings = check_stale_embeddings(conn, EMBEDDING_VERSION)
        chunk_hash_check = check_chunk_hash_mismatches(conn)
        placeholder_pdf_hash = check_placeholder_pdf_hash(conn)
    finally:
        conn.close()

    with get_session() as session:
        macro_leakage = check_macro_leakage(session)
        target_causality_violations = check_target_causality(session)

    absa_quality = check_absa_quality(ANNOTATIONS_PATH)

    predictive_leakage_rows = macro_leakage["n_violations"] + target_causality_violations

    report = {
        "run_id": f"dq_{time.strftime('%Y%m%dT%H%M%S')}",
        "documents": n_documents,
        "chunks": n_chunks,
        "orphan_chunks": orphan_chunks,
        "null_pdf_document_id": null_pdf_document_id,
        "stale_embeddings": stale_embeddings,
        "predictive_leakage_rows": predictive_leakage_rows,
        "duplicate_keys": duplicate_keys["total"],
        "duplicate_keys_by_table": duplicate_keys["by_table"],
        "duplicate_keys_by_table_null_key_rows_excluded": duplicate_keys["by_table_null_key_rows_excluded"],
        "wrong_dim_embeddings": wrong_dim_embeddings,
        "chunk_hash_mismatches": chunk_hash_check["n_mismatched"],
        "chunk_hash_checked": chunk_hash_check["n_checked"],
        "chunk_hash_sample_mismatches": chunk_hash_check["sample_mismatches"],
        "chunk_hash_not_yet_computed": chunk_hash_check["n_null_hash"],
        "chunk_hash_sample_not_yet_computed": chunk_hash_check["sample_null_hash"],
        "macro_leakage_rows": macro_leakage["n_violations"],
        "macro_leakage_checked": macro_leakage["n_checked"],
        "macro_leakage_sample_violations": macro_leakage["sample_violations"],
        "target_causality_violations": target_causality_violations,
        "absa_llm_unreviewed_in_dev_test": absa_quality["n_llm_unreviewed_in_dev_test"],
        "absa_split_leakage_documents": absa_quality["n_documents_split_across_multiple_splits"],
        "absa_sample_split_leakage_documents": absa_quality["sample_split_leakage_documents"],
        "absa_n_records": absa_quality["n_records"],
        "placeholder_pdf_hash": placeholder_pdf_hash["n_placeholder"],
        "placeholder_pdf_hash_sample_source_files": placeholder_pdf_hash["sample_source_files"],
    }
    report["ok"] = all(
        report[k] == 0 for k in (
            "orphan_chunks", "null_pdf_document_id", "duplicate_keys", "wrong_dim_embeddings",
            "chunk_hash_mismatches", "macro_leakage_rows", "target_causality_violations",
            "absa_llm_unreviewed_in_dev_test", "absa_split_leakage_documents",
        )
    )  # stale_embeddings deliberately excluded from "ok" - a nonzero count is expected/routine
       # (re-embedding is a manual, explicit step), not a correctness failure by itself.
       # placeholder_pdf_hash also deliberately excluded - a nonzero count is an
       # ACKNOWLEDGED gap (source PDF confirmed unrecoverable, see
       # check_placeholder_pdf_hash()'s docstring), not a bug to fail the run over,
       # but still surfaced every time so it's never forgotten.
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=Path("reports/data_quality_report.json"))
    args = ap.parse_args()

    report = build_report()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"=== Data Quality Report ({report['run_id']}) ===")
    print(f"documents={report['documents']}  chunks={report['chunks']}")
    for key in ("orphan_chunks", "null_pdf_document_id", "stale_embeddings", "predictive_leakage_rows",
                "duplicate_keys", "wrong_dim_embeddings", "chunk_hash_mismatches", "macro_leakage_rows",
                "target_causality_violations", "absa_llm_unreviewed_in_dev_test", "absa_split_leakage_documents"):
        flag = "" if report[key] == 0 else "  <-- NONZERO"
        print(f"  {key:35s} {report[key]}{flag}")
    # Informational only, not part of "ok" (same treatment as stale_embeddings) -
    # rows legitimately embedded before chunk_hash existed, not a defect.
    if report["chunk_hash_not_yet_computed"]:
        print(f"  {'chunk_hash_not_yet_computed':35s} {report['chunk_hash_not_yet_computed']}  "
              f"(informational - pre-chunk_hash rows, not a mismatch)")
    if report["null_pdf_document_id"] > 0:
        print("  NOTE: pdf_document_id IS NULL for some rows - duplicate_keys check on "
              "disclosure_chunks excludes those rows (see duplicate_keys_by_table_null_key_rows_excluded "
              "in the JSON report) rather than spuriously grouping them together.")
    if report["placeholder_pdf_hash"]:
        print(f"  {'placeholder_pdf_hash':35s} {report['placeholder_pdf_hash']}  "
              f"(informational - source PDF confirmed unrecoverable, registered via "
              f"--allow-missing-pdf-placeholder; see placeholder_pdf_hash_sample_source_files)")
    print(f"\noverall ok={report['ok']}")
    print(f"Report written: {args.report}")


if __name__ == "__main__":
    main()
