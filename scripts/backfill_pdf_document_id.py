"""One-time backfill for disclosure_chunks.pdf_document_id on an EXISTING
corpus that already has embedded chunks from before migration 006/007
(pdf_documents / disclosure_chunks.pdf_document_id) existed.

WHY THIS SCRIPT EXISTS (found 2026-07-30, running migrations for real
against a corpus with 661,657 already-embedded chunks): migration
007_disclosure_chunks_pdf_document_id.sql's own comment says to backfill
by re-running `python3 -m src.retrieval.vector_store` (upsert_records()),
claiming it's "cheap: no re-embedding needed... ON CONFLICT DO UPDATE just
refreshes pdf_document_id". THAT IS NO LONGER TRUE as of this same day's
migration 010 / Task #2 re-keying: _needs_embedding() now keys existing
rows by (pdf_document_id, page, chunk_index), read directly from
disclosure_chunks. Right after migrations 006/007 run, every existing row's
pdf_document_id is NULL - so _existing_chunk_state() returns keys like
(None, page, chunk_index), which will NEVER match the freshly-resolved
REAL pdf_document_id upsert_records() computes per record. Every single
chunk would look "new" and get FULLY RE-EMBEDDED - hours of GPU/CPU work
that accomplishes nothing except recomputing embeddings that were already
correct.

The actual cheap fix: pdf_documents.source_file and disclosure_chunks.
source_file are the SAME string (see 006_pdf_documents.sql /
001_init_schema.sql's disclosure_chunks definition - `source_file` was
never dropped, only `disclosure_id` was) - a single UPDATE...FROM joining
on that string fills in pdf_document_id for every existing row in one
cheap, index-free-scan pass, no Python, no model, no re-embedding at all.

Run order (after migrations 002-009 and 011, BEFORE migration 010):
    python3 -m scripts.register_pdf_documents --manifest data/processed/ingest_manifest.csv
    python3 -m scripts.backfill_pdf_document_id      <- THIS SCRIPT
    # then check the printed orphan count is 0 before running:
    docker exec -i thesis_postgres psql -U thesis_user -d financial_db \\
        < infra/postgres/migrations/010_lock_pdf_document_id.sql

If orphans remain (a source_file in disclosure_chunks with NO matching row
in pdf_documents), it means register_pdf_documents.py hasn't seen that PDF
yet - this script prints every distinct orphaned source_file so you know
exactly which file(s) to chase down (re-run pdf_ingest_router.py on
whichever raw PDF folder contains them, if register_pdf_documents.py's
--manifest doesn't already cover it).

Safe to re-run: the UPDATE only touches rows still NULL, so a partial run
(new pdf_documents registered since the last backfill pass) just fills in
the newly-coverable rows without re-touching already-backfilled ones.

BATCHED, not one giant UPDATE (fixed 2026-07-30, found running this for
real against the actual 661,657-row corpus): disclosure_chunks rows are
heavy (large chunk_text + VECTOR(1024) embedding), and Postgres MVCC copies
the ENTIRE row into a new version on any UPDATE, even though only
pdf_document_id changes - so one single UPDATE over ~650k such rows ran for
12+ minutes with zero visible progress and zero commits, and got silently
orphaned server-side when the user Ctrl+C'd the (unresponsive-looking)
Python client without cancelling the actual in-flight query (confirmed via
pg_stat_activity: the query kept running for the client's entire lifetime
plus well beyond). Splitting into BATCH_SIZE-row chunks by id range, with a
commit after EACH batch, fixes both problems: progress is printed and
visible batch by batch, and a Ctrl+C (or crash) only loses the CURRENT
in-flight batch's work - every prior batch already committed stays done,
so a re-run picks up close to where it left off (rows already backfilled
are skipped by the `pdf_document_id IS NULL` guard, same as before).
"""
import argparse

MIN_MAX_NULL_ID_SQL = "SELECT min(id), max(id) FROM disclosure_chunks WHERE pdf_document_id IS NULL"

BACKFILL_BATCH_SQL = """
    UPDATE disclosure_chunks dc
    SET pdf_document_id = pd.id
    FROM pdf_documents pd
    WHERE dc.source_file = pd.source_file
      AND dc.pdf_document_id IS NULL
      AND dc.id >= %s AND dc.id < %s
"""

COUNT_REMAINING_NULL_SQL = "SELECT count(*) FROM disclosure_chunks WHERE pdf_document_id IS NULL"

ORPHANED_SOURCE_FILES_SQL = """
    SELECT DISTINCT dc.source_file
    FROM disclosure_chunks dc
    LEFT JOIN pdf_documents pd ON dc.source_file = pd.source_file
    WHERE dc.pdf_document_id IS NULL AND pd.id IS NULL
    ORDER BY dc.source_file
"""

DEFAULT_BATCH_SIZE = 20000


def backfill(batch_size: int = DEFAULT_BATCH_SIZE, progress_callback=None) -> dict:
    """progress_callback(lo, hi, max_id, n_updated_batch, n_updated_total), if
    given, is called after EVERY committed batch - main() uses this to print
    progress; a caller that doesn't care about progress can omit it."""
    from src.retrieval.vector_store import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(MIN_MAX_NULL_ID_SQL)
            min_id, max_id = cur.fetchone()

        n_updated_total = 0
        if min_id is not None:
            lo = min_id
            while lo <= max_id:
                hi = lo + batch_size
                with conn.cursor() as cur:
                    cur.execute(BACKFILL_BATCH_SQL, (lo, hi))
                    n_updated_batch = cur.rowcount
                conn.commit()  # commit EVERY batch - see module docstring
                n_updated_total += n_updated_batch
                if progress_callback:
                    progress_callback(lo, hi, max_id, n_updated_batch, n_updated_total)
                lo = hi

        with conn.cursor() as cur:
            cur.execute(COUNT_REMAINING_NULL_SQL)
            n_remaining_null = cur.fetchone()[0]

            cur.execute(ORPHANED_SOURCE_FILES_SQL)
            orphaned_source_files = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()

    return {
        "n_updated": n_updated_total,
        "n_remaining_null": n_remaining_null,
        "orphaned_source_files": orphaned_source_files,
    }


def _print_progress(lo, hi, max_id, n_updated_batch, n_updated_total):
    print(f"  batch id [{lo}, {hi}): +{n_updated_batch} row(s) backfilled "
          f"(running total: {n_updated_total}) - up to id {min(hi, max_id)}/{max_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                     help=f"Rows per UPDATE batch, committed after each (default {DEFAULT_BATCH_SIZE}). "
                          f"Smaller = more frequent progress/commits, more total overhead; larger = "
                          f"fewer commits, longer gaps between visible progress.")
    args = ap.parse_args()

    result = backfill(args.batch_size, progress_callback=_print_progress)
    print(f"Backfilled pdf_document_id for {result['n_updated']} disclosure_chunks row(s) "
          f"via source_file match against pdf_documents.")
    if result["n_remaining_null"] == 0:
        print("0 row(s) remain with pdf_document_id IS NULL - safe to run migration "
              "010_lock_pdf_document_id.sql now.")
    else:
        print(f"{result['n_remaining_null']} row(s) STILL have pdf_document_id IS NULL - "
              f"DO NOT run migration 010 yet, it will refuse (pre-flight NULL check).")
        print(f"{len(result['orphaned_source_files'])} distinct source_file(s) have no matching "
              f"pdf_documents row - run scripts/register_pdf_documents.py again after making sure "
              f"data/processed/ingest_manifest.csv covers these files:")
        for sf in result["orphaned_source_files"][:50]:
            print(f"  - {sf}")
        if len(result["orphaned_source_files"]) > 50:
            print(f"  ... and {len(result['orphaned_source_files']) - 50} more")


if __name__ == "__main__":
    main()
