"""Loads a `chunks_embedded.jsonl` file (produced by
notebooks/colab_embed_chunks.ipynb running on Colab's GPU) and upserts it
into pgvector. No model loading here - this is the "local, DB-only" half
of the embed/upsert split done 2026-07-21 to move the GPU-heavy bge-m3
embedding step off the user's Mac (M3 Max, but a 656,918-chunk job was
tying up the machine for hours) onto Colab.

Commits every `--batch-commit` rows (default 500), not once at the end -
same crash-safety fix applied to vector_store.upsert_records() the same
day (a Ctrl+C mid-run under a single giant transaction would otherwise
discard everything already processed). Safe to interrupt and re-run with
the same input file - UPSERT_SQL's ON CONFLICT DO UPDATE makes re-writing
an already-loaded row a harmless no-op.

Accepts a plain `.jsonl` or a gzip-compressed `.jsonl.gz` transparently
(the notebook's download cell gzips the output before sending it to the
browser, since raw embeddings inflate the file ~10x over the source
chunks.jsonl) - no need to unzip by hand first.

2026-07-29: UPSERT_SQL gained two more columns (chunk_hash, embedding_version
- see vector_store.py's stale-embedding fix) after this script was written,
which silently broke it - it still called cur.execute(UPSERT_SQL, ...) with
only the OLD 8-value tuple, so the very next run of this exact script would
have failed with a "not enough arguments" error the moment a fresh Python
process imported the new UPSERT_SQL (a run already in progress before that
fix keeps using whatever UPSERT_SQL string it imported at startup, which is
why an old run could still be printing progress after the source file
changed underneath it). Fixed by adding the same _record_chunk_hash()
fallback + EMBEDDING_VERSION used by vector_store.py's own
upsert_precomputed() - records from an older Colab run with no "chunk_hash"
field get it computed on the fly from chunk_text.

2026-07-29 (same day, third param-count change - migration
003_disclosure_lineage_and_versions.sql): UPSERT_SQL gained THREE more
columns (disclosure_id, chunking_version, normalization_version). Fixed the
same way, via vector_store.py's own _lineage_fields(conn, r) helper - a
record from an old Colab file with no chunking_version/normalization_version
field just gets None for those (honest "unknown", not a guess).
Run migration 003 before re-running this script, same as migration 002.

2026-07-29 (evening, same param COUNT but a semantic/name change - migration
006_pdf_documents.sql + 007_disclosure_chunks_pdf_document_id.sql):
disclosure_id was renamed pdf_document_id and now points at a NEW
pdf_documents table instead of `disclosures` - see
vector_store.resolve_pdf_document_id()'s docstring for why (the old
ticker+year+doctype match against `disclosures` couldn't disambiguate
multiple real PDFs for the same ticker+year; pdf_documents gives every
physical PDF its own row instead).

2026-07-30 (Task #2 from the post-v3 methodology review, migration
010_lock_pdf_document_id.sql): disclosure_chunks.pdf_document_id is now
NOT NULL - a row from a file scripts/register_pdf_documents.py hasn't seen
yet can no longer be silently written with pdf_document_id=NULL. This
script now does a cheap FIRST PASS over --input (parsing just the
"source_file" field of every line, no vector/embedding work) to collect
every distinct source_file, resolves them all via
vector_store.resolve_pdf_document_ids_or_raise() BEFORE the real write
pass starts, and raises UnregisteredDocumentError (naming every missing
file) if any weren't registered - run scripts/register_pdf_documents.py
first if this happens. _lineage_fields()'s signature also changed (now
takes the already-resolved pdf_document_id as a plain argument instead of
looking it up itself per-record via a live `conn`) - updated below.

Usage:
    python3 -m scripts.upsert_precomputed_embeddings --input /path/to/chunks_embedded.jsonl
    python3 -m scripts.upsert_precomputed_embeddings --input /path/to/chunks_embedded.jsonl.gz
"""
import argparse
import gzip
import json
from pathlib import Path

from src.retrieval.vector_store import (
    EMBEDDING_VERSION, UPSERT_SQL, _lineage_fields, _record_chunk_hash, _vector_literal, get_connection,
    resolve_pdf_document_ids_or_raise,
)


def load_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="chunks_embedded.jsonl from the Colab notebook")
    ap.add_argument("--batch-commit", type=int, default=500)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    conn = get_connection()
    n = 0
    try:
        # First pass (cheap - source_file only, no vector work): collect
        # every distinct source_file and fail loudly, up front, if any is
        # unregistered - see this script's 2026-07-30 docstring update.
        distinct_stub_records = [{"source_file": r["source_file"]} for r in load_jsonl(path)]
        doc_ids_by_file = resolve_pdf_document_ids_or_raise(conn, distinct_stub_records)

        with conn.cursor() as cur:
            for r in load_jsonl(path):
                pdf_document_id, chunking_version, normalization_version = _lineage_fields(
                    r, doc_ids_by_file[r["source_file"]])
                cur.execute(UPSERT_SQL, (
                    r["ticker"], r["doctype"], r["source_file"], r["page"],
                    r["chunk_index"], r["chunk_text"], r["char_count"],
                    _vector_literal(r["embedding"]), _record_chunk_hash(r), EMBEDDING_VERSION,
                    pdf_document_id, chunking_version, normalization_version,
                ))
                n += 1
                if n % args.batch_commit == 0:
                    conn.commit()
                    print(f"  upserted {n} (committed)")
            conn.commit()
    finally:
        conn.close()
    print(f"Done. {n} chunk(s) upserted into disclosure_chunks.")


if __name__ == "__main__":
    main()
