"""
Vector store client (dense retrieval, optional BM25 hybrid).

Talks to the `vectorstore` Postgres/pgvector service defined in
docker-compose.yml (table schema in
infra/pgvector/migrations/001_disclosure_chunks.sql - run that migration
by hand first, see the comment at its top for the exact `docker exec`
command). Uses psycopg2 directly with vectors formatted as pgvector's text
literal ('[0.1,0.2,...]'::vector) rather than pulling in the separate
`pgvector` python package - one fewer dependency, and the literal format is
simple enough not to need a helper library.

Run as a script to embed chunks.jsonl (produced by
src/retrieval/text_splitter.py) and load the result into pgvector:
  python3 -m src.retrieval.vector_store --input data/processed/chunks.jsonl

NOTE: this could not be run end-to-end in the sandbox that wrote it - no
pgvector/postgres available there to connect to, and no network access to
download the bge-m3 model weights. The SQL and embedding logic follow
documented psycopg2/pgvector/sentence-transformers patterns, but run this
for real on your machine and report back any error - don't assume it's
bug-free just because it looks right.
"""
import argparse
import json
import re
from pathlib import Path

from src.core import config
from src.retrieval.embeddings import EMBEDDING_MODEL_NAME, embed_texts
from src.retrieval.text_splitter import chunk_hash as _compute_chunk_hash

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

# Bump the trailing "_v1" if embed_texts()'s behavior changes in a way that
# doesn't change EMBEDDING_MODEL_NAME itself (e.g. normalize_embeddings
# flipped, or a different bge-m3 revision pinned) -- anything that changes
# the actual vectors produced for the same text must be reflected here so
# stale-embedding detection (see _existing_chunk_state() below) knows to
# re-embed rather than trust an old vector computed under a different rule.
EMBEDDING_VERSION = f"{EMBEDDING_MODEL_NAME}_normalized_v1"


class UnregisteredDocumentError(Exception):
    """Raised (2026-07-30, Task #2 from the post-v3 methodology review -
    docs/PROGRESS.md has the full writeup) when one or more records'
    source_file has no row in pdf_documents yet - i.e.
    scripts/register_pdf_documents.py was never run (or hasn't been re-run)
    for that file. Before this, an unresolved pdf_document_id silently
    became NULL in disclosure_chunks (an "honest NULL" - reasonable while
    the registration step was brand new). Now that the full pipeline order
    (pdf_ingest_router.py -> register_pdf_documents.py -> text_splitter.py
    -> vector_store.py) is the ONLY supported order and
    disclosure_chunks.pdf_document_id is NOT NULL at the DB level
    (migration 010), a chunk with no registered document is a BUG in the
    run, not a state to store - failing loudly here, before any DB write
    happens, is much cheaper to debug than a NOT NULL constraint violation
    mid-batch (which could leave a partially-written batch) or, worse, a
    silently accepted NULL on a database that hasn't applied migration 010
    yet."""
    def __init__(self, source_files):
        self.source_files = sorted(source_files)
        super().__init__(
            f"{len(self.source_files)} source_file(s) have no pdf_documents row - run "
            f"scripts/register_pdf_documents.py for these before vector_store.py: "
            f"{self.source_files[:10]}{'...' if len(self.source_files) > 10 else ''}"
        )


_PDF_DOCUMENT_ID_CACHE = {}  # source_file -> id or None, memoized per-process


def resolve_pdf_document_id(conn, source_file: str):
    """EXACT (not fuzzy) lookup of pdf_documents.id for a given source_file.

    SUPERSEDES the 2026-07-29-morning resolve_disclosure_id() (removed the
    same day, evening round) after a review correctly pointed out its real
    gap: matching purely on (ticker, fiscal_year, doctype) can't disambiguate
    when a ticker has SEVERAL real PDFs for the same fiscal year (annual
    report, audited BCTC, a later correction, a governance report) - the old
    function would silently pick one candidate row in `disclosures` (a
    DIFFERENT table, describing structured-filing metadata, not physical
    PDF identity) with no way to know if it picked the right one.

    Fix: give every physical PDF its OWN identity row in a NEW table,
    `pdf_documents` (see infra/postgres/migrations/006_pdf_documents.sql),
    keyed by source_file (UNIQUE) - created by scripts/register_pdf_documents.py
    which runs AFTER pdf_ingest_router.py and BEFORE text_splitter.py, i.e.
    the id is now ASSIGNED at ingest time (deterministic, unambiguous: one
    row per PDF file), not GUESSED later from ticker+year+doctype. This
    function is therefore no longer a fuzzy resolver - it's a plain unique
    key lookup, with only in-process memoization (no ambiguity possible)."""
    if source_file in _PDF_DOCUMENT_ID_CACHE:
        return _PDF_DOCUMENT_ID_CACHE[source_file]
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM pdf_documents WHERE source_file = %s", (source_file,))
        row = cur.fetchone()
    pdf_document_id = row[0] if row else None
    _PDF_DOCUMENT_ID_CACHE[source_file] = pdf_document_id
    return pdf_document_id


def resolve_pdf_document_ids_or_raise(conn, records: list) -> dict:
    """Resolves pdf_document_id for every DISTINCT source_file across
    `records` UP FRONT (one pass, before any embedding/DB-write work
    starts), and raises UnregisteredDocumentError listing ALL unresolved
    files at once if any are missing - added 2026-07-30 (Task #2) so a run
    over a large batch fails immediately and completely, rather than
    getting partway through embedding (expensive) before discovering file
    #400 of 500 was never registered. Returns {source_file: pdf_document_id}
    for every file in `records` - a plain dict lookup at write time, no
    further DB round-trips needed per-record beyond what
    resolve_pdf_document_id()'s cache already avoids."""
    distinct_files = {r["source_file"] for r in records}
    by_file = {sf: resolve_pdf_document_id(conn, sf) for sf in distinct_files}
    missing = {sf for sf, doc_id in by_file.items() if doc_id is None}
    if missing:
        raise UnregisteredDocumentError(missing)
    return by_file


def extract_year(source_file: str):
    """Best-effort FISCAL year from a source filename. Returns None if no
    4-digit 20xx year is found - common for the real-PDF corpus (e.g. draft
    AGM-resolution filenames with no year in the name), so callers (see
    rag_pipeline.py's temporal-anchor logic, RQ4) must handle None rather
    than assume every chunk has a known year. There is no `year` column in
    disclosure_chunks (see infra/pgvector/migrations/001_disclosure_chunks.sql)
    - extracting from the filename at query time avoids yet another schema
    migration + full re-ingest cycle.

    Takes the LAST 20xx token in the filename, not the first - fixed
    2026-07-22 after a real eval run surfaced 3 confirmed cases (AVG, CMD,
    AAV) where a "auditor of X in 2025" question got the CORRECT 2024
    auditor back mislabeled as 2025. Root cause, confirmed via
    scripts/diagnose_temporal_anchor_year_bleed.py's real filename dump:
    the actual downloaded TiniX corpus's report-folder names encode TWO
    years, e.g. '1_avg_2025_4_1_134382f_vi_baocaotaichinhrieng_kiemtoan_2024'
    - '2025' right after the ticker is the FILING date, 'kiemtoan_2024' at
    the end is the FISCAL year the report is actually about. The original
    single-sample validation (2026-07-19, before this project had general
    internet access to download the full dataset) only ever saw single-year
    filenames like 'PNJ_Baocaotaichinh_2024_Kiemtoan_Hopnhat', where
    first-vs-last made no difference - the bug was invisible until the real
    bulk download revealed the two-year naming convention. Every filename
    sample checked (real-PDF corpus's single-year names, and TiniX's
    two-year '..._kiemtoan_YYYY' / '..._baocaotaichinh_YYYY' names) has the
    fiscal year as the LAST 20xx token, so this is a strict improvement, not
    a new tradeoff - re-verify against scripts/diagnose_temporal_anchor_year_bleed.py's
    output if a future TiniX naming variant shows up that doesn't fit this
    pattern. No re-ingest needed - year is computed at query time, not
    stored, so this takes effect on the very next query."""
    matches = _YEAR_RE.findall(source_file)
    return int(matches[-1]) if matches else None

# v5 (2026-07-30, Task #2 from the post-v3 methodology review): natural key
# LOCKED to (pdf_document_id, page, chunk_index), replacing (ticker,
# source_file, page, chunk_index) - see migration 010_lock_pdf_document_id.sql
# for the full rationale (ticker/source_file are redundant as key material
# once every chunk is guaranteed a real pdf_document_id via
# resolve_pdf_document_ids_or_raise()/UnregisteredDocumentError - they're
# still stored as plain columns, just no longer part of the ON CONFLICT
# target). ticker/doctype are now included in DO UPDATE SET (they weren't
# before, since they used to be PART of the conflict key and therefore could
# never legitimately differ on a conflict) - now that pdf_document_id alone
# identifies the row, a corrected ticker/doctype guess on a later run should
# actually update it. pdf_document_id itself is NOT in DO UPDATE SET: it's
# the conflict target, so Postgres matches on it already; by construction
# (resolve_pdf_document_ids_or_raise() runs before any write in this file) it
# is never NULL going into this statement, so the old COALESCE-guard against
# clobbering a resolved id with NULL is no longer needed.
UPSERT_SQL = """
INSERT INTO disclosure_chunks
    (ticker, doctype, source_file, page, chunk_index, chunk_text, char_count,
     embedding, chunk_hash, embedding_version, pdf_document_id, chunking_version,
     normalization_version)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s)
ON CONFLICT (pdf_document_id, page, chunk_index)
DO UPDATE SET chunk_text = EXCLUDED.chunk_text,
              char_count = EXCLUDED.char_count,
              embedding = EXCLUDED.embedding,
              ticker = EXCLUDED.ticker,
              doctype = EXCLUDED.doctype,
              chunk_hash = EXCLUDED.chunk_hash,
              embedding_version = EXCLUDED.embedding_version,
              chunking_version = EXCLUDED.chunking_version,
              normalization_version = EXCLUDED.normalization_version;
"""

SIMILARITY_SQL = """
SELECT ticker, doctype, source_file, page, chunk_index, chunk_text,
       1 - (embedding <=> %s::vector) AS similarity
FROM disclosure_chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

# Ticker-scoped variant - filters BEFORE the LIMIT instead of after.
# Added 2026-07-19: with the corpus at 18k+ chunks across many tickers, a
# global top-K similarity search for a single-company question can easily
# surface ZERO chunks from the requested ticker (its real matches rank
# outside the global top-K even though they're the best match WITHIN that
# ticker) - confirmed as a real bug via a live query ("Hội đồng quản trị
# PNJ...", 5,328 PNJ rows in the table, none in the global top 15).
# rag_pipeline.answer() used to work around this by over-fetching
# (top_k * 3) and filtering client-side, which is exactly the "wastes
# meaningful top_k slots on the wrong company" case this function's
# original docstring warned would eventually need a real SQL WHERE clause
# instead - this is that fix.
#
# IMPORTANT (found 2026-07-19, same day): adding `WHERE ticker = %s` to
# this query does NOT by itself guarantee it searches all of that ticker's
# rows. Postgres's planner can still choose the ANN index to satisfy
# `ORDER BY embedding <=> ...` (with the original IVFFlat index, it only
# probed 1 of 100 lists by default, ~180 rows out of 18,186), and only THEN
# apply the ticker filter to that small approximate candidate set - not to
# the ticker's full ~5,000 rows. Confirmed live: `similarity_search(...,
# ticker='PNJ', top_k=20)` returned only 1 row even though PNJ has 5,328
# rows in the table. See `similarity_search()` below for the fix (forces an
# exact scan for ticker-scoped queries).
#
# NOTE (2026-07-26, index changed IVFFlat -> HNSW, see
# infra/pgvector/migrations/001_disclosure_chunks.sql): this is NOT an
# IVFFlat-specific defect - it's "index-then-filter" query execution, which
# HNSW does too unless `hnsw.iterative_scan` is enabled (pgvector >= 0.8.0).
# The exact-scan workaround below is therefore kept as-is after the HNSW
# migration, not removed - switching ANN algorithm alone doesn't fix this.
# See the migration file's comment for what would actually let ticker-
# scoped queries use the index again.
SIMILARITY_SQL_TICKER = """
SELECT ticker, doctype, source_file, page, chunk_index, chunk_text,
       1 - (embedding <=> %s::vector) AS similarity
FROM disclosure_chunks
WHERE embedding IS NOT NULL AND ticker = %s
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

# Ticker + year-pattern variant - added 2026-07-19 after confirming (see
# scripts/global_rank_pwc.py, now deleted) that widening fetch_k on
# SIMILARITY_SQL_TICKER alone doesn't reliably work: for "Công ty kiểm toán
# của PNJ năm 2024 là ai?", the correct chunk ranked #33 GLOBALLY across all
# of PNJ's ~5,000 chunks (all years combined) - not deeply buried, but a
# company with many same-year boilerplate chunks (balance sheet pages,
# generic "kiểm toán" mentions that don't actually name the auditor) can
# still crowd out the true answer within the SAME year before a post-hoc
# year filter ever gets to it, no matter how wide fetch_k is set. The fix:
# when the question names a year, filter to that year's files IN SQL before
# ranking, not after - so LIMIT applies to a true "top-K within this year"
# instead of "top-K globally, hopefully with enough of this year in it".
# Relies on the TiniX-sourced filename convention (TICKER_Baocaotaichinh_
# YYYY_...) via a `source_file LIKE %_YYYY_%` pattern - see
# rag_pipeline.answer() for how the pattern is built and the fallback to
# the unfiltered ticker search when no files match (e.g. non-TiniX sources
# with no year in the filename).
SIMILARITY_SQL_TICKER_YEAR = """
SELECT ticker, doctype, source_file, page, chunk_index, chunk_text,
       1 - (embedding <=> %s::vector) AS similarity
FROM disclosure_chunks
WHERE embedding IS NOT NULL AND ticker = %s AND source_file LIKE %s
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

# Doctype-scoped variants - added 2026-07-24, inspired by FinAgentBench
# (arXiv:2508.14052)'s Stage-1 "document-type ranking before chunk-level
# ranking": a query about "Nghị quyết HĐQT" shouldn't have its top-k slots
# competed for by financial_statement boilerplate that happens to share
# surface vocabulary, the same class of problem year-scoping (above) was
# built to fix for temporal noise. Folded into RQ4's existing
# metadata-filtering-before-similarity-search citation (arXiv:2510.24402
# covers company+period; doctype is a third metadata field on the same
# `disclosure_chunks` row, same mechanism) rather than a new RQ - see
# docs/PROGRESS.md's 2026-07-24 entry.
#
# Deliberately NOT a hard filter with no fallback: TiniX-sourced tickers
# (the majority of the corpus, see docs/PROGRESS.md - 194/204 in-scope
# tickers) only ever have doctype='financial_statement' rows at all, and
# even for tickers with multiple doctypes, some real facts (e.g. board
# member names) can legitimately appear INSIDE a financial_statement's own
# governance notes, not just in a dedicated governance_report file. A
# classifier that's wrong or too narrow must not silently return zero
# chunks - see similarity_search()'s fallback-to-unfiltered retry below.
SIMILARITY_SQL_TICKER_DOCTYPE = """
SELECT ticker, doctype, source_file, page, chunk_index, chunk_text,
       1 - (embedding <=> %s::vector) AS similarity
FROM disclosure_chunks
WHERE embedding IS NOT NULL AND ticker = %s AND doctype = %s
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

SIMILARITY_SQL_TICKER_YEAR_DOCTYPE = """
SELECT ticker, doctype, source_file, page, chunk_index, chunk_text,
       1 - (embedding <=> %s::vector) AS similarity
FROM disclosure_chunks
WHERE embedding IS NOT NULL AND ticker = %s AND source_file LIKE %s AND doctype = %s
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

# ---------------------------------------------------------------------------
# Lexical (BM25-style) search - 2026-08-06, một nửa của hybrid_search().
#
# Vì sao cần: dense embedding giỏi ngữ nghĩa nhưng yếu với chuỗi PHẢI khớp
# nguyên văn - tên công ty kiểm toán ("PwC", "iCPA"), con số, mã số thuế,
# tên riêng. Đây đúng là lớp câu hỏi tài chính hay hỏi nhất. Kết hợp sparse
# (khớp từ đúng nghĩa đen) + dense (ngữ nghĩa) rồi hợp nhất bằng Reciprocal
# Rank Fusion là cách chuẩn trong tài liệu IR:
#   - RRF: Cormack, Clarke & Buettcher (SIGIR 2009) - "Reciprocal rank
#     fusion outperforms Condorcet and individual rank learning methods".
#   - Hybrid sparse+dense cho RAG: "Blended RAG" (arXiv:2404.07220) và
#     hướng dẫn hybrid search chính thức của pgvector.
#
# Chi tiết tiếng Việt: dùng config 'simple' (không stemming) - giữ nguyên
# dấu, lowercase, tách theo khoảng trắng. Không có stemmer tiếng Việt trong
# Postgres core, nhưng với truy vấn khớp-nguyên-văn (tên, số) thì 'simple'
# là đủ và dự đoán được. plainto_tsquery AND tất cả các từ -> chặt quá cho
# câu hỏi dài; websearch_to_tsquery (PG >= 11) xử lý tự nhiên hơn nhưng vẫn
# AND - nên hybrid_search() bên dưới truyền CẢ câu hỏi lẫn danh sách từ khóa
# OR (viết bằng ' | ' cho to_tsquery) và lấy union. ts_rank đủ tốt để xếp
# hạng trong-danh-sách (RRF chỉ cần THỨ HẠNG, không cần điểm tuyệt đối).
#
# Index: infra/pgvector/migrations/002_chunk_text_fts.sql thêm GIN index
# trên to_tsvector('simple', chunk_text). CHƯA có index thì các query dưới
# vẫn CHẠY ĐÚNG (seq scan) - chỉ chậm hơn; ticker-scoped (~vài nghìn dòng)
# thì seq scan vốn đã là hành vi của nhánh vector (SET enable_indexscan=off).
LEXICAL_SQL = """
SELECT ticker, doctype, source_file, page, chunk_index, chunk_text,
       ts_rank(to_tsvector('simple', chunk_text), q) AS lex_rank
FROM disclosure_chunks, websearch_to_tsquery('simple', %s) q
WHERE to_tsvector('simple', chunk_text) @@ q
ORDER BY lex_rank DESC
LIMIT %s;
"""

LEXICAL_SQL_TICKER = """
SELECT ticker, doctype, source_file, page, chunk_index, chunk_text,
       ts_rank(to_tsvector('simple', chunk_text), q) AS lex_rank
FROM disclosure_chunks, websearch_to_tsquery('simple', %s) q
WHERE ticker = %s AND to_tsvector('simple', chunk_text) @@ q
ORDER BY lex_rank DESC
LIMIT %s;
"""

LEXICAL_SQL_TICKER_YEAR = """
SELECT ticker, doctype, source_file, page, chunk_index, chunk_text,
       ts_rank(to_tsvector('simple', chunk_text), q) AS lex_rank
FROM disclosure_chunks, websearch_to_tsquery('simple', %s) q
WHERE ticker = %s AND source_file LIKE %s AND to_tsvector('simple', chunk_text) @@ q
ORDER BY lex_rank DESC
LIMIT %s;
"""


def get_connection():
    import psycopg2
    return psycopg2.connect(
        host=config.PGVECTOR_HOST, port=config.PGVECTOR_PORT,
        dbname=config.PGVECTOR_DB, user=config.PGVECTOR_USER,
        password=config.PGVECTOR_PASSWORD,
    )


def _vector_literal(vec) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def load_jsonl_records(path: Path) -> list:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _existing_chunk_state(conn) -> dict:
    """Returns {(pdf_document_id, page, chunk_index): (chunk_hash,
    embedding_version)} for every disclosure_chunks row WITH a non-null
    embedding - i.e. rows a previous upsert run already fully committed.

    Added 2026-07-23 (as _existing_embedded_keys, key-only) after a real
    re-run surfaced a genuine performance bug: text_splitter.py always
    re-chunks the FULL corpus, so after downloading just 12 new tickers,
    chunks.jsonl came back with all 656,918 chunks, old and new mixed
    together, and nothing skipped the expensive bge-m3 embedding step for
    chunks already committed in an earlier run.

    UPGRADED 2026-07-29 to also return chunk_hash/embedding_version, after a
    real gap was pointed out: the key-only version skipped a record purely
    because its KEY already had a non-null embedding, with NO check that
    the stored TEXT still matched -- so a corrected/restated disclosure
    page (same ticker/file/page/chunk_index, different chunk_text after
    text_splitter.py correctly re-chunked it) would rechunk correctly
    upstream but the DB would silently keep the OLD text and OLD embedding
    forever, because the row was never re-selected for embedding. Callers
    now compare BOTH chunk_hash (did the text change?) and
    embedding_version (did the embedding model/config change?) before
    trusting an existing row, not just key presence.

    RE-KEYED 2026-07-30 (Task #2) from (ticker, source_file, page,
    chunk_index) to (pdf_document_id, page, chunk_index) - matches the
    locked natural key (migration 010, UPSERT_SQL v5). ticker/source_file
    are still selected/returned nowhere here since they're no longer part
    of the comparison key; source_file->pdf_document_id resolution happens
    once, up front, via resolve_pdf_document_ids_or_raise()."""
    state = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pdf_document_id, page, chunk_index, chunk_hash, embedding_version "
            "FROM disclosure_chunks WHERE embedding IS NOT NULL"
        )
        for pdf_document_id, page, chunk_index, c_hash, e_version in cur.fetchall():
            state[(pdf_document_id, page, chunk_index)] = (c_hash, e_version)
    return state


def _record_chunk_hash(record: dict) -> str:
    """record["chunk_hash"] if text_splitter.py already computed it (every
    chunk written since 2026-07-29); otherwise compute it on the fly from
    chunk_text for backward compatibility with chunks.jsonl/precomputed
    files written by an older version of the pipeline (e.g.
    data/processed/chunks_embedded.jsonl.gz from the 2026-07-22 Colab run,
    which predates chunk_hash entirely)."""
    return record.get("chunk_hash") or _compute_chunk_hash(record["chunk_text"])


def _lineage_fields(record: dict, pdf_document_id) -> tuple:
    """(pdf_document_id, chunking_version, normalization_version) for one
    record, for the v5 UPSERT_SQL params.

    chunking_version/normalization_version come straight from the record if
    text_splitter.py stamped them (every chunk written since 2026-07-29);
    records from an older chunks.jsonl/precomputed file (pre-dating this
    field, e.g. the 2026-07-22 Colab run) fall back to None rather than a
    guessed value - an honest "unknown", same backward-compat stance as
    _record_chunk_hash().

    RE-SHAPED 2026-07-30 (Task #2): pdf_document_id is now passed IN
    (already resolved for every record up front by
    resolve_pdf_document_ids_or_raise(), which raises UnregisteredDocumentError
    before this function is ever called if any record's source_file has no
    pdf_documents row) rather than looked up here per-record - this function
    no longer touches the DB at all, and can never return a None
    pdf_document_id."""
    return (
        pdf_document_id,
        record.get("chunking_version"),
        record.get("normalization_version"),
    )


def _needs_embedding(record: dict, pdf_document_id, existing: dict, current_embedding_version: str) -> bool:
    """True if `record` must be (re-)embedded rather than reused as-is:
    genuinely new key, changed chunk content, or a different embedding
    model/version than what's currently stored.

    RE-KEYED 2026-07-30 (Task #2) to (pdf_document_id, page, chunk_index) -
    matches _existing_chunk_state()'s new key and the locked UNIQUE
    constraint (migration 010)."""
    key = (pdf_document_id, record["page"], record["chunk_index"])
    old = existing.get(key)
    if old is None:
        return True
    old_hash, old_version = old
    if old_hash != _record_chunk_hash(record):
        return True
    if old_version != current_embedding_version:
        return True
    return False


def upsert_records(records: list, batch_size: int = 32, skip_existing: bool = True) -> int:
    """Embeds records['chunk_text'] in batches and upserts into pgvector.
    Returns the number of rows actually embedded+written (NOT counting
    skipped-as-already-done records - see skip_existing below).

    Commits after EVERY batch (2026-07-21 fix) - the original version
    wrapped the entire loop in one `with conn:` block, which only commits
    once at the very end. On a 656,918-chunk run that takes hours, that
    meant a Ctrl+C, crash, or sleep at row 21,632 (as actually happened)
    committed ZERO rows despite the console showing continuous progress -
    all already-embedded work was silently discarded. Per-batch commits
    make this script safe to interrupt and resume: just re-run with the
    same input file, UPSERT_SQL's ON CONFLICT DO UPDATE makes already-done
    rows a harmless no-op re-write.

    skip_existing=True (default, 2026-07-23 fix, upgraded 2026-07-29): before
    embedding, queries disclosure_chunks for every (pdf_document_id, page,
    chunk_index) that already has a non-null embedding, ALONG WITH its
    stored chunk_hash/embedding_version, and drops a record from the batch
    loop ONLY if its key, chunk_hash, AND embedding_version all still
    match what's in the DB (see _needs_embedding()) - they never reach
    embed_texts(). This is what actually fixes the "runs from the start
    every time" problem while still catching real content changes: a
    corrected/restated page gets a different chunk_hash even if its key is
    unchanged, so it's correctly re-embedded instead of silently skipped
    with stale text+vector left in the DB. Pass skip_existing=False to
    force a full re-embed of everything regardless of hash/version match.

    FAILS FAST (2026-07-30, Task #2) via resolve_pdf_document_ids_or_raise()
    - every record's source_file is resolved to a pdf_document_id BEFORE any
    embedding or DB write happens; if even one file was never registered by
    scripts/register_pdf_documents.py, this raises UnregisteredDocumentError
    immediately (naming every offending file) instead of silently writing a
    NULL pdf_document_id (no longer possible anyway - disclosure_chunks.
    pdf_document_id is NOT NULL as of migration 010) or failing loudly only
    after burning GPU/API time embedding a batch that was going to fail on
    write regardless."""
    if not records:
        return 0
    conn = get_connection()
    n_written = 0
    n_skipped = 0
    try:
        doc_ids_by_file = resolve_pdf_document_ids_or_raise(conn, records)
        if skip_existing:
            existing = _existing_chunk_state(conn)
            before = len(records)
            records = [r for r in records
                       if _needs_embedding(r, doc_ids_by_file[r["source_file"]], existing, EMBEDDING_VERSION)]
            n_skipped = before - len(records)
            print(f"  {n_skipped}/{before} chunk(s) already embedded with matching content+version "
                  f"- skipping, {len(records)} left to embed")
        if not records:
            return 0
        with conn.cursor() as cur:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                texts = [r["chunk_text"] for r in batch]
                vectors = embed_texts(texts, batch_size=batch_size)
                for r, vec in zip(batch, vectors):
                    pdf_document_id, chunking_version, normalization_version = _lineage_fields(
                        r, doc_ids_by_file[r["source_file"]])
                    cur.execute(UPSERT_SQL, (
                        r["ticker"], r["doctype"], r["source_file"], r["page"],
                        r["chunk_index"], r["chunk_text"], r["char_count"],
                        _vector_literal(vec), _record_chunk_hash(r), EMBEDDING_VERSION,
                        pdf_document_id, chunking_version, normalization_version,
                    ))
                    n_written += 1
                conn.commit()
                print(f"  embedded+loaded {min(i + batch_size, len(records))}/{len(records)} (committed)")
    finally:
        conn.close()
    return n_written


def upsert_precomputed(records: list, batch_size: int = 32, skip_existing: bool = True) -> int:
    """Same DB-write behavior as upsert_records(), but for records that
    ALREADY have an "embedding" key (a list of 1024 floats) - never calls
    embed_texts()/loads the bge-m3 model at all. This is the local-machine
    half of the Colab split added 2026-07-23 (see
    scripts/embed_chunks_offline.py's module docstring for the full
    workflow and why the split exists - short version: Colab can't reach a
    Postgres container running on your Mac, so embedding happens on Colab's
    GPU and gets written to a plain JSONL file, and this function does only
    the DB write back on your own machine - fast, no GPU/model needed,
    light enough to run on a machine that couldn't handle the full
    chunk-and-embed pipeline itself).

    Same skip_existing semantics as upsert_records() (2026-07-29 upgrade):
    a record is only skipped if its key, chunk_hash, AND embedding_version
    all still match the DB - see _needs_embedding(). Records from an OLDER
    precomputed file that predates chunk_hash entirely (e.g. the
    2026-07-22 Colab run's chunks_embedded.jsonl.gz) fall back to computing
    chunk_hash from chunk_text on the fly (see _record_chunk_hash()), so
    this stays backward-compatible without needing to re-run Colab.

    FAILS FAST (2026-07-30, Task #2) via resolve_pdf_document_ids_or_raise()
    - same up-front, all-or-nothing resolution as upsert_records(), see its
    docstring for the full rationale."""
    if not records:
        return 0
    conn = get_connection()
    n_written = 0
    try:
        doc_ids_by_file = resolve_pdf_document_ids_or_raise(conn, records)
        if skip_existing:
            existing = _existing_chunk_state(conn)
            before = len(records)
            records = [r for r in records
                       if _needs_embedding(r, doc_ids_by_file[r["source_file"]], existing, EMBEDDING_VERSION)]
            print(f"  {before - len(records)}/{before} chunk(s) already embedded with matching content+version "
                  f"- skipping, {len(records)} left to write")
        if not records:
            return 0
        with conn.cursor() as cur:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                for r in batch:
                    pdf_document_id, chunking_version, normalization_version = _lineage_fields(
                        r, doc_ids_by_file[r["source_file"]])
                    cur.execute(UPSERT_SQL, (
                        r["ticker"], r["doctype"], r["source_file"], r["page"],
                        r["chunk_index"], r["chunk_text"], r["char_count"],
                        _vector_literal(r["embedding"]), _record_chunk_hash(r), EMBEDDING_VERSION,
                        pdf_document_id, chunking_version, normalization_version,
                    ))
                    n_written += 1
                conn.commit()
                print(f"  loaded {min(i + batch_size, len(records))}/{len(records)} (committed)")
    finally:
        conn.close()
    return n_written


def prune_orphaned_chunks(current_keys: set, dry_run: bool = False, max_delete_ratio: float = 0.10,
                           force_over_ratio: bool = False) -> int:
    """Deletes every row in disclosure_chunks whose (pdf_document_id, page,
    chunk_index) is NOT in current_keys.

    Added 2026-07-29: UPSERT_SQL's ON CONFLICT DO UPDATE keeps re-runs
    idempotent for rows that still exist, but nothing ever deleted a row
    whose source .txt page disappeared (PDF removed/renamed, a duplicate
    cleaned up, etc.) -- those rows stayed in disclosure_chunks forever,
    orphaned, still returned by similarity_search().

    RE-KEYED 2026-07-30 (Task #2) from (ticker, source_file, page,
    chunk_index) to (pdf_document_id, page, chunk_index), matching the
    locked natural key (migration 010). Callers must build current_keys
    using resolve_pdf_document_ids_or_raise() (or equivalent) against the
    SAME set of records passed to upsert_records()/upsert_precomputed() -
    see main() below for the reference implementation.

    current_keys should be the FULL set of (pdf_document_id, page,
    chunk_index) tuples from the CURRENT chunks.jsonl (i.e. the file
    src/retrieval/text_splitter.py just wrote, reflecting every .txt page
    presently in text_native/) -- NOT a partial/manually-filtered subset,
    and NOT a Colab-precomputed-embeddings file that may only cover some
    tickers. Calling this with anything less than the true full corpus's
    key set will delete rows that are actually still valid.

    max_delete_ratio (added 2026-07-29, safety guard raised as a real risk:
    running --prune-orphans by accident against a partial --input, e.g. a
    single-ticker file against a 200+-ticker DB, would otherwise silently
    delete every other ticker's chunks): refuses to delete more than this
    fraction of the CURRENT total DB row count in one call unless
    force_over_ratio=True. Raises RuntimeError instead of deleting when the
    computed orphan count exceeds the threshold - this is a hard stop, not
    a warning, because the whole point is to make a catastrophic
    wrong-input mistake loud instead of silent.

    dry_run=True reports what WOULD be deleted (and whether the ratio guard
    would trip) without deleting anything - use this first to sanity-check
    the count before actually pruning."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pdf_document_id, page, chunk_index FROM disclosure_chunks")
            db_keys = {(d, p, c) for d, p, c in cur.fetchall()}
        orphans = db_keys - current_keys
        ratio = (len(orphans) / len(db_keys)) if db_keys else 0.0
        if dry_run:
            over = ratio > max_delete_ratio
            print(f"  [dry-run] {len(orphans)} orphaned chunk(s) would be deleted "
                  f"({len(db_keys)} in DB, {len(current_keys)} in current chunks.jsonl, "
                  f"{ratio:.1%} of DB){' -- WOULD BE REFUSED by --max-delete-ratio' if over else ''}")
            return len(orphans)
        if not orphans:
            print("  0 orphaned chunks found - nothing to prune")
            return 0
        if ratio > max_delete_ratio and not force_over_ratio:
            raise RuntimeError(
                f"Refusing to delete {len(orphans)}/{len(db_keys)} chunks ({ratio:.1%} of the DB), "
                f"which exceeds max_delete_ratio={max_delete_ratio:.0%}. This usually means --input "
                f"is NOT the full current chunks.jsonl (e.g. a single-ticker or otherwise partial "
                f"file) - re-check --input, or pass --prune-force-ratio if this is genuinely expected "
                f"(e.g. a real mass deletion of delisted tickers)."
            )
        with conn.cursor() as cur:
            n_deleted = 0
            orphan_list = list(orphans)
            batch_size = 500
            for i in range(0, len(orphan_list), batch_size):
                batch = orphan_list[i:i + batch_size]
                cur.executemany(
                    "DELETE FROM disclosure_chunks "
                    "WHERE pdf_document_id = %s AND page = %s AND chunk_index = %s",
                    batch,
                )
                n_deleted += len(batch)
                conn.commit()
        print(f"  Pruned {n_deleted} orphaned chunk(s) from disclosure_chunks "
              f"({len(db_keys)} were in DB, {len(current_keys)} in current chunks.jsonl)")
        return n_deleted
    finally:
        conn.close()


def _run_similarity_query(cur, vec: str, top_k: int, ticker: str, year: int, doctype: str) -> list:
    """One SQL round-trip for the given filter combination - factored out of
    similarity_search() so the doctype fallback-retry (see below) can call
    it twice (with, then without, doctype) without duplicating the
    branch-selection logic."""
    if ticker and year and doctype:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        pattern = f"%_{year}_%"
        cur.execute(SIMILARITY_SQL_TICKER_YEAR_DOCTYPE, (vec, ticker, pattern, doctype, vec, top_k))
    elif ticker and year:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        pattern = f"%_{year}_%"
        cur.execute(SIMILARITY_SQL_TICKER_YEAR, (vec, ticker, pattern, vec, top_k))
    elif ticker and doctype:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        cur.execute(SIMILARITY_SQL_TICKER_DOCTYPE, (vec, ticker, doctype, vec, top_k))
    elif ticker:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        cur.execute(SIMILARITY_SQL_TICKER, (vec, ticker, vec, top_k))
    else:
        # doctype-only (no ticker) isn't a real call pattern in this project
        # yet (rag_pipeline.answer() always has a ticker by the time it
        # classifies doctype) - fall through to the plain unfiltered ANN
        # search (HNSW as of 2026-07-26, was IVFFlat before) rather than
        # adding a 4th untested/unused SQL variant.
        cur.execute(SIMILARITY_SQL, (vec, vec, top_k))
    cols = ["ticker", "doctype", "source_file", "page", "chunk_index", "chunk_text", "similarity"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def similarity_search(query: str, top_k: int = 5, ticker: str = None, year: int = None,
                       doctype: str = None) -> list:
    """Query-time helper for src/retrieval/rag_pipeline.py. Embeds `query`
    with the same model/normalization as ingestion and returns the top_k
    closest chunks by cosine similarity. If `ticker` is given, the filter is
    pushed into the SQL WHERE clause (SIMILARITY_SQL_TICKER) so LIMIT is
    applied AFTER narrowing to that company, not before - see the
    SIMILARITY_SQL_TICKER comment for why this matters at this corpus
    size (2026-07-19).

    If `year` is ALSO given, uses SIMILARITY_SQL_TICKER_YEAR instead -
    filters to that year's source files (via a `source_file LIKE %_YYYY_%`
    pattern) BEFORE ranking by similarity, so `top_k` is a true top-K within
    that year rather than a top-K over all years that then needs filtering
    after the fact (see SIMILARITY_SQL_TICKER_YEAR's comment - confirmed
    2026-07-19 that post-hoc filtering can still miss the right chunk when a
    company has many same-year chunks that outrank it globally without
    actually answering the question). Caller is responsible for falling
    back to a year=None call if this returns nothing (e.g. non-TiniX
    sources with no year in the filename won't match the LIKE pattern).

    If `doctype` is ALSO given (2026-07-24, FinAgentBench-inspired - see
    SIMILARITY_SQL_TICKER_DOCTYPE's comment), the search is further narrowed
    to that document type BEFORE ranking, same "filter before LIMIT, not
    after" principle as year. Unlike year, this filter self-heals: if the
    doctype-scoped query returns ZERO rows (e.g. the classified doctype
    doesn't actually exist for this ticker, or the real answer lives in a
    different document type than guessed), this function automatically
    retries the SAME query with `doctype` dropped and returns that instead -
    a wrong/narrow doctype guess degrades to the pre-existing (year-only or
    unfiltered) behavior rather than silently returning nothing. Callers
    don't need their own fallback logic for this, unlike the year case.

    Ticker-scoped calls also disable index scans for the duration of the
    query (`SET LOCAL enable_indexscan/enable_bitmapscan = off`), forcing an
    exact sequential scan instead of the ANN index (HNSW as of 2026-07-26,
    was IVFFlat before - see infra/pgvector/migrations/001_disclosure_chunks.sql
    for why the switch, and why it does NOT remove the need for this
    workaround). Without this, Postgres can satisfy `ORDER BY embedding <=>
    ...` using the index's own approximate traversal (with the original
    IVFFlat setup, a single-list probe covering ~180 of 18,186 rows) and
    apply the ticker filter to that tiny approximate candidate set instead
    of the ticker's real ~5,000 rows - confirmed live (2026-07-19): with the
    WHERE-clause fix alone, `similarity_search(..., ticker='PNJ',
    top_k=20)` returned only 1 row despite PNJ having 5,328 rows in the
    table. A single ticker's row count is small enough that an exact scan
    is fast (no ANN index needed); this only applies when `ticker` is set -
    unscoped queries still use the ANN index as intended. (pgvector >= 0.8.0's
    `hnsw.iterative_scan` could let ticker-scoped queries use the index
    again instead of an exact scan - not enabled here yet, needs the
    deployed pgvector version confirmed first, see the migration file.)"""
    from src.retrieval.embeddings import embed_query
    vec = _vector_literal(embed_query(query))
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            results = _run_similarity_query(cur, vec, top_k, ticker, year, doctype)
            if not results and doctype:
                results = _run_similarity_query(cur, vec, top_k, ticker, year, None)
            for r in results:
                r["year"] = extract_year(r["source_file"])
            return results
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# web_chunks - added 2026-09-07 (see infra/postgres/migrations/014_web_chunks.sql
# for the full rationale: a SEPARATE table for scraped press/web content,
# fed by scripts/web_news_ingest.py, kept structurally apart from
# disclosure_chunks so it can never accidentally leak into the RQ1-RQ7
# official-disclosure benchmark. Deliberately a SIMPLER surface than the
# disclosure_chunks functions above: no pdf_document_id FK resolution (a
# web article has no PDF/physical-document identity), no ticker-scoped
# exact-scan workaround (the web corpus is expected to be small - dozens to
# low-hundreds of articles, not 650k+ chunks - so the HNSW index's
# index-then-filter behavior is not a practical recall problem here the way
# it is for disclosure_chunks at scale), no lexical/hybrid fusion (dense-only
# is enough for this fallback-only use case; add hybrid later if the web
# corpus grows enough for exact-string-match gaps to actually show up).
WEB_UPSERT_SQL = """
INSERT INTO web_chunks
    (ticker, source_type, domain, source_url, title, published_date,
     scraped_at, chunk_index, chunk_text, char_count, embedding,
     chunk_hash, embedding_version)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
ON CONFLICT (source_url, chunk_index)
DO UPDATE SET chunk_text = EXCLUDED.chunk_text,
              char_count = EXCLUDED.char_count,
              embedding = EXCLUDED.embedding,
              ticker = EXCLUDED.ticker,
              title = EXCLUDED.title,
              scraped_at = EXCLUDED.scraped_at,
              chunk_hash = EXCLUDED.chunk_hash,
              embedding_version = EXCLUDED.embedding_version;
"""

WEB_SIMILARITY_SQL = """
SELECT ticker, domain, source_url, title, published_date, chunk_text,
       1 - (embedding <=> %s::vector) AS similarity
FROM web_chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

WEB_SIMILARITY_SQL_TICKER = """
SELECT ticker, domain, source_url, title, published_date, chunk_text,
       1 - (embedding <=> %s::vector) AS similarity
FROM web_chunks
WHERE embedding IS NOT NULL AND ticker = %s
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


def _existing_web_chunk_state(conn) -> dict:
    """Same idea as _existing_chunk_state() but keyed by (source_url,
    chunk_index) - web_chunks' natural key, no pdf_document_id involved."""
    state = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_url, chunk_index, chunk_hash, embedding_version "
            "FROM web_chunks WHERE embedding IS NOT NULL"
        )
        for source_url, chunk_index, c_hash, e_version in cur.fetchall():
            state[(source_url, chunk_index)] = (c_hash, e_version)
    return state


def upsert_web_records(records: list, batch_size: int = 32, skip_existing: bool = True) -> int:
    """Embeds records['chunk_text'] and upserts into web_chunks - same
    per-batch-commit / skip_existing-by-hash-and-version pattern as
    upsert_records(), simplified for web_chunks' schema (no pdf_document_id
    resolution step). `records` are the JSONL rows written by
    scripts/web_news_ingest.py's `chunk` mode."""
    if not records:
        return 0
    conn = get_connection()
    n_written = 0
    try:
        if skip_existing:
            existing = _existing_web_chunk_state(conn)
            before = len(records)
            records = [
                r for r in records
                if existing.get((r["source_url"], r["chunk_index"])) != (r.get("chunk_hash"), EMBEDDING_VERSION)
            ]
            print(f"  {before - len(records)}/{before} web chunk(s) already embedded with matching "
                  f"content+version - skipping, {len(records)} left to embed")
        if not records:
            return 0
        with conn.cursor() as cur:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                texts = [r["chunk_text"] for r in batch]
                vectors = embed_texts(texts, batch_size=batch_size)
                for r, vec in zip(batch, vectors):
                    cur.execute(WEB_UPSERT_SQL, (
                        r.get("ticker"), r.get("source_type", "news_web"), r["domain"],
                        r["source_url"], r.get("title"), r.get("published_date"),
                        r["scraped_at"], r["chunk_index"], r["chunk_text"], r["char_count"],
                        _vector_literal(vec), r.get("chunk_hash") or _compute_chunk_hash(r["chunk_text"]),
                        EMBEDDING_VERSION,
                    ))
                    n_written += 1
                conn.commit()
                print(f"  embedded+loaded {min(i + batch_size, len(records))}/{len(records)} (committed)")
    finally:
        conn.close()
    return n_written


def web_similarity_search(query: str, top_k: int = 5, ticker: str = None) -> list:
    """Query-time helper for src/retrieval/rag_pipeline.py's web fallback
    (see web_fallback_answer() there) - dense-only cosine search over
    web_chunks, deliberately scoped-down vs similarity_search()/
    hybrid_search() (see the module-level comment above WEB_UPSERT_SQL for
    why: small corpus, fallback-only use case)."""
    from src.retrieval.embeddings import embed_query
    vec = _vector_literal(embed_query(query))
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if ticker:
                cur.execute(WEB_SIMILARITY_SQL_TICKER, (vec, ticker, vec, top_k))
            else:
                cur.execute(WEB_SIMILARITY_SQL, (vec, vec, top_k))
            cols = ["ticker", "domain", "source_url", "title", "published_date", "chunk_text", "similarity"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _run_lexical_query(cur, query: str, top_k: int, ticker: str, year: int) -> list:
    """Nhánh sparse của hybrid_search() - xem comment ở LEXICAL_SQL.
    Không có biến thể doctype: nhánh lexical là lưới vét khớp-nguyên-văn,
    lọc hẹp thêm chỉ làm tăng nguy cơ 0 kết quả mà không thêm precision
    (RRF đã tự phạt những chunk chỉ khớp 1 nhánh)."""
    if ticker and year:
        cur.execute(LEXICAL_SQL_TICKER_YEAR, (query, ticker, f"%_{year}_%", top_k))
    elif ticker:
        cur.execute(LEXICAL_SQL_TICKER, (query, ticker, top_k))
    else:
        cur.execute(LEXICAL_SQL, (query, top_k))
    cols = ["ticker", "doctype", "source_file", "page", "chunk_index", "chunk_text", "lex_rank"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def lexical_search(query: str, top_k: int = 5, ticker: str = None,
                   year: int = None) -> list:
    """FTS-only retrieval for a known disclosure label.

    This avoids another embedding pass when RAG already has a broad semantic
    result but must include a specific statement line such as net revenue or
    profit after tax.  It is a retrieval helper, not a generator and it does
    not relax ticker/year filtering.
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                rows = _run_lexical_query(cur, query, top_k, ticker, year)
            for row in rows:
                row["year"] = extract_year(row["source_file"])
                row["similarity"] = 0.0
                row["retrieval"] = "lexical"
            return rows
        finally:
            conn.close()
    except Exception:
        return []


def hybrid_search(query: str, top_k: int = 5, ticker: str = None, year: int = None,
                  doctype: str = None) -> list:
    """Hybrid dense + sparse với Reciprocal Rank Fusion (Cormack et al.,
    SIGIR 2009; k=60 theo khuyến nghị gốc của paper). 2026-08-06 - trả lời
    yêu cầu "RAG local phải trả lời thật tốt vì không được call API ngoài
    (bảo mật dữ liệu khách hàng)": nâng chất lượng ở tầng TRUY XUẤT (không
    cần model mạnh hơn, không gửi gì ra ngoài máy).

    Trả về cùng shape với similarity_search() (mỗi chunk có "similarity" +
    "year"), thêm "retrieval" ("both" | "vector" | "lexical") và
    "rrf_score". Chunk chỉ do nhánh lexical tìm thấy có similarity=0.0
    (không biết cosine của nó mà không tốn thêm round-trip) - thứ tự trả về
    là thứ tự RRF, downstream muốn xếp lại theo similarity thì các chunk
    lexical-only tự chìm xuống, chấp nhận được vì việc chúng CÓ MẶT trong
    danh sách mới là cái dense-only làm mất.

    Fail-soft: nhánh lexical lỗi kiểu gì (thiếu quyền, câu query rỗng sau
    khi tách từ, v.v.) cũng rơi về đúng kết quả dense thuần - hybrid không
    bao giờ tệ hơn similarity_search()."""
    RRF_K = 60
    fetch_k = max(top_k * 3, 15)

    dense = similarity_search(query, top_k=fetch_k, ticker=ticker, year=year, doctype=doctype)

    sparse = []
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                sparse = _run_lexical_query(cur, query, fetch_k, ticker, year)
            for r in sparse:
                r["year"] = extract_year(r["source_file"])
        finally:
            conn.close()
    except Exception:
        sparse = []
    if not sparse:
        return dense[:top_k]

    def key(c):
        return (c["ticker"], c["source_file"], c["page"], c["chunk_index"])

    fused: dict = {}
    for rank, c in enumerate(dense, start=1):
        entry = fused.setdefault(key(c), {**c, "rrf_score": 0.0, "retrieval": "vector"})
        entry["rrf_score"] += 1.0 / (RRF_K + rank)
    for rank, c in enumerate(sparse, start=1):
        k_ = key(c)
        if k_ in fused:
            fused[k_]["rrf_score"] += 1.0 / (RRF_K + rank)
            fused[k_]["retrieval"] = "both"
        else:
            entry = {**c, "similarity": 0.0, "rrf_score": 1.0 / (RRF_K + rank),
                     "retrieval": "lexical"}
            entry.pop("lex_rank", None)
            fused[k_] = entry
    for v in fused.values():
        v.pop("lex_rank", None)
    return sorted(fused.values(), key=lambda c: -c["rrf_score"])[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="chunks.jsonl from src/retrieval/text_splitter.py")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-skip-existing", action="store_true",
                     help="Force re-embedding of every record even if already present in "
                          "disclosure_chunks with a non-null embedding (default: skip them). "
                          "Use this after changing the embedding model, or after an ingestion "
                          "fix (e.g. tinix_ingest_adapter.py's fix_concatenated_words()) that "
                          "changed existing chunk_text and needs a real re-embed, not a skip.")
    ap.add_argument("--precomputed", action="store_true",
                     help="Input records already have an 'embedding' field (from "
                          "scripts/embed_chunks_offline.py, e.g. run on Google Colab) - skip "
                          "loading bge-m3/embedding entirely, just write to pgvector. Fast, "
                          "no GPU or model download needed for this step.")
    ap.add_argument("--prune-orphans", action="store_true",
                     help="After upserting, DELETE any disclosure_chunks row whose "
                          "(ticker, source_file, page, chunk_index) is no longer present in "
                          "--input (e.g. a PDF/page was removed or renamed since the last run). "
                          "ONLY safe when --input is the FULL current chunks.jsonl written by "
                          "src/retrieval/text_splitter.py, not a manually filtered subset or a "
                          "partial --precomputed file that only covers some tickers - either of "
                          "those would delete rows that are still valid. Combine with "
                          "--prune-dry-run first to check the count before actually deleting.")
    ap.add_argument("--prune-dry-run", action="store_true",
                     help="With --prune-orphans, report how many rows WOULD be deleted without "
                          "deleting anything.")
    ap.add_argument("--max-delete-ratio", type=float, default=0.10,
                     help="With --prune-orphans, refuse to delete more than this fraction of the "
                          "DB's current row count in one call (default 0.10 = 10%%). Safety guard "
                          "against accidentally running --prune-orphans with a partial --input.")
    ap.add_argument("--prune-force-ratio", action="store_true",
                     help="Override --max-delete-ratio and delete even if the orphan count exceeds "
                          "it. Only use this if you deliberately expect a large deletion (e.g. "
                          "removing many delisted tickers at once).")
    ap.add_argument("--web", action="store_true",
                     help="Input is web_chunks.jsonl (from scripts/web_news_ingest.py's `chunk` "
                          "mode), not disclosure chunks.jsonl - embeds and upserts into the "
                          "SEPARATE web_chunks table instead of disclosure_chunks (see "
                          "infra/postgres/migrations/014_web_chunks.sql). Ignores "
                          "--precomputed/--prune-orphans (not supported for web_chunks yet - the "
                          "web corpus is small enough that this hasn't been needed).")
    args = ap.parse_args()

    records = load_jsonl_records(Path(args.input))
    if not records:
        print(f"No records found in {args.input}")
        return

    if args.web:
        print(f"Loaded {len(records)} web chunk record(s). Embedding and upserting into web_chunks...")
        n = upsert_web_records(records, batch_size=args.batch_size, skip_existing=not args.no_skip_existing)
        print(f"Done. {n} web chunk(s) embedded and upserted into web_chunks.")
        return

    if args.precomputed:
        print(f"Loaded {len(records)} chunk records with precomputed embeddings. Upserting into pgvector...")
        n = upsert_precomputed(records, batch_size=args.batch_size, skip_existing=not args.no_skip_existing)
        print(f"Done. {n} chunk(s) upserted into disclosure_chunks.")
    else:
        from src.retrieval.embeddings import EMBEDDING_DIM
        print(f"Loaded {len(records)} chunk records. Embedding ({EMBEDDING_DIM}-dim) and upserting into pgvector...")
        n = upsert_records(records, batch_size=args.batch_size, skip_existing=not args.no_skip_existing)
        print(f"Done. {n} chunk(s) embedded and upserted into disclosure_chunks.")

    if args.prune_orphans or args.prune_dry_run:
        # Re-resolve pdf_document_id per record (2026-07-30, Task #2) - the
        # cache in resolve_pdf_document_id() makes this cheap (one query per
        # DISTINCT source_file, already paid for above during upsert). Any
        # record whose file still isn't registered would already have
        # raised UnregisteredDocumentError during the upsert call above, so
        # this is not expected to raise here in practice - kept as a real
        # call (not a silent re-use of a partial cache) so this stays
        # correct even if --prune-orphans is ever invoked as a standalone
        # step against a --precomputed file in the future.
        conn = get_connection()
        try:
            doc_ids_by_file = resolve_pdf_document_ids_or_raise(conn, records)
        finally:
            conn.close()
        current_keys = {(doc_ids_by_file[r["source_file"]], r["page"], r["chunk_index"]) for r in records}
        print("Checking for orphaned chunks (present in disclosure_chunks but not in --input)...")
        # --prune-dry-run always wins if both flags are passed - safer to
        # under-delete on ambiguous input than to silently delete rows.
        try:
            prune_orphaned_chunks(current_keys, dry_run=args.prune_dry_run,
                                   max_delete_ratio=args.max_delete_ratio,
                                   force_over_ratio=args.prune_force_ratio)
        except RuntimeError as e:
            print(f"REFUSED: {e}")


if __name__ == "__main__":
    main()
