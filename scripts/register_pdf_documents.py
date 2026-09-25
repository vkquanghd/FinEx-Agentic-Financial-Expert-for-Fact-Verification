"""Registers every PDF in scripts/pdf_ingest_router.py's ingest_manifest.csv
into `pdf_documents` (see infra/postgres/migrations/006_pdf_documents.sql) -
the NEW step that runs between pdf_ingest_router.py and text_splitter.py,
so a PDF's document identity is ASSIGNED at ingest/registration time, not
GUESSED later per-chunk (see src/retrieval/vector_store.py's
resolve_pdf_document_id()'s docstring for the full before/after rationale).

Run order (2026-07-29 evening, Task #7 from the round-2 architecture
review):
    python3 scripts/pdf_ingest_router.py --input data/raw/pdf_disclosures --output data/processed
    python3 -m scripts.register_pdf_documents --manifest data/processed/ingest_manifest.csv   <- THIS SCRIPT, NEW
    python3 -m src.retrieval.text_splitter --input data/processed/text_native --output data/processed/chunks.jsonl
    python3 -m src.retrieval.vector_store --input data/processed/chunks.jsonl

Idempotent / safe to re-run: uses source_file as the natural key
(UNIQUE in pdf_documents). Unchanged files (same pdf_hash AND
extraction_version) are a no-op - the `WHERE ... IS DISTINCT FROM ...`
guard on the UPSERT's DO UPDATE means document_version only increments when
a file's actual content or extraction logic changed, not on every routine
re-run.

linked_disclosure_id (best-effort ticker+fiscal_year+doctype match against
the SEPARATE, ETL-sourced `disclosures` table) is resolved here, ONCE per
PDF document - this is the same heuristic the OLD resolve_disclosure_id()
used, just moved from "once per chunk, at upsert time" to "once per
document, at registration time", which is both cheaper and more honest
about what it actually is: a secondary, still-fuzzy cross-reference, not
the chunk's primary document identity (that's now source_file/pdf_hash,
exact by construction).

Legacy manifest support (added 2026-07-30, found running this for real
against a manifest predating the pdf_hash/ingest_version columns): a
manifest row missing pdf_hash gets a REAL SHA-256 computed from the actual
PDF file under --pdf-dir (default data/raw/pdf_disclosures, matching
pdf_ingest_router.py's own --input default) BEFORE any DB write is
attempted - see _resolve_pdf_hashes()/MissingPdfHashError. pdf_hash is
CHAR(64) NOT NULL in the DB (migrations/006_pdf_documents.sql), so unlike
ticker/doctype/extraction_version there is no honest NULL/empty fallback
for it.

Permanently-missing raw PDFs (added 2026-07-30, found running this for
real against a corpus where ~2890 of ~3231 original PDFs are no longer
present under --pdf-dir at all - confirmed not recoverable from any
on-disk backup, including data/raw/tinix_ocr's partial OCR-text archive,
which has text but never the original PDF bytes for any file): pass
--allow-missing-pdf-placeholder to register these anyway with
MISSING_SOURCE_PDF_HASH_SENTINEL instead of raising MissingPdfHashError,
so migration 010_lock_pdf_document_id.sql can proceed rather than staying
blocked indefinitely on documents whose source bytes are gone for good.
This is OFF by default - a normal run still fails loudly on missing files,
exactly as before. See _resolve_pdf_hashes()'s docstring for the full
honesty/transparency rationale (never a silent, unflagged guess) and how
this auto-upgrades to a real hash the moment a real PDF is later found and
dropped into --pdf-dir.

Orphans with NO manifest row at all (added 2026-07-30, found running the
above for real - it still registered 0 of the ~2890 missing files, because
register_all() only iterates --manifest's rows, and the manifest doesn't
list these files AT ALL - data/processed/ingest_manifest.csv is rewritten
from scratch every pdf_ingest_router.py run, listing only what's CURRENTLY
under --input, which now covers far fewer files than the corpus originally
had): pass --register-orphans-from-chunks to ALSO register every
source_file present in disclosure_chunks but absent from both --manifest
and pdf_documents, deriving ticker/doctype/n_pages_total straight from
disclosure_chunks (already NOT NULL there) instead of a manifest row - see
register_orphans_from_chunks().
"""
import argparse
import csv
from pathlib import Path

from sqlalchemy import text

from src.core.db import get_session
from src.retrieval.vector_store import extract_year
from scripts.pdf_ingest_router import _file_hash

# Default matches pdf_ingest_router.py's own --input default - the raw PDF
# folder register_all() falls back to hashing FROM when a legacy manifest
# row has no pdf_hash column at all. See _resolve_pdf_hashes()'s docstring:
# pdf_documents.pdf_hash is CHAR(64) NOT NULL (migrations/006_pdf_documents.sql)
# - found running this for real (2026-07-30) that None is not an acceptable
# fallback here, unlike ticker/doctype/extraction_version which ARE nullable/
# empty-string-able. A real hash must exist; it's either already in the
# manifest, or must be computed fresh from the actual file bytes.
DEFAULT_PDF_DIR = Path("data/raw/pdf_disclosures")

UPSERT_PDF_DOCUMENT = text("""
    INSERT INTO pdf_documents
        (source_file, pdf_hash, ticker, doctype, fiscal_year, n_pages_total, extraction_version)
    VALUES
        (:source_file, :pdf_hash, :ticker, :doctype, :fiscal_year, :n_pages_total, :extraction_version)
    ON CONFLICT (source_file) DO UPDATE SET
        pdf_hash = EXCLUDED.pdf_hash,
        ticker = EXCLUDED.ticker,
        doctype = EXCLUDED.doctype,
        fiscal_year = EXCLUDED.fiscal_year,
        n_pages_total = EXCLUDED.n_pages_total,
        extraction_version = EXCLUDED.extraction_version,
        document_version = pdf_documents.document_version + 1,
        is_active = TRUE,
        updated_at = now()
    WHERE pdf_documents.pdf_hash IS DISTINCT FROM EXCLUDED.pdf_hash
       OR pdf_documents.extraction_version IS DISTINCT FROM EXCLUDED.extraction_version
    RETURNING id, (xmax = 0) AS inserted
""")

SELECT_PDF_DOCUMENT_ID = text("SELECT id FROM pdf_documents WHERE source_file = :source_file")

# Same best-effort strategy as the old vector_store.resolve_disclosure_id() -
# moved here since it's now a per-DOCUMENT (not per-chunk) concern. See this
# module's docstring for why that move is a real improvement, not just a
# relocation.
FIND_LINKED_DISCLOSURE = text("""
    SELECT id FROM disclosures WHERE ticker = :ticker AND fiscal_year = :fiscal_year
    ORDER BY (period = 'YEAR') DESC, disclosure_date DESC NULLS LAST
""")

UPDATE_LINKED_DISCLOSURE = text(
    "UPDATE pdf_documents SET linked_disclosure_id = :linked_disclosure_id WHERE id = :id"
)

# Added 2026-07-30 alongside pdf_ingest_router.py's deletion detection (found
# while designing the PDF lifecycle integration test): a PDF removed from
# --input gets its stale text_native pages deleted there and is listed in
# deleted_files.csv - this soft-deletes (is_active=FALSE) the matching
# pdf_documents row rather than deleting it, consistent with the "simple"
# versioning policy already documented in 001_init_schema.sql (only the
# active document state is retained, but a row's EXISTENCE is still kept for
# lineage/audit - a hard DELETE would either cascade-orphan or block on
# disclosure_chunks' FK depending on how deletion is actually carried out,
# and would lose the audit trail of "this ticker/year WAS backed by a PDF
# at some point"). Chunk cleanup itself is unaffected by this - it already
# happens via vector_store.py's existing --prune-orphans step once
# text_splitter.py's next run naturally excludes the deleted file's (now
# also-deleted) text_native pages from chunks.jsonl.
UPDATE_DEACTIVATE = text(
    "UPDATE pdf_documents SET is_active = FALSE, updated_at = now() "
    "WHERE source_file = :source_file AND is_active = TRUE"
)


def load_manifest_by_file(manifest_path: Path) -> dict:
    """Collapses ingest_manifest.csv (one row PER PAGE) down to one entry
    PER FILE - pdf_hash/ingest_version/ticker/doctype/n_pages_total repeat
    identically across every page row for the same file, so the first row
    seen for a file is representative."""
    by_file = {}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["file"]
            if fname not in by_file:
                by_file[fname] = row
    return by_file


def resolve_linked_disclosure(session, ticker: str, source_file: str, doctype: str):
    """Best-effort, nullable - see this module's docstring. Not memoized
    across calls the way the old per-chunk resolver was, since this now
    runs once per document (hundreds of PDFs, not hundreds of thousands of
    chunks) - not worth the added state for this call volume."""
    fiscal_year = extract_year(source_file)
    if fiscal_year is None:
        return None
    rows = session.execute(FIND_LINKED_DISCLOSURE, {"ticker": ticker, "fiscal_year": fiscal_year}).fetchall()
    return rows[0][0] if rows else None


class MissingPdfHashError(Exception):
    """Raised by _resolve_pdf_hashes() BEFORE any DB write happens (same
    fail-fast-on-the-whole-batch pattern as vector_store.py's
    UnregisteredDocumentError / resolve_pdf_document_ids_or_raise(), added
    2026-07-30 Task #16) - one clear error listing EVERY file that couldn't
    be resolved, instead of crashing partway through register_all() with a
    half-registered pdf_documents table."""


# Sentinel used ONLY when the caller explicitly opts into
# allow_missing_as_placeholder=True (added 2026-07-30) for a file whose raw
# PDF bytes are confirmed gone for good (checked: not under --pdf-dir, not
# recoverable from data/raw/tinix_ocr's OCR-text archive, which never has
# the original PDF - only extracted text). Deliberately NOT a real SHA-256 -
# all-zeros can never collide with a genuine hash (real SHA-256 of any
# non-empty PDF is, for all practical purposes, never "0"*64). Multiple
# different missing-PDF files sharing this identical sentinel is FINE for
# correctness: pdf_documents rows are keyed by source_file (UNIQUE), never
# by pdf_hash - the hash's only other job (change-detection on the NEXT
# register_all() run for the SAME source_file) is moot for a file with no
# real bytes to ever re-hash from, and is honestly reported, not hidden -
# see data_quality_report.py's check_placeholder_pdf_hash().
MISSING_SOURCE_PDF_HASH_SENTINEL = "0" * 64


def _resolve_pdf_hashes(by_file: dict, pdf_dir: Path, allow_missing_as_placeholder: bool = False) -> dict:
    """Returns {fname: pdf_hash} for every file in by_file.

    pdf_documents.pdf_hash is CHAR(64) NOT NULL (migrations/006_
    pdf_documents.sql) - found running this for real 2026-07-30 against a
    manifest predating the pdf_hash column: unlike ticker/doctype/
    extraction_version (nullable or empty-string-able), pdf_hash has no
    honest "unknown" value the DB will accept. A manifest row that already
    has pdf_hash is trusted as-is (no re-hashing an already-registered
    file's row). A row missing it gets a REAL SHA-256 computed straight
    from the actual PDF bytes under pdf_dir, reusing pdf_ingest_router.py's
    own _file_hash() - the exact same function that script will use to
    compute this same file's hash on its NEXT real run, so an honestly-
    computed hash here will correctly match then (a placeholder/dummy value
    would not, and would also collide across every legacy file, silently
    breaking the "same pdf_hash means unchanged" convention on top of the
    NOT NULL crash).

    By default (allow_missing_as_placeholder=False), raises
    MissingPdfHashError (naming every affected file) if pdf_dir doesn't
    contain a file already missing pdf_hash - fails before any DB write,
    not partway through. If allow_missing_as_placeholder=True, those files
    get MISSING_SOURCE_PDF_HASH_SENTINEL instead of raising - an explicit,
    opt-in acknowledgment that the real bytes are gone, not a silent
    default. Either way, a file that HAS its real PDF on disk (now, or on
    some future run once found/restored) always gets its real hash - this
    check re-runs fresh every call, it's never cached against a prior
    placeholder registration."""
    resolved = {}
    missing_on_disk = []
    for fname, row in by_file.items():
        existing_hash = row.get("pdf_hash") or None
        if existing_hash:
            resolved[fname] = existing_hash
            continue
        pdf_path = pdf_dir / fname
        if not pdf_path.exists():
            missing_on_disk.append(fname)
            if allow_missing_as_placeholder:
                resolved[fname] = MISSING_SOURCE_PDF_HASH_SENTINEL
            continue
        resolved[fname] = _file_hash(pdf_path)

    if missing_on_disk and not allow_missing_as_placeholder:
        raise MissingPdfHashError(
            f"{len(missing_on_disk)} manifest row(s) have no pdf_hash AND no matching "
            f"file under {pdf_dir} to compute one from - cannot register these without "
            f"either a pdf_hash in the manifest or the actual PDF present on disk. "
            f"Pass --pdf-dir pointing at wherever these PDFs actually live, re-run "
            f"pdf_ingest_router.py so the manifest gets a real pdf_hash, or pass "
            f"--allow-missing-pdf-placeholder to register these anyway with a documented "
            f"placeholder hash (see MISSING_SOURCE_PDF_HASH_SENTINEL). Affected files: "
            + ", ".join(missing_on_disk[:20])
            + (f", ... and {len(missing_on_disk) - 20} more" if len(missing_on_disk) > 20 else "")
        )
    return resolved


def register_all(manifest_path: Path, pdf_dir: Path = DEFAULT_PDF_DIR,
                  allow_missing_as_placeholder: bool = False) -> dict:
    """Returns {"n_files", "n_new", "n_updated", "n_unchanged", "n_linked",
    "n_placeholder_hash"}."""
    by_file = load_manifest_by_file(manifest_path)
    # resolves the WHOLE batch before any DB write
    pdf_hashes = _resolve_pdf_hashes(by_file, pdf_dir, allow_missing_as_placeholder)
    n_new = n_updated = n_unchanged = n_linked = 0
    n_placeholder_hash = sum(1 for h in pdf_hashes.values() if h == MISSING_SOURCE_PDF_HASH_SENTINEL)

    with get_session() as session:
        for fname, row in by_file.items():
            # pdf_documents.source_file MUST be the file STEM (no ".pdf"), NOT
            # fname/row["file"] (which has it) - fixed 2026-07-30, found running
            # this for real: pdf_ingest_router.py names text_native/ pages after
            # path.stem (see process_pdf()'s out_file), so text_splitter.py's
            # parse_filename() reconstructs disclosure_chunks.source_file WITHOUT
            # the extension too. Storing fname (WITH ".pdf") here made EVERY
            # pdf_documents.source_file permanently unmatchable against
            # disclosure_chunks.source_file - not just for files missing from
            # --pdf-dir, but for ALL 341 files this script had already
            # successfully registered (backfill_pdf_document_id.py correctly
            # backfilled 0 rows for exactly this reason, confirmed against the
            # real DB). stem is used ONLY for the DB column; fname (with
            # extension) is still used for the actual on-disk path lookup in
            # _resolve_pdf_hashes() and for extract_year()/deactivate_deleted(),
            # both of which work identically with or without the extension.
            source_file = Path(fname).stem
            params = {
                "source_file": source_file,
                "pdf_hash": pdf_hashes[fname],
                "ticker": row["ticker"],
                "doctype": row["doctype"],
                "fiscal_year": extract_year(fname),
                "n_pages_total": int(row["n_pages_total"]) if row.get("n_pages_total") else None,
                # .get(), not row["ingest_version"] (fixed 2026-07-30, found running this
                # for real against a manifest predating the 2026-07-29 ingest_version
                # column) - unlike pdf_hash, extraction_version IS nullable-as-empty-
                # string safely: a legacy row with extraction_version="" will always look
                # "changed" against a fresh non-empty value the next real
                # pdf_ingest_router.py run computes (harmless - one honest re-registration).
                "extraction_version": row.get("ingest_version") or "",
            }
            result = session.execute(UPSERT_PDF_DOCUMENT, params).fetchone()
            if result is None:
                # WHERE guard on DO UPDATE skipped the write entirely - genuinely unchanged.
                n_unchanged += 1
                doc_id = session.execute(SELECT_PDF_DOCUMENT_ID, {"source_file": source_file}).fetchone()[0]
            else:
                doc_id, inserted = result
                if inserted:
                    n_new += 1
                else:
                    n_updated += 1

            linked_id = resolve_linked_disclosure(session, params["ticker"], fname, params["doctype"])
            if linked_id is not None:
                session.execute(UPDATE_LINKED_DISCLOSURE, {"linked_disclosure_id": linked_id, "id": doc_id})
                n_linked += 1

    return {"n_files": len(by_file), "n_new": n_new, "n_updated": n_updated,
            "n_unchanged": n_unchanged, "n_linked": n_linked,
            "n_placeholder_hash": n_placeholder_hash}


# --- Registering orphans that have NO manifest row at all --------------------
# Added 2026-07-30, found running register_all(allow_missing_as_placeholder=True)
# for real: it still registered 0 of the ~2890 truly-missing files, because
# register_all() only ever iterates ingest_manifest.csv's rows - and the
# manifest ITSELF no longer lists these files at all (pdf_ingest_router.py
# REWRITES data/processed/ingest_manifest.csv from scratch every run, listing
# only whatever's CURRENTLY under --input - which now has only 358 of the
# corpus's original ~3231 PDFs). There is no manifest row to read ticker/
# doctype/n_pages_total from for these files - but disclosure_chunks itself
# already has ticker/doctype/page (NOT NULL columns, populated when these
# files were originally chunked/embedded, before their PDFs went missing) -
# so pull metadata straight from there instead of requiring a manifest row.
SELECT_ORPHAN_AMBIGUOUS_SOURCE_FILES = text("""
    SELECT dc.source_file
    FROM disclosure_chunks dc
    LEFT JOIN pdf_documents pd ON dc.source_file = pd.source_file
    WHERE dc.pdf_document_id IS NULL AND pd.id IS NULL
    GROUP BY dc.source_file
    HAVING COUNT(DISTINCT dc.ticker) > 1 OR COUNT(DISTINCT dc.doctype) > 1
""")

SELECT_ORPHAN_CHUNK_METADATA = text("""
    SELECT dc.source_file, MIN(dc.ticker) AS ticker, MIN(dc.doctype) AS doctype,
           MAX(dc.page) AS n_pages_total
    FROM disclosure_chunks dc
    LEFT JOIN pdf_documents pd ON dc.source_file = pd.source_file
    WHERE dc.pdf_document_id IS NULL AND pd.id IS NULL
    GROUP BY dc.source_file
""")


class AmbiguousOrphanChunkError(Exception):
    """Raised by register_orphans_from_chunks() if a source_file's own
    chunks disagree on ticker or doctype - should never happen in practice
    (pdf_ingest_router.py assigns both ONCE per file, at ingest time, so
    every chunk from the same file should agree), but checked explicitly
    rather than assumed, matching this project's "never silently guess,
    never hide a disagreement" convention (see compare_t_learner_vs_dml.py's
    disagreement_warning for the same philosophy applied elsewhere). MIN()
    picking an arbitrary one of several disagreeing values would be a
    silent, undetectable wrong guess - refusing and naming the files is the
    honest choice."""
    def __init__(self, source_files):
        self.source_files = sorted(source_files)
        super().__init__(
            f"{len(self.source_files)} orphaned source_file(s) have INCONSISTENT ticker/doctype "
            f"across their own disclosure_chunks rows - refusing to guess which is correct. "
            f"Investigate manually: {self.source_files[:10]}{'...' if len(self.source_files) > 10 else ''}"
        )


def register_orphans_from_chunks(pdf_dir: Path = DEFAULT_PDF_DIR,
                                  allow_missing_as_placeholder: bool = False) -> dict:
    """Registers a pdf_documents row for every source_file present in
    disclosure_chunks (already chunked/embedded) but absent from BOTH
    ingest_manifest.csv AND pdf_documents - see the module-level comment
    above for why register_all() alone cannot reach these files.

    ticker/doctype/n_pages_total come from disclosure_chunks itself (MIN/MAX
    across that file's chunks - safe only after confirming, via
    SELECT_ORPHAN_AMBIGUOUS_SOURCE_FILES, that ticker/doctype are IDENTICAL
    across all of a file's chunks; raises AmbiguousOrphanChunkError instead
    of guessing if not). pdf_hash: tries a real hash from
    pdf_dir/<source_file>.pdf first (in case the file target actually exists
    there under a name the manifest just never listed); falls back to
    MISSING_SOURCE_PDF_HASH_SENTINEL only if allow_missing_as_placeholder,
    exactly like register_all()'s _resolve_pdf_hashes(). extraction_version
    is "" (unknown - this file never had a manifest row at all).

    Returns {"n_orphans_found", "n_registered", "n_placeholder_hash",
    "n_linked"}."""
    with get_session() as session:
        ambiguous = {r[0] for r in session.execute(SELECT_ORPHAN_AMBIGUOUS_SOURCE_FILES).fetchall()}
        if ambiguous:
            raise AmbiguousOrphanChunkError(ambiguous)

        rows = session.execute(SELECT_ORPHAN_CHUNK_METADATA).fetchall()

        # Resolve the WHOLE batch's pdf_hash before any DB write - same
        # discipline as _resolve_pdf_hashes().
        pdf_hashes = {}
        missing_on_disk = []
        for source_file, ticker, doctype, n_pages_total in rows:
            pdf_path = pdf_dir / f"{source_file}.pdf"
            if pdf_path.exists():
                pdf_hashes[source_file] = _file_hash(pdf_path)
            elif allow_missing_as_placeholder:
                pdf_hashes[source_file] = MISSING_SOURCE_PDF_HASH_SENTINEL
            else:
                missing_on_disk.append(source_file)

        if missing_on_disk:
            raise MissingPdfHashError(
                f"{len(missing_on_disk)} orphaned source_file(s) (no manifest row, no pdf_documents "
                f"row) also have no matching file under {pdf_dir} - pass "
                f"allow_missing_as_placeholder=True (--allow-missing-pdf-placeholder) to register "
                f"these anyway. Affected: "
                + ", ".join(sorted(missing_on_disk)[:20])
                + (f", ... and {len(missing_on_disk) - 20} more" if len(missing_on_disk) > 20 else "")
            )

        n_registered = n_placeholder_hash = n_linked = 0
        for source_file, ticker, doctype, n_pages_total in rows:
            pdf_hash = pdf_hashes[source_file]
            if pdf_hash == MISSING_SOURCE_PDF_HASH_SENTINEL:
                n_placeholder_hash += 1
            params = {
                "source_file": source_file,
                "pdf_hash": pdf_hash,
                "ticker": ticker,
                "doctype": doctype,
                "fiscal_year": extract_year(source_file),
                "n_pages_total": n_pages_total,
                "extraction_version": "",
            }
            result = session.execute(UPSERT_PDF_DOCUMENT, params).fetchone()
            if result is not None:
                n_registered += 1
                doc_id, _inserted = result
                linked_id = resolve_linked_disclosure(session, ticker, source_file, doctype)
                if linked_id is not None:
                    session.execute(UPDATE_LINKED_DISCLOSURE, {"linked_disclosure_id": linked_id, "id": doc_id})
                    n_linked += 1

    return {"n_orphans_found": len(rows), "n_registered": n_registered,
            "n_placeholder_hash": n_placeholder_hash, "n_linked": n_linked}


def deactivate_deleted(deleted_manifest_path: Path) -> int:
    """Reads deleted_files.csv (written by pdf_ingest_router.py's deletion
    detection, if any PDF was removed from --input since the last run) and
    soft-deletes (is_active=FALSE) the matching pdf_documents row for each.
    Returns the number of rows actually flipped (0 if the file doesn't exist,
    or every listed file was already inactive - idempotent/safe to re-run)."""
    if not deleted_manifest_path.exists():
        return 0
    n_deactivated = 0
    with get_session() as session:
        with deleted_manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # Path(...).stem, not row["file"] directly - must match
                # register_all()'s source_file convention (stem, no ".pdf"),
                # fixed alongside the same 2026-07-30 discovery.
                result = session.execute(UPDATE_DEACTIVATE, {"source_file": Path(row["file"]).stem})
                if result.rowcount:
                    n_deactivated += 1
    return n_deactivated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/ingest_manifest.csv"),
                     help="ingest_manifest.csv from scripts/pdf_ingest_router.py")
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR,
                     help="Raw PDF folder (pdf_ingest_router.py's --input) - only read from when a "
                          "legacy manifest row has no pdf_hash column, to compute a real one.")
    ap.add_argument("--allow-missing-pdf-placeholder", action="store_true",
                     help="Register files whose raw PDF is missing from --pdf-dir anyway, using a "
                          "documented placeholder pdf_hash (MISSING_SOURCE_PDF_HASH_SENTINEL) instead "
                          "of raising MissingPdfHashError. Use only after confirming the file is truly "
                          "unrecoverable - a real hash is always preferred and always used automatically "
                          "if the real PDF is later found and placed under --pdf-dir.")
    ap.add_argument("--register-orphans-from-chunks", action="store_true",
                     help="ALSO register source_files that exist in disclosure_chunks (already "
                          "embedded) but have NO row in --manifest at all - happens when the manifest "
                          "was regenerated against a smaller --input than what was originally "
                          "ingested. Derives ticker/doctype/n_pages_total from disclosure_chunks "
                          "itself. Combine with --allow-missing-pdf-placeholder if those files' raw "
                          "PDFs are also gone.")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Not found: {args.manifest} - run scripts/pdf_ingest_router.py first.")

    summary = register_all(args.manifest, args.pdf_dir, args.allow_missing_pdf_placeholder)
    print(f"pdf_documents: {summary['n_files']} distinct file(s) in manifest - "
          f"{summary['n_new']} new, {summary['n_updated']} updated (content/version changed), "
          f"{summary['n_unchanged']} unchanged (no-op). "
          f"{summary['n_linked']}/{summary['n_files']} linked to a `disclosures` row (best-effort).")
    if summary["n_placeholder_hash"]:
        print(f"WARNING: {summary['n_placeholder_hash']} file(s) registered with a PLACEHOLDER pdf_hash "
              f"(MISSING_SOURCE_PDF_HASH_SENTINEL) - their real PDF is missing from {args.pdf_dir}. "
              f"Tracked in data_quality_report.py's check_placeholder_pdf_hash(). Will auto-upgrade to "
              f"a real hash if the actual PDF is ever found and placed there, then this script re-run.")

    if args.register_orphans_from_chunks:
        orphan_summary = register_orphans_from_chunks(args.pdf_dir, args.allow_missing_pdf_placeholder)
        print(f"pdf_documents (from disclosure_chunks, no manifest row): "
              f"{orphan_summary['n_orphans_found']} orphaned source_file(s) found - "
              f"{orphan_summary['n_registered']} registered, "
              f"{orphan_summary['n_linked']}/{orphan_summary['n_registered']} linked to a `disclosures` row.")
        if orphan_summary["n_placeholder_hash"]:
            print(f"WARNING: {orphan_summary['n_placeholder_hash']} of those registered with a "
                  f"PLACEHOLDER pdf_hash - their raw PDF is missing from {args.pdf_dir}.")

    deleted_manifest_path = args.manifest.parent / "deleted_files.csv"
    n_deactivated = deactivate_deleted(deleted_manifest_path)
    if n_deactivated:
        print(f"pdf_documents: {n_deactivated} row(s) soft-deleted (is_active=FALSE) - "
              f"source file(s) no longer present in the last pdf_ingest_router.py run "
              f"(see {deleted_manifest_path}).")


if __name__ == "__main__":
    main()
