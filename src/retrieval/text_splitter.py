#!/usr/bin/env python3
"""
Chunking / text splitting for notes-to-FS and annual report text.

Turns the text-native pages produced by scripts/pdf_ingest_router.py into
retrieval-sized chunks, written to a JSONL file that src/retrieval/vector_store.py
then embeds (via src/retrieval/embeddings.py) and loads into pgvector.

Chunking strategy, and why it's NOT paragraph-based: `pdftotext -raw` output
from real Vietstock PDFs (checked against the TCH annual report sample) has
NO blank-line paragraph breaks at all -- it just wraps at the PDF's physical
line width (~40-45 chars/line here, because the source is a multi-column
magazine-style layout). Splitting on blank lines would produce exactly one
giant "paragraph" per page. Instead: join all lines in a page into one
continuous string (single newlines here are line-wrap artifacts, not real
breaks), then slide a fixed character window over it, breaking at the
nearest whitespace so words aren't cut mid-token, with overlap between
consecutive chunks so a sentence spanning a chunk boundary isn't lost from
both sides.

This is a first-pass strategy, not a final one -- a heading-aware version
(short ALL-CAPS lines like "RỦI RO VỀ LUẬT PHÁP" look like real section
headers in the samples checked) is a reasonable improvement once more real
pages are available to validate the heuristic against.

Each output page keeps its own chunks separate (no cross-page merging), so
ticker/doctype/page metadata on every chunk stays exact.

Input filename pattern (produced by pdf_ingest_router.py):
    <TICKER>__<doctype>__<original_stem>__p<page>.txt

Run:
  python3 -m src.retrieval.text_splitter --input data/processed/text_native --output data/processed/chunks.jsonl

Incremental runs (2026-07-29, upgraded same day after a real gap was found):
safe to re-run any time you drop new PDFs' text pages into --input. A file
is only re-chunked if it's new, or its CONTENT actually changed, or the
chunking config itself changed -- tracked in chunk_manifest.csv next to
--output, same pattern as scripts/pdf_ingest_router.py's ingest_manifest.csv.
Chunks for unchanged files are carried forward as-is from the previous
chunks.jsonl instead of re-running chunk_text() on them.

Change detection is by SHA-256 content hash, not mtime/size -- a file
copied/re-synced (e.g. iCloud) with the same bytes but a new mtime must
NOT be treated as changed, and this also makes the manifest robust to
timestamp-losing copies. Hash is computed once per file per run (fast,
pure I/O, no chunking cost) even for files that turn out unchanged.

The manifest also stores CHUNKING_VERSION, a string derived automatically
from CHUNK_SIZE/CHUNK_OVERLAP/MIN_CHUNK_SIZE below. If you edit any of
those three constants, CHUNKING_VERSION changes and EVERY cached entry is
treated as stale on the next run -- no need to remember --force for that
case. --force still exists for forcing a full re-chunk when the config
didn't change but you still want one (e.g. after a chunk_text() bugfix).
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

CHUNK_SIZE = 800     # target characters per chunk
CHUNK_OVERLAP = 120  # characters of overlap between consecutive chunks
MIN_CHUNK_SIZE = 80  # chunks shorter than this get merged into the previous one instead of standing alone

# Bump the trailing "_v1" by hand if chunk_text()'s ALGORITHM changes in a
# way that doesn't touch the three constants above (e.g. the whitespace-
# break logic itself) -- that's the one case CHUNKING_VERSION can't detect
# automatically since it's only derived from the constants, not the code.
CHUNKING_VERSION = f"sliding_{CHUNK_SIZE}_{CHUNK_OVERLAP}_min{MIN_CHUNK_SIZE}_v1"

# Separate from CHUNKING_VERSION (2026-07-29, review-prompted): normalize_page_text()
# can change independently of the sliding-window constants above (e.g. a fix to
# how line-wrap/hyphenation/page-header stripping works) and would silently
# change every chunk's content without CHUNKING_VERSION noticing, since that
# constant only reflects CHUNK_SIZE/CHUNK_OVERLAP/MIN_CHUNK_SIZE. There is no
# constant to derive this from automatically (normalize_page_text() is pure
# regex, not parameterized) -- bump this BY HAND whenever that function's
# logic changes, same as CHUNKING_VERSION's own "_v1" suffix.
NORMALIZATION_VERSION = "collapse_linewrap_v1"


def _file_hash(path: Path) -> str:
    """SHA-256 of file bytes, streamed so large files don't need to fit in
    RAM at once (matches the block-read pattern used elsewhere in this
    project, e.g. any content-hash helper in scripts/)."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def chunk_hash(text: str) -> str:
    """SHA-256 of a chunk's FINAL text (after MIN_CHUNK_SIZE merging), used
    downstream by src/retrieval/vector_store.py to detect content changes
    that a stable (ticker, source_file, page, chunk_index) key alone can't
    see -- e.g. a page whose content_hash changed (correction/restatement)
    but whose chunk COUNT and boundaries happen to stay the same still gets
    a different chunk_hash per chunk, since the underlying text changed.

    Added 2026-07-29 after a real gap was pointed out: vector_store.py's
    skip_existing logic used to check ONLY whether the key already had a
    non-null embedding, never whether the stored text actually still
    matched chunks.jsonl -- so a corrected/restated page could rechunk
    correctly here while the DB silently kept the OLD text+embedding
    forever under the same key. Whitespace-normalized before hashing (join
    on split()) so this hash is robust to incidental whitespace differences
    that don't change meaning, matching the normalization pattern already
    used for de-duplication elsewhere in this project."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


_MACOS_DUP_SUFFIX = re.compile(r"\s+\d+$")  # e.g. "...__p1 2" -> "...__p1" (Finder/iCloud duplicate-file naming)


def parse_filename(path: Path):
    """Split '<TICKER>__<doctype>__<stem>__p<page>.txt' back into parts.

    ticker and doctype never contain '__' themselves (they come from
    guess_ticker()/guess_doctype() in pdf_ingest_router.py, both simple
    tokens), so splitting on the double-underscore separator is safe even
    though doctype values like "financial_statement" contain a single "_".

    Also strips a trailing macOS Finder/iCloud duplicate-file marker
    (" 2", " 3", ...) if present, e.g. "...__p1 2.txt" -> page 1, flagged
    as `is_duplicate_copy=True` so the caller can skip re-chunking content
    that's already been chunked from the non-suffixed original (found in
    a real corpus: /Users/mac/Desktop/... is under iCloud Desktop sync,
    which produces exactly this " N" suffix on conflicted files).
    """
    stem = path.stem  # drops .txt
    is_duplicate_copy = bool(_MACOS_DUP_SUFFIX.search(stem))
    stem = _MACOS_DUP_SUFFIX.sub("", stem)

    parts = stem.split("__")
    if len(parts) < 4:
        return None
    ticker = parts[0]
    doctype = parts[1]
    page_part = parts[-1]
    m = re.match(r"p(\d+)$", page_part)
    if not m:
        return None
    page = int(m.group(1))
    source_file = "__".join(parts[2:-1])
    return {
        "ticker": ticker, "doctype": doctype, "source_file": source_file,
        "page": page, "is_duplicate_copy": is_duplicate_copy,
    }


def normalize_page_text(raw_text: str) -> str:
    # Collapse line-wrap newlines into spaces; real paragraph breaks (if any
    # ever show up as blank lines) are preserved as a single space too --
    # word-level splitting doesn't care either way.
    text = re.sub(r"\s*\n\s*", " ", raw_text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Fixed-size sliding window over `text`, breaking at whitespace, with
    overlap between consecutive chunks. Exposed as a standalone function so
    src/retrieval/rag_pipeline.py (query-time, not built yet) or tests can
    reuse it without going through the file-based CLI below."""
    if len(text) <= size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            space_pos = text.rfind(" ", start, end)
            if space_pos > start:
                end = space_pos
        piece = text[start:end].strip()
        if piece:
            if chunks and len(piece) < MIN_CHUNK_SIZE:
                chunks[-1] = (chunks[-1] + " " + piece).strip()
            else:
                chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # ensure forward progress even on odd inputs
    return chunks


MANIFEST_FIELDNAMES = ["file", "file_hash", "chunking_version", "normalization_version"]


def load_existing_manifest(manifest_path: Path) -> dict:
    """filename -> {"hash": str, "chunking_version": str, "normalization_version": str},
    or {} if no manifest yet."""
    cache = {}
    if not manifest_path.exists():
        return cache
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cache[row["file"]] = {
                "hash": row["file_hash"],
                "chunking_version": row.get("chunking_version", ""),
                "normalization_version": row.get("normalization_version", ""),
            }
    return cache


def load_existing_chunks(chunks_path: Path) -> dict:
    """(ticker, source_file, page) -> list of chunk record dicts, in original
    chunk_index order, or {} if chunks_path doesn't exist yet (first run)."""
    by_page = {}
    if not chunks_path.exists():
        return by_page
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            page_key = (record["ticker"], record["source_file"], record["page"])
            by_page.setdefault(page_key, []).append(record)
    return by_page


def split_directory(in_dir: Path, out_path: Path, verbose: bool = True, force: bool = False) -> dict:
    """Chunk every .txt in in_dir and write JSONL records to out_path.

    Incremental by default: a file whose content hash AND chunking_version
    match the last run (recorded in chunk_manifest.csv next to out_path)
    has its previously written chunks carried forward unchanged instead of
    being re-split. Pass force=True to ignore the manifest and re-chunk
    everything regardless (e.g. after a chunk_text() algorithm change that
    CHUNKING_VERSION can't detect on its own).

    Returns a small summary dict (used by both the CLI below and tests)."""
    import sys
    import time

    txt_files = sorted(in_dir.glob("*.txt"))
    if verbose:
        print(f"Found {len(txt_files)} .txt file(s) in {in_dir} -- starting...", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.parent / "chunk_manifest.csv"
    old_manifest = {} if force else load_existing_manifest(manifest_path)
    old_chunks_by_page = {} if force else load_existing_chunks(out_path)

    n_chunks = 0
    n_reused_files = 0
    n_rechunked_files = 0
    skipped_names = []
    n_duplicates_skipped = 0
    seen_pages = set()  # (ticker, source_file, page) already written this run
    new_manifest_rows = []
    t_start = time.monotonic()

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, txt_path in enumerate(txt_files, start=1):
            if verbose and (i % 100 == 0 or i == len(txt_files)):
                elapsed = time.monotonic() - t_start
                print(f"  ...{i}/{len(txt_files)} files ({elapsed:.1f}s elapsed)", flush=True)

            t_file_start = time.monotonic()
            meta = parse_filename(txt_path)
            if meta is None:
                skipped_names.append(txt_path.name)
                continue

            page_key = (meta["ticker"], meta["source_file"], meta["page"])
            if page_key in seen_pages:
                # Same (ticker, source_file, page) already chunked from
                # another file this run -- almost always a filesystem-level
                # duplicate (e.g. iCloud Desktop sync conflict producing
                # "...__p1 2.txt" next to "...__p1.txt"), not new content.
                # Chunking it again would silently duplicate that page's
                # content in pgvector and inflate its retrieval score.
                n_duplicates_skipped += 1
                continue
            seen_pages.add(page_key)

            # Content hash first -- needed either way (to reuse, or to
            # record in the manifest for next time), and it's cheap (pure
            # I/O) compared to the regex chunking it lets us skip.
            file_hash = _file_hash(txt_path)
            cached_meta = old_manifest.get(txt_path.name)
            if cached_meta is not None:
                old_records = old_chunks_by_page.get(page_key) or []
                unchanged = (
                    cached_meta["hash"] == file_hash
                    and cached_meta["chunking_version"] == CHUNKING_VERSION
                    and cached_meta["normalization_version"] == NORMALIZATION_VERSION
                    and bool(old_records)
                    # Defense in depth: also require every stored chunk
                    # record to carry the matching per-record stamp, not
                    # just the file-level manifest row -- catches the case
                    # of a hand-edited/stale chunks.jsonl that disagrees
                    # with the manifest.
                    and all(r.get("chunking_version") == CHUNKING_VERSION
                            and r.get("normalization_version") == NORMALIZATION_VERSION
                            for r in old_records)
                )
            else:
                # No manifest entry (e.g. this is the very first run after
                # incremental chunking was added, so chunk_manifest.csv
                # doesn't exist yet) -- but chunks.jsonl from a PRIOR run
                # MAY already have chunks for this exact page. Only trust
                # that existing output if EVERY one of its chunk records
                # carries a "chunking_version" stamp equal to the CURRENT
                # CHUNKING_VERSION (added 2026-07-29, tightened same day
                # after a real gap was pointed out: blindly trusting
                # presence alone -- with no manifest AND no per-record
                # version proof -- could silently reuse chunks produced
                # under a different CHUNK_SIZE/CHUNK_OVERLAP, e.g. if
                # chunk_manifest.csv was deleted by accident while
                # chunks.jsonl itself was from an older config). Records
                # written before this fix have no "chunking_version" key at
                # all, so .get(...) returns None, correctly failing the
                # comparison and forcing one safe one-time rechunk instead
                # of trusting unverifiable data.
                old_records = old_chunks_by_page.get(page_key) or []
                unchanged = bool(old_records) and all(
                    r.get("chunking_version") == CHUNKING_VERSION
                    and r.get("normalization_version") == NORMALIZATION_VERSION
                    for r in old_records
                )

            if unchanged:
                # File untouched since last run -- reuse its already-computed
                # chunks instead of re-reading/re-splitting the page text.
                for record in old_chunks_by_page[page_key]:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_chunks += 1
                n_reused_files += 1
                new_manifest_rows.append({
                    "file": txt_path.name, "file_hash": file_hash,
                    "chunking_version": CHUNKING_VERSION, "normalization_version": NORMALIZATION_VERSION,
                })
                continue

            n_rechunked_files += 1
            raw = txt_path.read_text(encoding="utf-8")
            file_read_time = time.monotonic() - t_file_start
            if verbose and file_read_time > 2.0:
                # A single small .txt file taking >2s to read is not normal
                # disk I/O -- on a project folder under iCloud Desktop sync,
                # this usually means the file is an iCloud "evicted"
                # placeholder being downloaded on demand, not a script bug.
                print(f"  (slow read: {txt_path.name} took {file_read_time:.1f}s -- "
                      f"possible iCloud placeholder download, not a hang)", file=sys.stderr, flush=True)

            normalized = normalize_page_text(raw)
            pieces = chunk_text(normalized)
            for idx, piece in enumerate(pieces):
                record = {
                    "ticker": meta["ticker"], "doctype": meta["doctype"],
                    "source_file": meta["source_file"], "page": meta["page"],
                    "chunk_index": idx, "chunk_text": piece, "char_count": len(piece),
                    "chunk_hash": chunk_hash(piece), "chunking_version": CHUNKING_VERSION,
                    "normalization_version": NORMALIZATION_VERSION,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1
            new_manifest_rows.append({
                "file": txt_path.name, "file_hash": file_hash,
                "chunking_version": CHUNKING_VERSION, "normalization_version": NORMALIZATION_VERSION,
            })

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        w.writeheader()
        w.writerows(new_manifest_rows)

    if skipped_names:
        skip_report = out_path.parent / "chunks_skipped_files.txt"
        skip_report.write_text("\n".join(skipped_names), encoding="utf-8")
    else:
        skip_report = None

    return {
        "n_files": len(txt_files), "n_skipped_files": len(skipped_names),
        "n_duplicates_skipped": n_duplicates_skipped,
        "n_reused_files": n_reused_files, "n_rechunked_files": n_rechunked_files,
        "n_chunks": n_chunks, "skipped_names": skipped_names, "skip_report": skip_report,
        "manifest_path": manifest_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="data/processed/text_native directory")
    ap.add_argument("--output", required=True, help="output .jsonl path, e.g. data/processed/chunks.jsonl")
    ap.add_argument("--force", action="store_true",
                     help="Re-chunk every page even if unchanged since the last run "
                          "(e.g. after changing CHUNK_SIZE/CHUNK_OVERLAP)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    if not any(in_dir.glob("*.txt")):
        print(f"No .txt files found in {in_dir} -- run pdf_ingest_router.py first.")
        return

    summary = split_directory(in_dir, Path(args.output), force=args.force)
    print(f"Wrote {summary['n_chunks']} chunks from "
          f"{summary['n_files'] - summary['n_skipped_files'] - summary['n_duplicates_skipped']} pages "
          f"({summary['n_reused_files']} page(s) reused unchanged from last run, "
          f"{summary['n_rechunked_files']} page(s) newly (re)chunked, "
          f"{summary['n_skipped_files']} file(s) skipped - unrecognized filename pattern, "
          f"{summary['n_duplicates_skipped']} file(s) skipped - duplicate copy of an already-chunked page) "
          f"-> {args.output}")
    print(f"Manifest: {summary['manifest_path']}")
    if summary["skipped_names"]:
        print(f"\nFirst {min(10, len(summary['skipped_names']))} skipped filename(s):")
        for name in summary["skipped_names"][:10]:
            print(f"  {name}")
        print(f"Full list: {summary['skip_report']}")


if __name__ == "__main__":
    main()
