#!/usr/bin/env python3
"""
SUPERSEDED - do not use. Kept only because this file couldn't be deleted
in the session that created it. The real, current version of this logic
is src/retrieval/text_splitter.py (matches the project's existing scaffold
naming) - use that instead:

    python3 -m src.retrieval.text_splitter --input data/processed/text_native --output data/processed/chunks.jsonl

Everything below is preserved only for reference / git history, not run.
---
Chunks the text-native pages produced by scripts/pdf_ingest_router.py into
retrieval-sized pieces and writes them to a JSONL file, ready for
src/retrieval/embedder.py to embed and load into pgvector.

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
  python3 src/retrieval/chunker.py --input data/processed/text_native --output data/processed/chunks.jsonl
"""
import argparse
import json
import re
from pathlib import Path

CHUNK_SIZE = 800     # target characters per chunk
CHUNK_OVERLAP = 120  # characters of overlap between consecutive chunks
MIN_CHUNK_SIZE = 80  # chunks shorter than this get merged into the previous one instead of standing alone

FILENAME_RE = re.compile(r"^(?P<ticker>[^_]+)__(?P<doctype>[^_]+(?:_[^_]+)*?)__(?P<stem>.+)__p(?P<page>\d+)$")


def parse_filename(path: Path):
    """Split '<TICKER>__<doctype>__<stem>__p<page>.txt' back into parts.

    ticker and doctype never contain '__' themselves (they come from
    guess_ticker()/guess_doctype() in pdf_ingest_router.py, both simple
    tokens), so splitting on the double-underscore separator is safe even
    though doctype values like "financial_statement" contain a single "_".
    """
    stem = path.stem  # drops .txt
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
    return {"ticker": ticker, "doctype": doctype, "source_file": source_file, "page": page}


def normalize_page_text(raw_text: str) -> str:
    # Collapse line-wrap newlines into spaces; real paragraph breaks (if any
    # ever show up as blank lines) are preserved as a single space too --
    # word-level splitting doesn't care either way.
    text = re.sub(r"\s*\n\s*", " ", raw_text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    if len(text) <= size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # back up to the nearest space so we don't cut a word in half
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="data/processed/text_native directory")
    ap.add_argument("--output", required=True, help="output .jsonl path, e.g. data/processed/chunks.jsonl")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(in_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {in_dir} -- run pdf_ingest_router.py first.")
        return

    n_chunks = 0
    n_skipped_files = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for txt_path in txt_files:
            meta = parse_filename(txt_path)
            if meta is None:
                print(f"Skip (unrecognized filename pattern): {txt_path.name}")
                n_skipped_files += 1
                continue

            raw = txt_path.read_text(encoding="utf-8")
            normalized = normalize_page_text(raw)
            pieces = chunk_text(normalized)

            for idx, piece in enumerate(pieces):
                record = {
                    "ticker": meta["ticker"],
                    "doctype": meta["doctype"],
                    "source_file": meta["source_file"],
                    "page": meta["page"],
                    "chunk_index": idx,
                    "chunk_text": piece,
                    "char_count": len(piece),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"Wrote {n_chunks} chunks from {len(txt_files) - n_skipped_files} pages "
          f"({n_skipped_files} file(s) skipped) -> {out_path}")


if __name__ == "__main__":
    main()
