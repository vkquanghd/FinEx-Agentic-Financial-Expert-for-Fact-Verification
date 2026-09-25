#!/usr/bin/env python3
"""
Adapter for the "TiniX Vietnam OCR Annual Financial Statements (2015-2025)"
dataset (HuggingFace: vduydong/ocr_annual_financials, CC-BY-NC-4.0) - turns
its already-OCR'd report files into the exact same per-page .txt shape that
scripts/pdf_ingest_router.py produces from raw PDFs, so the rest of the RAG
pipeline (src/retrieval/text_splitter.py -> embeddings.py -> vector_store.py)
runs completely unchanged on top of it.

Why this exists: docs/Thesis_Direction_Redesign.pdf's data table (section 5)
flags "PDF disclosure đầy đủ (kèm Thuyết minh)" as the one still-missing
input for the RAG path - "hiện có vài file mẫu, chưa đủ quy mô cho RAG/Level
2". TiniX has 18,231 already-OCR'd Vietnamese annual reports (balance sheet,
income statement, cash flow, AND Thuyết minh BCTC) across 1,491 tickers,
2015-2025 - directly closes that gap for any ticker it covers, without
needing this project's own OCR step (data/processed/ocr_queue.csv) at all
for those specific documents.

Format this adapter expects (as seen in the dataset preview, 2026-07-19):
    <root>/<TICKER>/<YEAR>/<report_folder>/<report_folder>_extracted.txt
  e.g. ocr_annual_financials/A32/2018/A32_Baocaotaichinh_2018_Kiemtoan/
       A32_Baocaotaichinh_2018_Kiemtoan_extracted.txt
Each file contains ALL pages of one report concatenated, separated by
"===== PAGE N =====" markers, with an occasional bare "[Non-Text]" line
right after a page marker (stamp/logo/signature image region - not real
text, stripped here).

This project's sandbox has no general internet access (outbound requests
other than the specific browsing/search tools are blocked by an allowlist
proxy - confirmed 2026-07-19), so THIS SCRIPT WAS NOT RUN AGAINST THE FULL
DATASET. It was validated against one real sample file fetched via the web
browsing tool (A32/2018 audited financial statement, ~35 pages) - the
parsing logic below is confirmed correct against that real file's exact
format. Run this for real on a machine with internet access:

  1. Download the ticker(s) you want, e.g. via huggingface_hub:
       pip install huggingface_hub
       python3 -c "
       from huggingface_hub import snapshot_download
       snapshot_download(repo_id='vduydong/ocr_annual_financials', repo_type='dataset',
                          allow_patterns=['VNM/*','CTD/*','PNJ/*','HPG/*'],  # tickers already in this project's companies table - edit as needed
                          local_dir='data/raw/tinix_ocr')
       "
     (or `huggingface-cli download vduydong/ocr_annual_financials --repo-type dataset --include "VNM/*" --local-dir data/raw/tinix_ocr`)
  2. python3 scripts/tinix_ingest_adapter.py --input data/raw/tinix_ocr --output data/processed
  3. python3 -m src.retrieval.text_splitter --input data/processed/text_native --output data/processed/chunks.jsonl
  4. python3 -m src.retrieval.vector_store --input data/processed/chunks.jsonl

Step 3-4 are UNCHANGED from the existing real-PDF pipeline (Step 10 in
README.md) - this adapter's only job is to make TiniX look like
pdf_ingest_router.py's output.

License note: CC-BY-NC-4.0 - non-commercial use only (fine for this
academic thesis; do not redistribute the raw dataset itself as part of the
project handoff, only cite it - see the dataset card's citation block).
"""
import argparse
import csv
import re
from pathlib import Path

PAGE_MARKER_RE = re.compile(r"^=====\s*PAGE\s+(\d+)\s*=====\s*$", re.MULTILINE)
_NON_TEXT_LINE_RE = re.compile(r"^\[Non-Text\]\s*$", re.MULTILINE)
MIN_CHARS_PER_PAGE = 20  # below this, treat the page as effectively empty (skip, don't write junk)


def fix_concatenated_words(text: str) -> str:
    """TiniX's raw <table> extraction sometimes concatenates adjacent list
    items within a single <td> with NO separator at all - e.g. a board
    member list becomes "...Ngọc DungÔng Lê Trí Thông..." (no space/comma
    between "Dung" and "Ông"), or a title list becomes "Chủ tịchPhó Chủ
    tịch...". This silently merges what should be N separate names/titles
    into one run-on token, which measurably hurts dense-retrieval quality
    for governance-table content: confirmed 2026-07-19 that the correct
    PNJ board-member chunk (containing the fully correct answer) didn't
    rank in the top 20 similarity hits for "Hội đồng quản trị PNJ gồm
    những ai?" - see docs/PROGRESS.md for the full investigation.

    Heuristic fix: normal Vietnamese prose essentially never has a
    lowercase letter immediately followed by an uppercase letter with
    zero characters between them - real word/sentence boundaries always
    have whitespace. The only place this actually happens in TiniX's raw
    text is these smashed-together table-cell lists. So: insert a space
    at every lowercase -> uppercase transition with no space between.
    This is a no-op anywhere a space already exists (i.e. everywhere in
    normal prose) - it only fires where a separator was dropped. Applied
    only to TiniX-sourced text (this adapter), not to real-PDF OCR output
    (pdf_ingest_router.py has its own separate path, unaffected)."""
    out = []
    prev = ""
    for ch in text:
        if prev and ch.isupper() and prev.islower():
            out.append(" ")
        out.append(ch)
        prev = ch
    return "".join(out)

# Same doctype vocabulary as scripts/pdf_ingest_router.py's guess_doctype(),
# so chunks from TiniX and chunks from real ingested PDFs share doctype
# labels (matters for any downstream filtering/analysis by doctype).
_DOCTYPE_RULES = [
    ("baocaothuongnien", "annual_report"),
    ("baocaotaichinh", "financial_statement"),
    ("giaitrinh", "kqkd_explanation"),
    ("nghiquyet", "agm_resolution"),
    ("baocaoquantri", "governance_report"),
    ("boardresolution", "board_resolution"),
]


def guess_doctype(name: str) -> str:
    low = name.lower().replace(" ", "").replace("_", "")
    for key, label in _DOCTYPE_RULES:
        if key in low:
            return label
    return "other"


_TICKER_RE = re.compile(r"^[A-Za-z0-9]{2,6}$")


def find_report_files(root: Path):
    """Yield (ticker, txt_path) for every '<...>_extracted.txt' (or any .txt)
    found under root/<TICKER>/... - ticker is taken from the first path
    component under root, not guessed from the filename, since TiniX's own
    folder structure already gives it exactly (more reliable than
    guess_ticker()'s filename heuristics in pdf_ingest_router.py).

    Does NOT force-uppercase the ticker (fixed 2026-07-23, real bug found
    via scripts/diagnose_missing_12_tickers.py): the original version did
    `rel.parts[0].upper()`, which silently turned the real mixed-case OTC
    ticker "Beta" into "BETA" on ingestion. `companies`/every other table
    in this project stores it as "Beta" (confirmed real via
    top200_candidate_companies.csv - OTC tickers don't follow HOSE/HNX/
    UPCOM's standard all-caps convention), so every RAG query scoped to
    ticker='Beta' silently found zero rows even though 10 real TiniX files
    for it were ingested and embedded successfully - `has_rag` never
    reflected the true state. Standard tickers are unaffected either way
    since TiniX's own folder names for them are already uppercase (e.g.
    "A32", "AAA"). `_TICKER_RE` widened to allow lowercase letters so
    "Beta" (and any other real mixed-case OTC ticker) still passes the
    folder-name sanity check instead of being silently skipped."""
    for txt_path in sorted(root.rglob("*.txt")):
        rel = txt_path.relative_to(root)
        if len(rel.parts) < 2:
            continue  # not under a <TICKER>/... subfolder, skip
        ticker = rel.parts[0]
        if not _TICKER_RE.match(ticker):
            continue
        yield ticker, txt_path


def split_pages(raw_text: str):
    """Return {page_number: page_text}, using the '===== PAGE N ====='
    markers. Text before the first marker (if any) is discarded - in the
    one real sample checked it was always empty (file starts immediately
    with '===== PAGE 1 ====='), so this shouldn't lose real content."""
    matches = list(PAGE_MARKER_RE.finditer(raw_text))
    pages = {}
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end]
        body = _NON_TEXT_LINE_RE.sub("", body).strip()
        body = fix_concatenated_words(body)
        pages[page_num] = body
    return pages


def process_file(ticker: str, txt_path: Path, text_dir: Path, manifest_rows: list) -> int:
    doctype = guess_doctype(txt_path.name)
    stem = txt_path.stem  # e.g. "A32_Baocaotaichinh_2018_Kiemtoan_extracted"
    if stem.endswith("_extracted"):
        stem = stem[: -len("_extracted")]

    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    pages = split_pages(raw)

    n_written = 0
    for page_num, body in sorted(pages.items()):
        n_chars = len(body)
        if n_chars < MIN_CHARS_PER_PAGE:
            manifest_rows.append({
                "source": str(txt_path), "ticker": ticker, "doctype": doctype,
                "page": page_num, "char_count": n_chars, "status": "skipped_too_short",
            })
            continue
        out_file = text_dir / f"{ticker}__{doctype}__{stem}__p{page_num}.txt"
        out_file.write_text(body, encoding="utf-8")
        manifest_rows.append({
            "source": str(txt_path), "ticker": ticker, "doctype": doctype,
            "page": page_num, "char_count": n_chars, "status": "written",
        })
        n_written += 1
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                     help="Root folder of the downloaded TiniX subset, e.g. data/raw/tinix_ocr "
                          "(expects <input>/<TICKER>/<YEAR>/.../*.txt)")
    ap.add_argument("--output", required=True,
                     help="Same --output folder used for scripts/pdf_ingest_router.py, e.g. data/processed "
                          "(writes into <output>/text_native/, alongside any real-PDF pages already there)")
    ap.add_argument("--tickers", default="",
                     help="Optional comma-separated ticker allowlist, e.g. VNM,CTD,PNJ - "
                          "skips any ticker folder not in this list. Default: process every ticker found.")
    args = ap.parse_args()

    in_root = Path(args.input)
    if not in_root.exists():
        print(f"Input folder not found: {in_root}")
        return

    text_dir = Path(args.output) / "text_native"
    text_dir.mkdir(parents=True, exist_ok=True)

    allow = {t.strip().upper() for t in args.tickers.split(",") if t.strip()} or None

    manifest_rows = []
    n_files = n_written_total = n_tickers = 0
    seen_tickers = set()

    for ticker, txt_path in find_report_files(in_root):
        if allow is not None and ticker not in allow:
            continue
        seen_tickers.add(ticker)
        n_files += 1
        n_written_total += process_file(ticker, txt_path, text_dir, manifest_rows)

    manifest_path = Path(args.output) / "tinix_ingest_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "ticker", "doctype", "page", "char_count", "status"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"{n_files} report file(s) processed across {len(seen_tickers)} ticker(s): {sorted(seen_tickers)}")
    print(f"{n_written_total} page(s) written to {text_dir}")
    print(f"Manifest: {manifest_path}")
    print("\nNext steps (unchanged from the real-PDF pipeline, README.md Step 10):")
    print(f"  python3 -m src.retrieval.text_splitter --input {text_dir} --output {Path(args.output) / 'chunks.jsonl'}")
    print(f"  python3 -m src.retrieval.vector_store --input {Path(args.output) / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
