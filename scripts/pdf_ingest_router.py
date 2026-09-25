#!/usr/bin/env python3
"""
PDF ingestion router: scans a flat folder of Vietstock PDF disclosures and
splits PAGES (not whole files) into "text_ok" (usable directly, no OCR) vs
"needs_ocr" (scanned image or garbled text), based on:

  1. Extractable character count per page (pdftotext -raw).
  2. Legibility ratio: fraction of non-whitespace characters that are NOT in
     a "suspicious" Unicode block. This catches a real failure mode found in
     actual Vietstock PDFs during testing (TCH annual report, PVE board
     resolution): some pages have plenty of extractable characters but the
     font's cmap is broken, so Vietnamese diacritics silently decode to
     Cyrillic/Greek/PUA codepoints (e.g. "TỔNG" -> "TӘNG"). A char-count-only
     check would wrongly accept this garbage as "text_ok".

Why per-page, not per-document: mixed documents exist (e.g. PVE's board
resolution has a garbled logo page followed by clean real contract text).
Classifying at document level would either discard good pages or keep bad ones.

Why this matters for bias: text-native vs scanned is NOT randomly distributed
across the corpus -- it correlates with document type (annual reports tend to
be text-native; signed financial statements/AGM resolutions/governance reports
tend to be scanned) and potentially with company (older/smaller companies may
only have scanned filings). If OCR introduces more noise than native text
extraction, RAG answer quality could end up confounded by document type/company
rather than reflecting the retrieval method itself. The manifest this script
writes is the audit trail to check for that confound later (group results by
doctype/ticker/status and look for systematic gaps).

Output (under --output):
  text_native/<TICKER>__<doctype>__<original_stem>__p<page>.txt
      One file per usable page, ready for chunking + embedding.
  ingest_manifest.csv
      Every page seen, with ticker, doctype, char_count, legibility_ratio,
      and final status. This is what you join against benchmark results
      later to check for bias by document type / company.
  ocr_queue.csv
      (file, page) pairs still needing OCR -- feed this into the OCR step.

Incremental runs: safe to re-run any time you drop new PDFs into --input.
Already-processed files are recognized by SHA-256 content hash (2026-07-29 -
was (filename, file size, mtime) before; changed after a review pointed out
that a PDF restored/re-synced with the same bytes but a different mtime
should NOT be treated as changed, mirroring the same fix already made in
src/retrieval/text_splitter.py's chunk_manifest.csv) recorded in
ingest_manifest.csv from the previous run, and are SKIPPED, not reprocessed --
only new or changed files are actually scanned. INGEST_VERSION (derived from
CHAR_COUNT_MIN/LEGIBILITY_MIN below) is also checked, so changing either
threshold auto-invalidates the whole manifest without needing --force.
Pass --force to reprocess everything regardless (e.g. after a
pdftotext/legibility_ratio() logic change that isn't captured by
INGEST_VERSION). A manifest row written before 2026-07-29 has no pdf_hash/
ingest_version at all, so it safely fails this comparison once and that PDF
gets reprocessed a single time (cheap - pdftotext/pdfinfo only, no GPU/model).

Run:
  python3 scripts/pdf_ingest_router.py --input data/raw/pdf_disclosures --output data/processed
  python3 scripts/pdf_ingest_router.py --input data/raw/pdf_disclosures --output data/processed --force
"""
import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

CHAR_COUNT_MIN = 150   # below this, a page is almost certainly a scanned image
LEGIBILITY_MIN = 0.90  # below this, extracted text is likely garbled (broken font cmap)

# Auto-derived version stamp (2026-07-29, same pattern as src/retrieval/
# text_splitter.py's CHUNKING_VERSION) - bump CHAR_COUNT_MIN/LEGIBILITY_MIN
# and this changes automatically, invalidating the whole manifest without
# needing --force. Bump the trailing "_v1" by hand if extract_page_text()/
# legibility_ratio()'s ALGORITHM changes in a way that doesn't touch either
# constant.
INGEST_VERSION = f"charmin{CHAR_COUNT_MIN}_legmin{LEGIBILITY_MIN}_v1"


def _file_hash(path: Path) -> str:
    """SHA-256 of file bytes, streamed so large PDFs don't need to fit in
    RAM at once. Deliberately duplicated (not imported) from
    src/retrieval/text_splitter.py's identical helper - this script is
    designed to work standalone with no dependency beyond a flat CSV file
    (see the ticker-reference comment below), so it doesn't reach into
    src/ for anything."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()

# Unicode blocks that should never legitimately appear in Vietnamese business
# documents. Seeing them is a strong signal of a broken font cmap, not real text.
SUSPICIOUS_RANGES = [
    (0x0400, 0x04FF),  # Cyrillic
    (0x0370, 0x03FF),  # Greek
    (0x0530, 0x058F),  # Armenian
    (0xE000, 0xF8FF),  # Private Use Area
]


def is_suspicious(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in SUSPICIOUS_RANGES)


def pdf_page_count(path: Path) -> int:
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else 0


def extract_page_text(path: Path, page: int) -> str:
    r = subprocess.run(
        ["pdftotext", "-raw", "-f", str(page), "-l", str(page), str(path), "-"],
        capture_output=True, text=True,
    )
    return r.stdout or ""


def legibility_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = sum(1 for c in chars if is_suspicious(c))
    return 1 - (bad / len(chars))


# Common Vietnamese words that show up in Vietstock filenames and are NOT
# tickers (e.g. "0_Ban_CBTT_Tai_lieu_AGM_2026...pdf" -> "Ban" is "Bản/Ban"
# as in "Ban công bố thông tin", not a stock code). Extend this list as you
# encounter more false positives -- it's a denylist, not a real ticker DB.
_NOT_TICKER = {
    "ban", "cbtt", "tai", "lieu", "cong", "bo", "thong", "tin", "bao", "cao",
    "nghi", "quyet", "dhcd", "thuong", "nien", "quan", "tri", "giai", "trinh",
    "kqkd", "tai", "chinh", "en", "vn", "signed", "sign", "board", "resolution",
    # "agm" is a REAL ticker (An Giang Import-Export, confirmed in
    # data/reference/top200_candidate_companies.csv) but in this corpus it
    # shows up 12+ times as the English abbreviation for "Annual General
    # Meeting" inside VNM's own governance-document filenames (e.g.
    # "0_Ban_CBTT_Tai_lieu_AGM_2026_...pdf", "10_AGM_2026_AGM_Organization_
    # Regulations_...pdf") - confirmed by inspecting data/raw/pdf_disclosures/
    # (2026-07-18): every AGM-token file in the corpus sits alongside VNM_-
    # prefixed files for the same meeting, none are actually An Giang
    # Import-Export documents. Denylisting trades away the (currently
    # nonexistent in this corpus) real AGM-ticker case for eliminating a
    # much more common false positive.
    "agm", "meeting", "notice", "papers", "organization", "regulations",
}


# --- Ticker reference (for guess_ticker's company-name-matching fallback) ---
# Duplicated from scripts/seed_companies.py's CORE_16 + the same CSV it
# reads, rather than importing that module - keeps this script's only real
# dependency a flat CSV file, not a DB-writing script (seed_companies.py
# executes MySQL INSERTs when run, and pdf_ingest_router.py should keep
# working standalone with no DB even reachable). Falls back to CSV-only (16
# fewer companies) if it's ever out of sync - not worth the coupling to
# keep these two lists byte-identical.
_CORE_16 = [
    ("VNM", "CTCP Sữa Việt Nam"),
    ("SAB", "Tổng Công ty cổ phần Bia - Rượu - Nước giải khát Sài Gòn"),
    ("MSN", "CTCP Tập đoàn Masan"),
    ("KDC", "CTCP Tập đoàn KIDO"),
    ("VIC", "Tập đoàn VINGROUP - CTCP"),
    ("VHM", "CTCP Vinhomes"),
    ("NVL", "CTCP Tập đoàn Đầu tư Địa ốc No Va"),
    ("KDH", "CTCP Đầu tư và Kinh doanh Nhà Khang Điền"),
    ("HPG", "CTCP Tập đoàn Hòa Phát"),
    ("GAS", "Tổng Công ty Khí Việt Nam - CTCP"),
    ("DPM", "Tổng công ty Phân bón và Hóa chất Dầu khí - CTCP"),
    ("DCM", "CTCP - Tổng công ty Phân bón Dầu khí Cà Mau"),
    ("MWG", "CTCP Đầu tư Thế giới Di động"),
    ("PNJ", "CTCP Vàng bạc Đá quý Phú Nhuận"),
    ("DGW", "CTCP Thế Giới Số"),
    ("FRT", "CTCP Bán lẻ Kỹ thuật số FPT"),
]
_CANDIDATES_CSV = Path(__file__).resolve().parents[1] / "data" / "reference" / "top200_candidate_companies.csv"

# Known brand-name aliases that show up in real filenames instead of the
# official legal company name - e.g. "4_Vinamilk_Draft_Charter_...pdf" and
# "7_Vinamilk_Corporate_Governance_Regulations_...pdf" (confirmed in
# data/raw/pdf_disclosures/, 2026-07-18) use the brand "Vinamilk", which
# doesn't substring-match "CTCP Sữa Việt Nam" at all. Add entries here as
# more brand-name-only filenames turn up; deliberately small/manual rather
# than a fuzzy-match library, since a wrong alias silently mislabels a
# document with someone else's ticker.
_ALIASES = [
    ("VNM", "Vinamilk"),
]

# Corporate-form words stripped from company names before building a
# matching key - generic across almost every VN company, so keeping them
# would cause false negatives whenever a filename drops one but not
# another (e.g. keeps "Cong_ty" but drops "Co_Phan").
_CORPORATE_FORM_WORDS = {"ctcp", "tct", "tnhh", "congty", "conty", "tapdoan", "tongcongty", "tong", "cophan", "phan", "jsc", "co", "ty"}


def _normalize_for_matching(text: str) -> str:
    """Strip diacritics, lowercase, drop everything but letters/digits - so
    company names and filenames compare on equal footing regardless of
    accents/underscores/casing (e.g. 'Sữa Việt Nam' and 'Sua_Viet_Nam'
    both normalize to a form containing 'suavietnam')."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", ascii_only.lower())


def _company_search_key(company_name: str) -> str:
    """'CTCP Sữa Việt Nam' -> 'suavietnam' (corporate-form words dropped)."""
    words = re.split(r"[\s\-]+", company_name)
    kept = [w for w in words if _normalize_for_matching(w) not in _CORPORATE_FORM_WORDS]
    return _normalize_for_matching(" ".join(kept))


def _load_ticker_reference() -> list:
    """[(ticker, search_key), ...], longest search_key first so the
    substring match in guess_ticker() prefers the most specific (least
    likely to false-positive) company name when more than one could
    technically match. Keys shorter than 6 normalized characters are
    dropped entirely - too short to substring-match safely."""
    ref = list(_CORE_16) + list(_ALIASES)
    if _CANDIDATES_CSV.exists():
        with _CANDIDATES_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ref.append((row["ticker"].strip(), row["company_name"].strip()))
    keyed = [(ticker, _company_search_key(name)) for ticker, name in ref]
    keyed = [(t, k) for t, k in keyed if len(k) >= 6]
    keyed.sort(key=lambda pair: -len(pair[1]))
    return keyed


_TICKER_REFERENCE = None  # lazy-loaded on first guess_ticker() call
_REAL_TICKERS = None


def guess_ticker(filename: str) -> str:
    # Vietstock filenames are NOT consistent across document types (see the
    # 7-file sample: NTC_..., TCH_..., but also "1_pve_2026_7_16_..." and
    # "0_Ban_CBTT_Tai_lieu_AGM_2026_...pdf", and manually-uploaded governance
    # docs named descriptively with no ticker at all, e.g.
    # "4_Du_thao_Dieu_le_CTCP_Sua_Viet_Nam_..." for VNM. CONFIRMED REAL BUG
    # (2026-07-18, found via a live RAG query returning "DU" as the source
    # ticker for that exact file): the old version of this function took the
    # first 2-5-char alphabetic token not on the stopword denylist, with NO
    # check that it was an actual ticker - "Du" (from "Du thao" = "Draft")
    # passed that check and was blindly returned.
    #
    # Fix, two passes: (1) try the fast first-token heuristic same as
    # before, but only ACCEPT a candidate if it's cross-checked against the
    # real 216-company ticker list; (2) if nothing validates, fall back to
    # matching the filename against company NAMES (diacritics/case/
    # underscore-insensitive substring match) - catches the VNM case above
    # via "Sua Viet Nam" -> "suavietnam" found inside the normalized filename.
    global _TICKER_REFERENCE, _REAL_TICKERS
    if _TICKER_REFERENCE is None:
        _TICKER_REFERENCE = _load_ticker_reference()
        _REAL_TICKERS = {t for t, _ in _TICKER_REFERENCE}

    stem = Path(filename).stem
    tokens = re.split(r"[_\W]+", stem)
    for tok in tokens:
        if 2 <= len(tok) <= 5 and tok.isalpha() and tok.lower() not in _NOT_TICKER:
            candidate = tok.upper()
            if candidate in _REAL_TICKERS:
                return candidate

    normalized_filename = _normalize_for_matching(stem)
    for ticker, key in _TICKER_REFERENCE:
        if key in normalized_filename:
            return ticker

    return "UNKNOWN"


def guess_doctype(filename: str) -> str:
    low = filename.lower().replace(" ", "").replace("_", "")
    rules = [
        ("baocaothuongnien", "annual_report"),
        ("baocaotaichinh", "financial_statement"),
        ("giaitrinh", "kqkd_explanation"),
        ("nghiquyet", "agm_resolution"),
        ("baocaoquantri", "governance_report"),
        ("boardresolution", "board_resolution"),
    ]
    for key, label in rules:
        if key in low:
            return label
    return "other"


def process_pdf(path: Path, out_dir: Path, manifest_rows: list, ocr_queue_rows: list, pdf_hash: str):
    ticker = guess_ticker(path.name)
    doctype = guess_doctype(path.name)
    n_pages = pdf_page_count(path)
    text_dir = out_dir / "text_native"
    text_dir.mkdir(parents=True, exist_ok=True)

    # Wipe any stale per-page output from a previous version of this exact
    # filename before rewriting -- matters if the new version has fewer pages
    # or different ticker/doctype guesses than last time.
    for old in text_dir.glob(f"*__{re.escape(path.stem)}__p*.txt"):
        old.unlink()

    for page in range(1, n_pages + 1):
        text = extract_page_text(path, page)
        n_chars = len(text.strip())
        legibility = legibility_ratio(text)
        status = "text_ok" if (n_chars >= CHAR_COUNT_MIN and legibility >= LEGIBILITY_MIN) else "needs_ocr"

        if status == "text_ok":
            out_file = text_dir / f"{ticker}__{doctype}__{path.stem}__p{page}.txt"
            out_file.write_text(text, encoding="utf-8")
        else:
            ocr_queue_rows.append({"file": str(path), "page": page, "ticker": ticker, "doctype": doctype})

        manifest_rows.append({
            "file": path.name, "pdf_hash": pdf_hash, "ingest_version": INGEST_VERSION,
            "ticker": ticker, "doctype": doctype, "page": page,
            "n_pages_total": n_pages, "char_count": n_chars,
            "legibility_ratio": round(legibility, 3), "status": status,
        })


def load_existing_manifest(manifest_path: Path) -> dict:
    """file name -> {"pdf_hash", "ingest_version", "rows": [manifest row
    dicts as read from CSV]}. A manifest row written before 2026-07-29 (or
    any row with a missing/blank column) has pdf_hash=None/ingest_version=""
    here, which will never equal a real hash/version - that PDF safely gets
    reprocessed exactly once instead of being incorrectly trusted."""
    cache = {}
    if not manifest_path.exists():
        return cache
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["file"]
            entry = cache.setdefault(fname, {
                "pdf_hash": row.get("pdf_hash") or None,
                "ingest_version": row.get("ingest_version") or "",
                "rows": [],
            })
            entry["rows"].append(row)
    return cache


def check_dependencies():
    missing = [tool for tool in ("pdfinfo", "pdftotext") if shutil.which(tool) is None]
    if missing:
        print(f"Missing required tool(s): {', '.join(missing)} (part of the 'poppler' package).")
        print("Install with:")
        print("  macOS:   brew install poppler")
        print("  Ubuntu:  sudo apt-get install poppler-utils")
        sys.exit(1)


def main():
    check_dependencies()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with all PDFs mixed together")
    ap.add_argument("--output", required=True, help="Folder to write text_native/, manifest, ocr_queue")
    ap.add_argument("--force", action="store_true",
                     help="Reprocess every file even if unchanged since the last run")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "ingest_manifest.csv"

    # Deletion detection (added 2026-07-30, found while designing the PDF
    # lifecycle integration test - scenario "delete PDF" was previously a
    # silent gap: this script only ever processes files CURRENTLY present in
    # --input, so a PDF removed from that folder would just vanish from the
    # rewritten manifest.csv, but its already-written text_native/*.txt pages
    # were NEVER deleted, so text_splitter.py would keep re-chunking them
    # forever and disclosure_chunks would never see them as orphaned. Always
    # computed from the manifest FILE on disk (not the `existing` cache
    # below, which --force clears) - deletion detection must run regardless
    # of --force, since --force is about content re-processing, not about
    # noticing files that disappeared.
    previous_filenames = set(load_existing_manifest(manifest_path).keys())

    existing = {} if args.force else load_existing_manifest(manifest_path)

    manifest_rows, ocr_queue_rows = [], []
    pdfs = sorted(in_dir.glob("*.pdf"))
    current_filenames = {p.name for p in pdfs}
    deleted_filenames = sorted(previous_filenames - current_filenames)

    text_dir = out_dir / "text_native"
    deleted_rows = []
    for fname in deleted_filenames:
        stem = Path(fname).stem
        n_removed = 0
        if text_dir.exists():
            for stale in text_dir.glob(f"*__{re.escape(stem)}__p*.txt"):
                stale.unlink()
                n_removed += 1
        deleted_rows.append({"file": fname})
        print(f"Deleted (no longer in --input): {fname} - removed {n_removed} stale text_native page(s)")

    if deleted_rows:
        deleted_path = out_dir / "deleted_files.csv"
        with deleted_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file"])
            w.writeheader()
            w.writerows(deleted_rows)
        print(f"Deleted-files manifest: {deleted_path} ({len(deleted_rows)} file(s)) - "
              f"run scripts/register_pdf_documents.py next to soft-delete (is_active=FALSE) "
              f"the matching pdf_documents row(s).")

    if not pdfs and not deleted_rows:
        print(f"No PDFs found in {in_dir}")
        return
    # NOTE: if pdfs is empty but deleted_rows is not (every remaining PDF was
    # removed since the last run), fall through so manifest_path/ocr_queue_path
    # below get rewritten as empty files instead of keeping stale rows for
    # files whose text_native pages were just deleted above.

    n_skipped = n_processed = 0
    for pdf in pdfs:
        pdf_hash = _file_hash(pdf)
        cached = existing.get(pdf.name)

        if cached and cached["pdf_hash"] == pdf_hash and cached["ingest_version"] == INGEST_VERSION:
            print(f"Skip (unchanged since last run): {pdf.name}")
            manifest_rows.extend(cached["rows"])
            for r in cached["rows"]:
                if r["status"] == "needs_ocr":
                    ocr_queue_rows.append({"file": str(pdf), "page": r["page"],
                                            "ticker": r["ticker"], "doctype": r["doctype"]})
            n_skipped += 1
            continue

        print(f"Processing {pdf.name} ...")
        process_pdf(pdf, out_dir, manifest_rows, ocr_queue_rows, pdf_hash=pdf_hash)
        n_processed += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "pdf_hash", "ingest_version", "ticker", "doctype",
                                           "page", "n_pages_total", "char_count",
                                           "legibility_ratio", "status"])
        w.writeheader()
        w.writerows(manifest_rows)

    ocr_queue_path = out_dir / "ocr_queue.csv"
    with ocr_queue_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "page", "ticker", "doctype"])
        w.writeheader()
        w.writerows(ocr_queue_rows)

    total = len(manifest_rows)
    ok = sum(1 for r in manifest_rows if r["status"] == "text_ok")
    by_doctype = {}
    for r in manifest_rows:
        d = by_doctype.setdefault(r["doctype"], {"text_ok": 0, "needs_ocr": 0})
        d[r["status"]] += 1

    print(f"\n{n_processed} file(s) processed, {n_skipped} file(s) skipped (unchanged), "
          f"{len(deleted_rows)} file(s) deleted since last run.")
    print(f"Done. {ok}/{total} pages usable directly, {total - ok} pages queued for OCR.")
    print("By document type:")
    for doctype, counts in sorted(by_doctype.items()):
        print(f"  {doctype:22s} text_ok={counts['text_ok']:4d}  needs_ocr={counts['needs_ocr']:4d}")
    print(f"\nManifest:  {manifest_path}")
    print(f"OCR queue: {ocr_queue_path}")


if __name__ == "__main__":
    main()
