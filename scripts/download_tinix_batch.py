"""Targeted re-download (2026-07-23, second pass): the first full batch
(2026-07-21, 219 tickers) brought RAG coverage to 209/222. This is the
*remaining gap* - the exact 13-ticker list printed by
scripts/export_rag_missing_tickers.py's latest run (has_dag=205/222,
has_ratios=204/222, has_price=207/222, has_rag=209/222 before this run).

One entry ("AAAA") was DROPPED from that script's raw output again - same
known issue as the first batch: verified via web search this is not a real
Vietnamese ticker (no listing found anywhere, unlike the genuine "AAA" - An
Phat Bioplastics). It's still sitting in `companies` as a bad row from a
malformed cell in one Vietstock export file (data/raw/vietstock_exports/
VietstockFinance_Bao-cao-tai-chinh_20260714-190643.xlsx) - still worth
deleting (DELETE FROM companies WHERE ticker='AAAA' AND NOT EXISTS data
referencing it elsewhere), it just hasn't been done yet. Left "Beta" IN -
confirmed real via top200_candidate_companies.csv, a legitimate OTC-market
ticker (OTC uses different naming conventions than HOSE/HNX/UPCOM's
standard 3-4 letter codes).

Usage:
    pip install huggingface_hub --break-system-packages   # if not already installed
    python3 -m scripts.download_tinix_batch
    python3 scripts/tinix_ingest_adapter.py --input data/raw/tinix_ocr --output data/processed
    python3 -m src.retrieval.text_splitter --input data/processed/text_native --output data/processed/chunks.jsonl
    python3 -m src.retrieval.vector_store --input data/processed/chunks.jsonl
    (no index rebuild needed - HNSW updates incrementally, see README Step 10)

NOTE: only 12 tickers this time (vs. 219 in the first batch) - should be
quick. Safe to interrupt and re-run - text_splitter/vector_store are
idempotent (upsert on the same unique key), and tinix_ingest_adapter.py
skips files it's already processed per its manifest. After this run,
re-run scripts/export_rag_missing_tickers.py to confirm has_rag reaches
222/222 (minus AAAA, which isn't a real ticker and should just be deleted
from `companies` instead of chased for coverage).
"""
from huggingface_hub import snapshot_download

TICKERS = [
    "AAN", "AMV", "API", "BMC", "BNA", "Beta", "C21", "CAD", "CAN", "CC1",
    "CDO", "CMI",
]

allow_patterns = [f"{t}/*" for t in TICKERS]

print(f"Downloading TiniX data for {len(TICKERS)} tickers...")
snapshot_download(
    repo_id="vduydong/ocr_annual_financials",
    repo_type="dataset",
    allow_patterns=allow_patterns,
    local_dir="data/raw/tinix_ocr",
)
print("Done. Next: run tinix_ingest_adapter.py -> text_splitter.py -> vector_store.py "
      "(see this file's module docstring for exact commands) - no index rebuild needed (HNSW).")
