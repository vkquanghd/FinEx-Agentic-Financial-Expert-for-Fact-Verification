"""One-off (2026-07-23): after downloading the 12-ticker targeted batch
(AAN, AMV, API, BMC, BNA, Beta, C21, CAD, CAN, CC1, CDO, CMI - see
scripts/download_tinix_batch.py) and re-running text_splitter.py +
vector_store.py (confirmed: 656,918/656,918 chunks have embeddings,
COUNT(*) matches exactly), scripts/export_rag_missing_tickers.py still
reports the SAME 13 tickers missing RAG, unchanged. Something in the
download -> chunk -> embed chain didn't actually pick up these 12 tickers,
but which stage is broken is unknown - this checks all three stages in one
pass instead of guessing:
  1. Did huggingface_hub actually download files for these tickers into
     data/raw/tinix_ocr/?
  2. Did text_splitter.py's output (data/processed/chunks.jsonl) end up
     with any chunks tagged with these tickers?
  3. Does disclosure_chunks (the live DB) have any rows for these tickers,
     under ANY ticker label (in case of a ticker-extraction/casing bug,
     not just "zero rows")?

Usage: python3 -m scripts.diagnose_missing_12_tickers
"""
import json
from pathlib import Path

TICKERS = ["AAN", "AMV", "API", "BMC", "BNA", "Beta", "C21", "CAD", "CAN", "CC1", "CDO", "CMI"]

print("=== Stage 1: data/raw/tinix_ocr/ folder contents ===")
raw_dir = Path("data/raw/tinix_ocr")
if not raw_dir.exists():
    print(f"  {raw_dir} does not exist at all!")
else:
    for t in TICKERS:
        tdir = raw_dir / t
        if tdir.exists():
            n_files = sum(1 for _ in tdir.rglob("*") if _.is_file())
            print(f"  {t}: folder exists, {n_files} file(s)")
        else:
            print(f"  {t}: NO FOLDER (not downloaded)")
    # case-insensitive check too, in case of a casing mismatch
    all_dirs = {p.name for p in raw_dir.iterdir() if p.is_dir()}
    print(f"  (total ticker folders in tinix_ocr: {len(all_dirs)})")
    for t in TICKERS:
        matches = [d for d in all_dirs if d.lower() == t.lower() and d != t]
        if matches:
            print(f"  {t}: case-mismatched folder(s) found: {matches}")

print("\n=== Stage 2: data/processed/chunks.jsonl ===")
chunks_path = Path("data/processed/chunks.jsonl")
if not chunks_path.exists():
    print(f"  {chunks_path} does not exist!")
else:
    counts = {t: 0 for t in TICKERS}
    all_tickers_seen = set()
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tk = rec.get("ticker")
            all_tickers_seen.add(tk)
            if tk in counts:
                counts[tk] += 1
    for t, c in counts.items():
        print(f"  {t}: {c} chunk(s) in chunks.jsonl")
    for t in TICKERS:
        matches = [tk for tk in all_tickers_seen if tk and tk.lower() == t.lower() and tk != t]
        if matches:
            print(f"  {t}: case-mismatched ticker label(s) in chunks.jsonl: {matches}")

print("\n=== Stage 3: live disclosure_chunks table ===")
from sqlalchemy import create_engine
import pandas as pd
from src.core import config

PG_URL = (
    f"postgresql+psycopg2://{config.PGVECTOR_USER}:{config.PGVECTOR_PASSWORD}"
    f"@{config.PGVECTOR_HOST}:{config.PGVECTOR_PORT}/{config.PGVECTOR_DB}"
)
pg_engine = create_engine(PG_URL)

ticker_list = ",".join(f"'{t}'" for t in TICKERS)
df = pd.read_sql(
    f"SELECT ticker, COUNT(*) AS n FROM disclosure_chunks WHERE ticker IN ({ticker_list}) GROUP BY ticker",
    pg_engine,
)
if df.empty:
    print("  0 rows for any of the 12 tickers, exact match.")
else:
    print(df.to_string(index=False))

print("\n  Case-insensitive ticker check (per-ticker ILIKE):")
with pg_engine.connect() as conn:
    from sqlalchemy import text
    for t in TICKERS:
        res = conn.execute(text("SELECT DISTINCT ticker FROM disclosure_chunks WHERE ticker ILIKE :t"), {"t": t}).fetchall()
        labels = [r[0] for r in res]
        if labels and labels != [t]:
            print(f"  {t}: found under different label(s): {labels}")
        elif not labels:
            pass  # already covered by the exact-match query above
