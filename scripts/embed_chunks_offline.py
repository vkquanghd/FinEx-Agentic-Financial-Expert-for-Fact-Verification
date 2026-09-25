"""Run the heavy bge-m3 embedding step on Google Colab's free GPU instead of
your local machine (2026-07-23, added after pdf_disclosures crossed ~3GB
and kept growing from the ongoing manual-download work - the concern:
local chunking/embedding of an ever-larger corpus will eventually be too
slow/heavy for a laptop, especially since the project folder lives under
iCloud-synced ~/Desktop, see docs/PROGRESS.md for known slowdowns there).

Deliberately has ZERO dependency on your local Postgres/MySQL - only
imports src.retrieval.embeddings (sentence-transformers/torch), no
psycopg2/sqlalchemy/mysql-connector at all. This is what makes it portable
to Colab: Colab's cloud VM cannot reach a Postgres container running on
your Mac (no network path in, and exposing it would be a real security
issue for a personal project DB with no hardening) - so this script does
ONLY the GPU-bound part (embedding) and writes plain JSONL, leaving the
actual DB write to a separate, lightweight, DB-only step
(`python3 -m src.retrieval.vector_store --input <this-output> --precomputed`)
that you run back on your own machine afterwards - that final step is fast
(no model loading, just INSERTs), safe to run locally even on a modest
machine.

Colab usage:
  1. On your Mac: zip just what's needed (not the whole repo/venv/.git):
       zip -r for_colab.zip src/retrieval/embeddings.py src/retrieval/__init__.py \
           src/__init__.py src/core/__init__.py scripts/embed_chunks_offline.py \
           data/processed/chunks.jsonl requirements.txt
     Upload for_colab.zip to Google Drive (or directly to the Colab session).
  2. In a Colab notebook (Runtime -> Change runtime type -> GPU, the free
     T4 is plenty for this):
       !unzip for_colab.zip -d /content/thesis
       %cd /content/thesis
       !pip install -q sentence-transformers torch
       !python3 -m scripts.embed_chunks_offline --input data/processed/chunks.jsonl \
           --output data/processed/chunks_embedded.jsonl
     First call downloads bge-m3's ~2.2GB weights (fine on Colab's
     connection). Confirm the printed device line says "cuda", not "cpu" -
     if it says cpu, the runtime type wasn't actually switched to GPU.
  3. Download chunks_embedded.jsonl from Colab back to your Mac (it's the
     same size as chunks.jsonl plus ~4KB/chunk for the 1024 floats - for
     656,918 chunks that's a few GB, budget for it).
  4. On your Mac (DB step, no GPU/model needed at all):
       python3 -m src.retrieval.vector_store --input data/processed/chunks_embedded.jsonl --precomputed

Resumable like the rest of this pipeline's ingestion scripts: writes one
line at a time and flushes after every batch, and skips any
(ticker, source_file, page, chunk_index) already present in the OUTPUT
file on restart - safe if a Colab session disconnects mid-run (a real risk
on the free tier, sessions can be reclaimed), just re-run the same command
and it picks up where it left off without re-paying for already-embedded
chunks.

Usage: python3 -m scripts.embed_chunks_offline --input data/processed/chunks.jsonl --output data/processed/chunks_embedded.jsonl
"""
import argparse
import json
from pathlib import Path

from src.retrieval.embeddings import embed_texts


def load_jsonl(path: Path) -> list:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def already_embedded_keys(out_path: Path) -> set:
    """Keys already written to the output file from a previous (possibly
    interrupted) run of this script - NOT the live DB (this script never
    touches the DB), so this is purely about resuming this script itself."""
    keys = set()
    if not out_path.exists():
        return keys
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keys.add((r["ticker"], r["source_file"], r["page"], r["chunk_index"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="chunks.jsonl from src/retrieval/text_splitter.py")
    ap.add_argument("--output", required=True, help="Where to write chunks + embeddings (JSONL)")
    ap.add_argument("--batch-size", type=int, default=64,
                     help="Larger than the local default (32) since a Colab GPU handles bigger batches fine")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    records = load_jsonl(in_path)
    if not records:
        print(f"No records found in {in_path}")
        return

    done = already_embedded_keys(out_path)
    if done:
        before = len(records)
        records = [
            r for r in records
            if (r["ticker"], r["source_file"], r["page"], r["chunk_index"]) not in done
        ]
        print(f"Resuming: {before - len(records)}/{before} already in {out_path}, {len(records)} left")

    if not records:
        print("Nothing left to embed - output file is already complete.")
        return

    from src.retrieval.embeddings import EMBEDDING_DIM
    print(f"Embedding {len(records)} chunk(s) ({EMBEDDING_DIM}-dim)...")

    with out_path.open("a", encoding="utf-8") as out_f:
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i + args.batch_size]
            texts = [r["chunk_text"] for r in batch]
            vectors = embed_texts(texts, batch_size=args.batch_size)
            for r, vec in zip(batch, vectors):
                out_r = dict(r)
                out_r["embedding"] = vec
                out_f.write(json.dumps(out_r, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"  embedded {min(i + args.batch_size, len(records))}/{len(records)} (flushed)")

    print(f"Done. Download {out_path} back to your machine, then run:")
    print(f"  python3 -m src.retrieval.vector_store --input {out_path} --precomputed")


if __name__ == "__main__":
    main()
