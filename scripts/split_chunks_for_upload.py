"""Split chunks.jsonl into smaller part files for reliable Google Drive
upload (2026-07-21: repeated browser uploads of the single 635MB
chunks.jsonl kept stalling partway - e.g. at 486MB, then 605MB on retry -
before finishing. Splitting into ~45MB parts makes each individual upload
small enough to complete reliably over an unstable connection. The paired
notebook (notebooks/colab_embed_chunks.ipynb) reads every
chunks_part*.jsonl file it finds in the Drive folder, so nothing else
needs to change on the Colab side - just upload the whole output folder
instead of one big file.

Splits by LINE count, never mid-line, so every part file is itself valid
JSONL - no risk of corrupting a record by cutting mid-byte the way the
raw-byte-truncated upload did.

Usage:
    python3 -m scripts.split_chunks_for_upload
    python3 -m scripts.split_chunks_for_upload --lines-per-part 30000  # smaller parts, flakier connection
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/chunks.jsonl")
    ap.add_argument("--output-dir", default="data/processed/chunks_parts")
    ap.add_argument("--lines-per-part", type=int, default=50000)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Not found: {in_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    part_idx = 0
    n_lines_in_part = 0
    out_f = None
    total = 0
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if out_f is None or n_lines_in_part >= args.lines_per_part:
                if out_f:
                    out_f.close()
                part_idx += 1
                part_path = out_dir / f"chunks_part{part_idx:03d}.jsonl"
                out_f = part_path.open("w", encoding="utf-8")
                n_lines_in_part = 0
            out_f.write(line if line.endswith("\n") else line + "\n")
            n_lines_in_part += 1
            total += 1
    if out_f:
        out_f.close()

    sizes = sorted(out_dir.glob("chunks_part*.jsonl"))
    total_mb = sum(p.stat().st_size for p in sizes) / 1e6
    print(f"Done. {total} lines split into {part_idx} part file(s) in {out_dir}/")
    print(f"Average part size: {total_mb / part_idx:.1f} MB")
    print(f"Upload the ENTIRE '{out_dir.name}' folder's contents to "
          f"MyDrive/thesis_embed/chunks_parts/ on Drive (one file at a time is fine - "
          f"each part is small enough to upload reliably on its own).")


if __name__ == "__main__":
    main()
