"""Task #22, roadmap item 5 (2026-07-30): builds the FROZEN candidate list
for the ABSA pilot labeling round - "~100-200 paragraphs, diverse (20-40
per document type, many tickers, many years, both clear and ambiguous
passages)", with a marked ~30-50-paragraph subset for double-annotation
(inter-annotator agreement, see scripts/absa/compute_interannotator_
agreement.py).

Writes ONE JSONL file (data/eval/absa_pilot_candidates.jsonl) that both
scripts/absa_annotate_cli.py --from-file and compute_interannotator_
agreement.py read from - this is deliberate: annotators must label the
EXACT same chunks for the double-annotation subset to mean anything (a
second live query, even with identical --ticker/--doctype arguments, is not
guaranteed to return the same rows if disclosure_chunks changed in between
- see absa_annotate_cli.py's load_candidate_chunks_from_file() docstring).

Stratification (real, from disclosure_chunks - this part needs a live
Postgres connection, cannot be exercised in this sandbox):
1. Groups candidate chunks by doctype, targets MIN_PER_DOCTYPE..
   MAX_PER_DOCTYPE (20-40) per doctype, capped by --total.
2. Within each doctype, samples across as many DISTINCT (ticker,
   fiscal_year-from-source_file) combinations as available, rather than
   letting one ticker/year dominate a stratum - "many tickers, many years"
   from the roadmap.
3. "Clear vs ambiguous" - HEURISTIC PROXY ONLY, documented honestly as
   such: a chunk containing BOTH a positive-signal keyword AND a negative-
   signal keyword (Vietnamese financial terms, see _POSITIVE_KEYWORDS/
   _NEGATIVE_KEYWORDS below) is flagged is_candidate_ambiguous=True - a
   real, ground-truth notion of "ambiguous" can only come from actual
   disagreement between annotators, which is exactly what this pilot is
   for. This flag exists ONLY to make sure the pilot sample doesn't
   accidentally skew toward only the easiest, most one-sided passages -
   NOT to pre-judge which chunks are hard.
4. A random --double-annotate-n (default 40, within the roadmap's 30-50
   range) subset of the FULL sample is marked double_annotate=True,
   stratified across doctypes proportionally so the kappa computation
   isn't dominated by one document type.

Usage (run on your own machine - needs a live Postgres connection):
    python3 -m scripts.absa.sample_pilot_paragraphs
    python3 -m scripts.absa.sample_pilot_paragraphs --total 150 --double-annotate-n 40
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

MIN_PER_DOCTYPE = 20
MAX_PER_DOCTYPE = 40
DEFAULT_TOTAL = 150
DEFAULT_DOUBLE_ANNOTATE_N = 40
OUTPUT_PATH = Path("data/eval/absa_pilot_candidates.jsonl")

# Small, hand-picked Vietnamese financial-sentiment keyword lists - a
# heuristic proxy for "mixed signal" chunks only, see module docstring
# point 3. Deliberately small/manual (same spirit as pdf_ingest_router.py's
# _NOT_TICKER denylist) rather than a trained classifier - wrong here just
# means the pilot sample is slightly less diverse, not a labeling error.
_POSITIVE_KEYWORDS = ["tăng", "cải thiện", "tăng trưởng", "vượt kế hoạch", "khả quan", "tích cực", "hiệu quả"]
_NEGATIVE_KEYWORDS = ["giảm", "suy giảm", "thua lỗ", "rủi ro", "vi phạm", "chậm trễ", "tiêu cực", "sụt giảm"]

FETCH_CANDIDATES_SQL = """
    SELECT ticker, doctype, source_file, page, chunk_index, chunk_text, char_count
    FROM disclosure_chunks
    WHERE embedding IS NOT NULL AND char_count >= 200
    ORDER BY ticker, source_file, page, chunk_index
"""


def _fiscal_year_from_source_file(source_file: str):
    """Best-effort 4-digit year extraction from a filename - same simple
    regex approach as src/predictive's extract_year() elsewhere in this
    project, used ONLY to diversify sampling across years, not as an
    authoritative fiscal_year (that lives in pdf_documents)."""
    m = re.search(r"(20[12]\d)", source_file)
    return int(m.group(1)) if m else None


def is_candidate_ambiguous(text: str) -> bool:
    low = text.lower()
    has_pos = any(kw in low for kw in _POSITIVE_KEYWORDS)
    has_neg = any(kw in low for kw in _NEGATIVE_KEYWORDS)
    return has_pos and has_neg


def fetch_all_candidates() -> list:
    from src.retrieval.vector_store import get_connection
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(FETCH_CANDIDATES_SQL)
            rows = cur.fetchall()
    finally:
        conn.close()
    cols = ["ticker", "doctype", "source_file", "page", "chunk_index", "chunk_text", "char_count"]
    return [dict(zip(cols, row)) for row in rows]


def stratified_sample(all_chunks: list, total: int, seed: int = 42) -> list:
    """Returns a diverse subset of `all_chunks` - see module docstring for
    the stratification rules. Deterministic given the same input + seed."""
    rng = random.Random(seed)
    by_doctype = defaultdict(list)
    for c in all_chunks:
        by_doctype[c["doctype"]].append(c)

    doctypes = sorted(by_doctype.keys())
    n_doctypes = len(doctypes)
    if n_doctypes == 0:
        return []

    # Even split of `total` across doctypes first, clamped to [MIN, MAX]
    # per doctype and to what's actually available - remaining budget (if
    # any doctype has fewer candidates than its share) is NOT silently
    # redistributed to other doctypes beyond MAX_PER_DOCTYPE, keeping the
    # "20-40 per doctype" balance intentional rather than accidental.
    target_per_doctype = max(MIN_PER_DOCTYPE, min(MAX_PER_DOCTYPE, total // n_doctypes))

    selected = []
    for doctype in doctypes:
        pool = by_doctype[doctype]
        # Prefer diversity across (ticker, fiscal_year) within this doctype:
        # shuffle groups, then round-robin one chunk per group before
        # taking a second from any group - avoids one ticker/year dominating.
        by_group = defaultdict(list)
        for c in pool:
            key = (c["ticker"], _fiscal_year_from_source_file(c["source_file"]))
            by_group[key].append(c)
        for group_chunks in by_group.values():
            rng.shuffle(group_chunks)
        group_keys = list(by_group.keys())
        rng.shuffle(group_keys)

        n_target = min(target_per_doctype, len(pool))
        picked, round_idx = [], 0
        while len(picked) < n_target:
            progressed = False
            for key in group_keys:
                if round_idx < len(by_group[key]):
                    picked.append(by_group[key][round_idx])
                    progressed = True
                    if len(picked) >= n_target:
                        break
            if not progressed:
                break
            round_idx += 1
        selected.extend(picked)

    # Mix of ambiguous-flagged and clear-flagged within the final sample -
    # a proxy check only (see module docstring point 3), not enforced as a
    # hard ratio if one type is scarce in the actual corpus.
    return selected


def mark_double_annotate(sample: list, n_double: int, seed: int = 43) -> list:
    """Marks `n_double` chunks double_annotate=True, stratified
    proportionally across doctypes present in `sample` - returns a NEW list
    (does not mutate the input dicts in place, for testability)."""
    rng = random.Random(seed)
    by_doctype = defaultdict(list)
    for i, c in enumerate(sample):
        by_doctype[c["doctype"]].append(i)

    n_double = min(n_double, len(sample))
    doctypes = sorted(by_doctype.keys())
    n_doctypes = len(doctypes)
    per_doctype_target = n_double // n_doctypes if n_doctypes else 0
    remainder = n_double - per_doctype_target * n_doctypes

    chosen_indices = set()
    for i, doctype in enumerate(doctypes):
        idxs = by_doctype[doctype][:]
        rng.shuffle(idxs)
        take = per_doctype_target + (1 if i < remainder else 0)
        chosen_indices.update(idxs[:take])

    result = []
    for i, c in enumerate(sample):
        c2 = dict(c)
        c2["double_annotate"] = i in chosen_indices
        result.append(c2)
    return result


def build_pilot_candidates(total: int = DEFAULT_TOTAL, n_double: int = DEFAULT_DOUBLE_ANNOTATE_N,
                            seed: int = 42) -> list:
    all_chunks = fetch_all_candidates()
    for c in all_chunks:
        c["is_candidate_ambiguous"] = is_candidate_ambiguous(c["chunk_text"])
    sample = stratified_sample(all_chunks, total=total, seed=seed)
    return mark_double_annotate(sample, n_double=n_double, seed=seed + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=DEFAULT_TOTAL,
                     help=f"Tổng số đoạn trích trong pilot (mặc định {DEFAULT_TOTAL}, roadmap yêu cầu 100-200).")
    ap.add_argument("--double-annotate-n", type=int, default=DEFAULT_DOUBLE_ANNOTATE_N,
                     help=f"Số đoạn trích cần 2 người gán độc lập (mặc định {DEFAULT_DOUBLE_ANNOTATE_N}, "
                          f"roadmap yêu cầu 30-50).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    candidates = build_pilot_candidates(total=args.total, n_double=args.double_annotate_n, seed=args.seed)
    if not candidates:
        raise SystemExit("0 candidate chunk(s) - check disclosure_chunks has embedded rows.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_doctype = defaultdict(int)
    n_ambiguous = n_double = 0
    for c in candidates:
        by_doctype[c["doctype"]] += 1
        n_ambiguous += int(c["is_candidate_ambiguous"])
        n_double += int(c["double_annotate"])

    print(f"Wrote {len(candidates)} pilot candidate(s) to {args.output}")
    print(f"  by doctype: {dict(sorted(by_doctype.items()))}")
    print(f"  is_candidate_ambiguous=True (heuristic proxy, NOT ground truth): {n_ambiguous}/{len(candidates)}")
    print(f"  double_annotate=True: {n_double}/{len(candidates)}")
    print(f"\nNext steps:")
    print(f"  python3 -m scripts.absa_annotate_cli --from-file {args.output} --annotator-id <person_1>")
    print(f"  python3 -m scripts.absa_annotate_cli --from-file {args.output} --double-annotate-only "
          f"--annotator-id <person_2>")
    print(f"  python3 -m scripts.absa.compute_interannotator_agreement --candidates-file {args.output}")


if __name__ == "__main__":
    main()
