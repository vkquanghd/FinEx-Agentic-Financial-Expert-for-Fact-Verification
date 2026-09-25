"""VI vs EN retrieval+generation latency/token benchmark - added 2026-09-08
in response to the thesis defense being presented in English while the
corpus/system prompts are Vietnamese-first: "if everything is built around
Vietnamese, how much longer does an English question take to answer, in
tokens and in wall time?" This is the concrete measurement backing that
question for the thesis's evaluation section/slides.

Design (see docs/PROGRESS.md 2026-09-08 entry for the full write-up):
  - Uses data/eval/bilingual_latency_pairs.jsonl - each item is the SAME
    underlying question asked in Vietnamese and in English, so any VI/EN
    gap measured reflects the LANGUAGE, not a harder/easier question.
    2026-09-08 (updated same day per user request): tickers deliberately
    scoped to the 3 groups identified from real coverage queries that day -
    the 4 "mã chính" (AAV/KTS/AAH/ACV - each in BOTH the "đủ dữ liệu" and
    TiniX-confirmed groups, the ones being drilled into for the thesis),
    the rest of the "đủ dữ liệu" (full-coverage) group (AAA/VCB/CTD/AGM/
    AAS/AAT/CMC), and the one TiniX-confirmed ticker not already covered
    by the 4 chính (SLS) - see each item's "group" field and `notes`.
  - Deliberately a slim, standalone replica of src.retrieval.rag_pipeline's
    core retrieval+generation call (hybrid_search -> build_prompt ->
    llm_factory.generate_with_stats), NOT a call into rag_pipeline.answer()
    itself - answer() layers on temporal anchoring/regime rerank/doctype
    filtering/web fallback, none of which are relevant to a controlled
    language-cost comparison and would add uncontrolled variance between
    the VI and EN runs of the same pair. Same "diagnostic script, separate
    from the production pipeline, safe to delete" pattern as
    scripts/rank_within_year.py (see rag_pipeline.py's own comment on it).
  - Token counts come from Ollama's own prompt_eval_count/eval_count
    (via llm_factory.generate_with_stats(), added alongside this script) -
    real server-reported counts, not a char-count proxy.

Requires a live Postgres + Ollama stack (same prerequisite as
rag_pipeline.py itself) - NOT runnable in a sandbox with no such stack.

Usage:
    python3 -m scripts.bilingual_latency_benchmark
    python3 -m scripts.bilingual_latency_benchmark --pairs data/eval/bilingual_latency_pairs.jsonl \\
        --top-k 5 --output data/eval/runs/bilingual_latency_2026-09-08.json --chart
"""
import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

DEFAULT_PAIRS = "data/eval/bilingual_latency_pairs.jsonl"
DEFAULT_TOP_K = 5


def load_pairs(path: Path) -> list:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_one(question: str, ticker: str, top_k: int) -> dict:
    """One retrieval+generation call, timed and token-counted. Mirrors the
    "ticker given, no year, not-yet-empty candidate pool" branch of
    rag_pipeline.answer() closely enough for a fair VI/EN comparison
    (same retrieval function, same top_k), without pulling in that
    function's temporal/regime/doctype logic - see module docstring."""
    import time

    from src.core.lang_detect import detect_language
    from src.models import llm_factory
    from src.models.generation import SYSTEM_PROMPT, SYSTEM_PROMPT_EN, build_prompt
    from src.retrieval.vector_store import hybrid_search

    lang = detect_language(question)
    system_prompt = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT

    t0 = time.monotonic()
    chunks = hybrid_search(question, top_k=top_k, ticker=ticker)
    retrieval_s = time.monotonic() - t0

    prompt = build_prompt(question, chunks, lang=lang)
    stats = llm_factory.generate_with_stats(prompt, system=system_prompt)

    return {
        "question": question,
        "lang": lang,
        "n_chunks_retrieved": len(chunks),
        "retrieval_s": round(retrieval_s, 3),
        "answer": stats["response"],
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "generation_total_s": round(stats["total_duration_s"], 3),
        "generation_prompt_eval_s": round(stats["prompt_eval_s"], 3),
        "generation_eval_s": round(stats["eval_s"], 3),
        "generation_load_s": round(stats["load_duration_s"], 3),
        "end_to_end_s": round(retrieval_s + stats["total_duration_s"], 3),
    }


def summarize(results: list, lang: str) -> dict:
    subset = [r for r in results if r["lang"] == lang]
    if not subset:
        return {"n": 0}

    def mean_of(key):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    return {
        "n": len(subset),
        "mean_end_to_end_s": mean_of("end_to_end_s"),
        "median_end_to_end_s": (round(statistics.median(
            [r["end_to_end_s"] for r in subset]), 3) if subset else None),
        "mean_retrieval_s": mean_of("retrieval_s"),
        "mean_generation_total_s": mean_of("generation_total_s"),
        "mean_prompt_tokens": mean_of("prompt_tokens"),
        "mean_completion_tokens": mean_of("completion_tokens"),
        "mean_tokens_per_s": (
            round(sum(r["completion_tokens"] for r in subset if r.get("completion_tokens"))
                  / sum(r["generation_eval_s"] for r in subset if r.get("generation_eval_s")), 2)
            if any(r.get("generation_eval_s") for r in subset) else None
        ),
    }


def maybe_render_chart(vi_summary: dict, en_summary: dict, out_path: Path):
    """Best-effort bar chart (end-to-end latency + completion tokens, VI vs
    EN) for slides - guarded so a machine without matplotlib still gets the
    JSON/table output, just no PNG (fail-soft, same philosophy as
    hybrid_search's lexical-branch fallback elsewhere in this codebase)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[chart] matplotlib not installed - skipping PNG chart (pip install matplotlib --break-system-packages)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    langs = ["Vietnamese", "English"]

    latencies = [vi_summary.get("mean_end_to_end_s") or 0, en_summary.get("mean_end_to_end_s") or 0]
    axes[0].bar(langs, latencies, color=["#2c7fb8", "#f03b20"])
    axes[0].set_ylabel("Mean end-to-end latency (s)")
    axes[0].set_title("Latency: VI vs EN")
    for i, v in enumerate(latencies):
        axes[0].text(i, v, f"{v:.1f}s", ha="center", va="bottom")

    tokens = [vi_summary.get("mean_completion_tokens") or 0, en_summary.get("mean_completion_tokens") or 0]
    axes[1].bar(langs, tokens, color=["#2c7fb8", "#f03b20"])
    axes[1].set_ylabel("Mean completion tokens")
    axes[1].set_title("Output tokens: VI vs EN")
    for i, v in enumerate(tokens):
        axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[chart] saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--output", default=None,
                     help="default: data/eval/runs/bilingual_latency_<date>.json")
    ap.add_argument("--chart", action="store_true", help="also save a PNG bar chart next to --output")
    args = ap.parse_args()

    pairs = load_pairs(Path(args.pairs))
    print(f"Loaded {len(pairs)} VI/EN pairs from {args.pairs}")

    results = []
    for item in pairs:
        for lang_key in ("question_vi", "question_en"):
            q = item[lang_key]
            print(f"\n[{item['id']}/{lang_key}] {q}")
            r = run_one(q, item["ticker"], args.top_k)
            r["pair_id"] = item["id"]
            r["ticker"] = item["ticker"]
            results.append(r)
            print(f"  -> lang={r['lang']} chunks={r['n_chunks_retrieved']} "
                  f"prompt_tokens={r['prompt_tokens']} completion_tokens={r['completion_tokens']} "
                  f"end_to_end_s={r['end_to_end_s']}")

    vi_summary = summarize(results, "vi")
    en_summary = summarize(results, "en")

    print("\n" + "=" * 70)
    print("SUMMARY (mean over all pairs)")
    print("=" * 70)
    print(f"{'metric':<28}{'Vietnamese':<20}{'English':<20}")
    for label, key in [
        ("n items", "n"),
        ("mean end-to-end (s)", "mean_end_to_end_s"),
        ("median end-to-end (s)", "median_end_to_end_s"),
        ("mean retrieval (s)", "mean_retrieval_s"),
        ("mean generation (s)", "mean_generation_total_s"),
        ("mean prompt tokens", "mean_prompt_tokens"),
        ("mean completion tokens", "mean_completion_tokens"),
        ("mean tokens/s (decode)", "mean_tokens_per_s"),
    ]:
        print(f"{label:<28}{str(vi_summary.get(key)):<20}{str(en_summary.get(key)):<20}")

    if vi_summary.get("mean_end_to_end_s") and en_summary.get("mean_end_to_end_s"):
        delta_pct = (en_summary["mean_end_to_end_s"] - vi_summary["mean_end_to_end_s"]) / vi_summary["mean_end_to_end_s"] * 100
        print(f"\nEnglish end-to-end latency is {delta_pct:+.1f}% vs Vietnamese (same questions, translated pairs).")

    out_path = Path(args.output) if args.output else Path(
        f"data/eval/runs/bilingual_latency_{datetime.now().strftime('%Y-%m-%d')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "top_k": args.top_k,
        "pairs_file": args.pairs,
        "vi_summary": vi_summary,
        "en_summary": en_summary,
        "detail": results,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")

    if args.chart:
        maybe_render_chart(vi_summary, en_summary, out_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
