"""RQ7 metrics - scores a resolver's output (src/dialogue/context_resolver.py)
against the gold labels in data/eval/rq7_conversational_benchmark.jsonl.

Same zero-top-level-import discipline as src/evaluation/metrics.py (see that
module's docstring) - every function here is pure Python, testable without
Postgres/Ollama/torch running anywhere.

Called by scripts/run_rq7_eval.py (runner, not yet written - see
docs/PROGRESS.md task list) once per (baseline resolver x benchmark item),
then aggregated via context_resolver_summary().
"""


def entity_carryover_accuracy(records: list[dict], field: str = "predicted_ticker",
                               gold_field: str = "gold_ticker") -> float:
    """Fraction of records where the resolver's extracted entity (ticker, by
    default - same function scores fiscal_year/indicator too via the field
    args) exactly matches gold. None == None counts as a match (both
    correctly found nothing to carry over, e.g. a self_contained item with
    no entity at all)."""
    if not records:
        return 0.0
    correct = sum(1 for r in records if r.get(field) == r.get(gold_field))
    return correct / len(records)


def temporal_carryover_accuracy(records: list[dict]) -> float:
    return entity_carryover_accuracy(records, "predicted_year", "gold_year")


def indicator_carryover_accuracy(records: list[dict]) -> float:
    return entity_carryover_accuracy(records, "predicted_indicator", "gold_indicator")


def routing_accuracy_post_rewrite(records: list[dict]) -> float:
    """Records must carry `predicted_route` (classify_intent() run on the
    resolver's rewritten standalone_question) and `gold_route` (the real
    expected_branch from the ORIGINAL single-turn benchmark item the
    conversation was built from). This is the metric that most directly
    answers RQ7's practical question: does a bad rewrite send the question
    down the wrong branch entirely, not just phrase it awkwardly."""
    if not records:
        return 0.0
    correct = sum(1 for r in records if r.get("predicted_route") == r.get("gold_route"))
    return correct / len(records)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def token_f1(predicted: str, gold: str) -> float:
    """Token-overlap F1 between the resolver's rewritten standalone question
    and the gold standalone question - same style of metric as SQuAD's F1,
    adapted for whole-question rewriting rather than short-span extraction.
    Whitespace tokenization only (no stemming/normalization) - Vietnamese
    word segmentation is a much bigger dependency than this metric needs to
    take on for a same-language-same-script comparison."""
    pred_tokens = _tokenize(predicted)
    gold_tokens = _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_question(predicted: str, gold: str) -> bool:
    """Case/whitespace-insensitive exact string match - a much stricter
    check than token_f1, used to report what fraction of rewrites are
    PERFECT, not just close."""
    return " ".join(_tokenize(predicted)) == " ".join(_tokenize(gold))


def recall_at_k(records: list[dict], k: int = 1) -> float:
    """Fraction of records where the gold ticker appears among the first k
    tickers the resolver considered - only meaningful for a resolver that
    exposes a ranked candidate list (`predicted_ticker_candidates`); records
    without that key are scored against just `predicted_ticker` (k=1
    equivalent), so this still works for the 3 baselines that only ever
    produce one ticker guess."""
    if not records:
        return 0.0
    hits = 0
    for r in records:
        gold_ticker = r.get("gold_ticker")
        candidates = r.get("predicted_ticker_candidates")
        if candidates:
            hits += 1 if gold_ticker in candidates[:k] else 0
        else:
            hits += 1 if r.get("predicted_ticker") == gold_ticker else 0
    return hits / len(records)


def latency_stats(records: list[dict], field: str = "latency_ms") -> dict:
    """mean/p50/p95 latency across records - resolvers that call an LLM
    (LLMPromptOnlyResolver, FineTunedContextResolver) are expected to be
    orders of magnitude slower than the 2 non-model baselines; this is a
    real cost/quality tradeoff RQ7's write-up needs to report, not just
    accuracy in isolation."""
    values = sorted(r[field] for r in records if r.get(field) is not None)
    if not values:
        return {"mean": None, "p50": None, "p95": None, "n": 0}
    n = len(values)
    mean = sum(values) / n

    def _pct(p):
        idx = min(n - 1, int(round(p * (n - 1))))
        return values[idx]

    return {"mean": mean, "p50": _pct(0.50), "p95": _pct(0.95), "n": n}


def context_resolver_summary(records: list[dict]) -> dict:
    """Top-level aggregate for one resolver's full run over the RQ7
    benchmark. `records` is a list of per-item dicts each expected to carry:
    predicted_standalone/gold_standalone_question, predicted_ticker/
    gold_ticker, predicted_year/gold_year, predicted_indicator/
    gold_indicator, predicted_route/gold_route, latency_ms, error_category.
    Missing keys degrade gracefully (via .get()) rather than raising, since
    not every baseline populates every field (e.g. the 2 non-LLM baselines
    never produce `predicted_indicator`)."""
    by_category: dict = {}
    for r in records:
        cat = r.get("error_category", "unknown")
        by_category.setdefault(cat, []).append(r)

    f1_scores = [token_f1(r.get("predicted_standalone", ""), r.get("gold_standalone_question", ""))
                 for r in records]
    exact_matches = [exact_match_question(r.get("predicted_standalone", ""),
                                           r.get("gold_standalone_question", ""))
                     for r in records]

    per_category_routing_accuracy = {
        cat: routing_accuracy_post_rewrite(items) for cat, items in by_category.items()
    }

    return {
        "n": len(records),
        "ticker_carryover_accuracy": entity_carryover_accuracy(records),
        "year_carryover_accuracy": temporal_carryover_accuracy(records),
        "indicator_carryover_accuracy": indicator_carryover_accuracy(records),
        "routing_accuracy_post_rewrite": routing_accuracy_post_rewrite(records),
        "mean_token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "exact_match_rate": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        "recall_at_1": recall_at_k(records, k=1),
        "latency": latency_stats(records),
        "per_error_category_routing_accuracy": per_category_routing_accuracy,
        "per_error_category_n": {cat: len(items) for cat, items in by_category.items()},
    }
