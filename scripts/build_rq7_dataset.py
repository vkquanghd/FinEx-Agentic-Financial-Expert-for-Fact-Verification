"""Builds data/eval/rq7_conversational_benchmark.jsonl - the RQ7 (Conversational
Context Resolver) evaluation set - from the EXISTING, already-verified
single-turn data/eval/benchmark.jsonl. See
docs/Ghi_chu_thao_luan_phan_bien_2026-07-26.md section 8.10 for the design
decision this implements (chốt 2026-07-28).

"No invented ground truth" (this project's standing principle, applied here
too): every `gold_standalone_question` / `gold_route` / `gold_ticker` value
this script writes is copied VERBATIM from a real item already in
benchmark.jsonl (routing/rag_faithfulness/whatif_exact_match categories,
the ones that carry a `ticker` + `expected_branch`) - never invented. The
only thing this script constructs synthetically is the REDUCED/context-
dependent surface wording of the follow-up turn (e.g. rewriting "Công ty
kiểm toán của CSC năm 2025 là ai?" down to "Còn CSC thì sao?") - that
reduction is the entire object of study for RQ7, so it has to be built, not
found; it is not presented as a "real" observed conversation, only as a
synthetic-but-grounded-in-real-gold conversational benchmark, same honesty
standard as synthetic_causal_calibration() in src/evaluation/metrics.py.

Covers 8 error categories (from the RQ7 discussion, section 8's dataset
construction plan):
  1. pronoun_reference      - "nó" replaces a ticker already in context
  2. ellipsis               - "Còn năm <Y> thì sao?" (year-only follow-up)
  3. result_reference       - "Con số đó là bao nhiêu vậy?" (re-ask, "đó")
  4. subject_switch         - "Còn <OTHER_TICKER> thì sao?"
  5. return_to_old_topic    - 3-turn: A, unrelated C, back to A via "nó"
  6. self_contained         - control: already-standalone question mid-conversation
  7. noisy_irrelevant_history - standalone question after an UNRELATED turn
  8. self_correction        - "À không, ý tôi là <OTHER_TICKER> cơ."

Run:
    python3 scripts/build_rq7_dataset.py
    python3 -m unittest tests.test_dialogue.test_build_rq7_dataset -v
"""
import json
from pathlib import Path

BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "benchmark.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "rq7_conversational_benchmark.jsonl"

# Categories in the source benchmark that carry a `ticker` + `expected_branch`
# - the only ones usable as conversation building blocks (nhanh_a_vs_b/
# regime_ablation/causal_comparison/absa_eval items are excluded: some lack a
# plain `ticker` field, or the question text doesn't literally contain the
# ticker substring, which several of the 8 transforms below require).
SOURCE_CATEGORIES = {"routing", "rag_faithfulness", "whatif_exact_match"}


def load_source_items() -> list[dict]:
    items = []
    with BENCHMARK_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item["category"] in SOURCE_CATEGORIES and item.get("ticker"):
                items.append(item)
    return items


def _by_ticker(items: list[dict]) -> dict:
    grouped: dict = {}
    for item in items:
        grouped.setdefault(item["ticker"], []).append(item)
    return grouped


def _gold_fields(item: dict) -> dict:
    return {
        "gold_standalone_question": item["question"],
        "gold_route": item["expected_branch"],
        "gold_ticker": item["ticker"],
        "gold_source_benchmark_id": item["id"],
    }


def _turn(idx: int, utterance: str, source_id: str = None, is_reduced: bool = False) -> dict:
    return {"turn_idx": idx, "speaker": "user", "utterance": utterance,
            "source_benchmark_id": source_id, "is_reduced": is_reduced}


def build_pronoun_reference(a: dict, b: dict) -> dict:
    """Requires a['ticker'] == b['ticker'] and the ticker substring literally
    present in b's question text (true for every item here, since ticker
    extraction from these exact questions is how they got their `ticker`
    field in the first place)."""
    reduced = b["question"].replace(b["ticker"], "nó", 1)
    return {
        "error_category": "pronoun_reference",
        "turns": [_turn(0, a["question"], a["id"]), _turn(1, reduced, b["id"], is_reduced=True)],
        **_gold_fields(b),
        "notes": f"Turn 2 synthetically reduces {b['id']}'s ticker mention to 'nó'; gold is {b['id']} unchanged.",
    }


def build_ellipsis(a: dict, b: dict, year: int) -> dict:
    reduced = f"Còn năm {year} thì sao?"
    return {
        "error_category": "ellipsis",
        "turns": [_turn(0, a["question"], a["id"]), _turn(1, reduced, b["id"], is_reduced=True)],
        **_gold_fields(b),
        "notes": f"Turn 2 is a synthetic year-only ellipsis; gold is {b['id']} (real question mentioning {year}) unchanged.",
    }


def build_result_reference(a: dict) -> dict:
    """Single-item category: user re-asks for 'the number just given' in
    different words. Gold is simply A repeated - a legitimate, common
    conversational case (clarification/repeat), not a degenerate one."""
    reduced = "Con số đó là bao nhiêu vậy?"
    return {
        "error_category": "result_reference",
        "turns": [_turn(0, a["question"], a["id"]), _turn(1, reduced, a["id"], is_reduced=True)],
        **_gold_fields(a),
        "notes": f"Turn 2 asks to repeat/restate the number from turn 1; gold is {a['id']} unchanged.",
    }


def build_subject_switch(a: dict, b: dict) -> dict:
    """Requires a['ticker'] != b['ticker']. The new ticker IS stated
    explicitly here (unlike pronoun_reference) - this is the "còn HPG thì
    sao?" pattern from the discussion notes, distinct from ellipsis by
    switching the ENTITY, not the time period."""
    reduced = f"Còn {b['ticker']} thì sao?"
    return {
        "error_category": "subject_switch",
        "turns": [_turn(0, a["question"], a["id"]), _turn(1, reduced, b["id"], is_reduced=True)],
        **_gold_fields(b),
        "notes": f"Turn 2 switches subject from {a['ticker']} to {b['ticker']}; gold is {b['id']} unchanged.",
    }


def build_return_to_old_topic(a: dict, c: dict) -> dict:
    """3-turn: A (real, full) -> C (unrelated topic, real, full) -> back to
    A via pronoun. Requires a's ticker literally in a['question']."""
    back_to_a = a["question"].replace(a["ticker"], "nó", 1)
    return {
        "error_category": "return_to_old_topic",
        "turns": [
            _turn(0, a["question"], a["id"]),
            _turn(1, c["question"], c["id"]),
            _turn(2, back_to_a, a["id"], is_reduced=True),
        ],
        **_gold_fields(a),
        "notes": f"Turn 3 returns to {a['id']}'s topic after the unrelated {c['id']} interrupts; gold is {a['id']} unchanged.",
    }


def build_self_contained(a: dict, b: dict) -> dict:
    """Control group: turn 2 is ALREADY standalone (b's real question, no
    reduction at all) even though it follows a relevant turn 1 - tests that
    a resolver doesn't over-eagerly rewrite a question that didn't need it."""
    return {
        "error_category": "self_contained",
        "turns": [_turn(0, a["question"], a["id"]), _turn(1, b["question"], b["id"])],
        **_gold_fields(b),
        "notes": f"Turn 2 ({b['id']}) is already standalone and should pass through unchanged.",
    }


def build_noisy_irrelevant_history(c: dict, a: dict) -> dict:
    """Turn 1 is UNRELATED (different ticker/topic) noise; turn 2 is A's
    real, already-standalone question. Distinct from self_contained: here
    the preceding turn is irrelevant/noisy rather than a relevant
    continuation, testing resistance to false entity carryover from noise."""
    return {
        "error_category": "noisy_irrelevant_history",
        "turns": [_turn(0, c["question"], c["id"]), _turn(1, a["question"], a["id"])],
        **_gold_fields(a),
        "notes": f"Turn 1 ({c['id']}) is irrelevant noise; turn 2 ({a['id']}) is standalone and must resolve to itself, unaffected by the noise.",
    }


def build_self_correction(a: dict, b: dict) -> dict:
    """Requires a['ticker'] != b['ticker']. User states A, then explicitly
    corrects the subject to B mid-conversation."""
    correction = f"À không, ý tôi là {b['ticker']} cơ."
    return {
        "error_category": "self_correction",
        "turns": [_turn(0, a["question"], a["id"]), _turn(1, correction, b["id"], is_reduced=True)],
        **_gold_fields(b),
        "notes": f"Turn 2 is a synthetic self-correction from {a['ticker']} to {b['ticker']}; gold is {b['id']} unchanged.",
    }


def build_dataset(items: list[dict]) -> list[dict]:
    by_ticker = _by_ticker(items)
    same_ticker_pairs = [(g[i], g[j]) for g in by_ticker.values() if len(g) >= 2
                         for i in range(len(g)) for j in range(len(g)) if i != j]
    tickers_with_items = list(by_ticker.keys())
    diff_ticker_pairs = [(by_ticker[t1][0], by_ticker[t2][0])
                         for idx, t1 in enumerate(tickers_with_items)
                         for t2 in tickers_with_items[idx + 1:]]

    conversations = []
    conv_id = 0

    def _next_id(prefix):
        nonlocal conv_id
        conv_id += 1
        return f"rq7_{prefix}_{conv_id:03d}"

    # Only a handful of tickers have >=2 items in the source categories (4
    # tickers -> 8 ordered pairs, as of the 2026-07-28 benchmark.jsonl), so
    # categories 1/2/6 below deliberately draw from the SAME small pool of
    # same_ticker_pairs rather than disjoint slices of it - each category
    # builds a DIFFERENT synthetic conversation (different reduced wording,
    # different conv_id) around the same real (a, b) gold pair, which is
    # fine: the gold labels are still the real, unmodified benchmark items;
    # only the surface conversational framing differs per category.

    # 1. pronoun_reference - up to 8 pairs
    for a, b in same_ticker_pairs[:8]:
        if b["ticker"] not in b["question"]:
            continue
        conv = build_pronoun_reference(a, b)
        conv["conv_id"] = _next_id("pronoun")
        conversations.append(conv)

    # 2. ellipsis - pairs where b's question contains an explicit year
    import re
    year_re = re.compile(r"\b(20\d{2})\b")
    for a, b in same_ticker_pairs[:8]:
        m = year_re.search(b["question"])
        if not m or a["id"] == b["id"]:
            continue
        conv = build_ellipsis(a, b, int(m.group(1)))
        conv["conv_id"] = _next_id("ellipsis")
        conversations.append(conv)

    # 3. result_reference - single items, up to 8
    for a in items[:8]:
        conv = build_result_reference(a)
        conv["conv_id"] = _next_id("result_ref")
        conversations.append(conv)

    # 4. subject_switch - up to 8 diff-ticker pairs
    for a, b in diff_ticker_pairs[:8]:
        conv = build_subject_switch(a, b)
        conv["conv_id"] = _next_id("subj_switch")
        conversations.append(conv)

    # 5. return_to_old_topic - needs a 3rd unrelated item C
    count = 0
    for a, b in same_ticker_pairs:
        if count >= 8:
            break
        if a["ticker"] not in a["question"]:
            continue
        c_candidates = [it for it in items if it["ticker"] != a["ticker"]]
        if not c_candidates:
            continue
        c = c_candidates[count % len(c_candidates)]
        conv = build_return_to_old_topic(a, c)
        conv["conv_id"] = _next_id("return_topic")
        conversations.append(conv)
        count += 1

    # 6. self_contained - up to 8 same-ticker pairs (same reuse rationale as
    # categories 1/2 above - a different conversation shape, same real pool)
    for a, b in same_ticker_pairs[:8]:
        conv = build_self_contained(a, b)
        conv["conv_id"] = _next_id("self_contained")
        conversations.append(conv)

    # 7. noisy_irrelevant_history - up to 8 diff-ticker pairs
    for c, a in diff_ticker_pairs[8:16]:
        conv = build_noisy_irrelevant_history(c, a)
        conv["conv_id"] = _next_id("noisy_history")
        conversations.append(conv)

    # 8. self_correction - up to 8 diff-ticker pairs
    for a, b in diff_ticker_pairs[16:24]:
        conv = build_self_correction(a, b)
        conv["conv_id"] = _next_id("self_correction")
        conversations.append(conv)

    return conversations


def main():
    items = load_source_items()
    conversations = build_dataset(items)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    by_cat = {}
    for conv in conversations:
        by_cat[conv["error_category"]] = by_cat.get(conv["error_category"], 0) + 1
    print(f"Wrote {len(conversations)} conversations to {OUTPUT_PATH}")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
