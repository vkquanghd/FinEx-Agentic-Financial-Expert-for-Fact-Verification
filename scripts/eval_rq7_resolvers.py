"""RQ7 - đo độ chính xác giải thực thể của các resolver KHÔNG cần LLM, theo
từng loại lỗi trong data/eval/rq7_conversational_benchmark.jsonl.

Vì sao cần script riêng (2026-08-14): RQ7 chưa từng được nối vào
src/evaluation/run_eval.py, nên 5 baseline trong src/dialogue/context_resolver.py
chưa bao giờ được chấm điểm trên chính bộ 64 câu đã dựng cho chúng. Script này
lấp khoảng trống đó cho các baseline chạy được offline, và cho con số
before/after của bản sửa SessionMemory 1-ô -> nhiều-ô.

CHỈ chạy 3 resolver không cần model/mạng:
  - no_history    : sàn dưới, bỏ qua lịch sử
  - raw_concat    : nối k lượt gần nhất
  - memory_aware  : MỚI 2026-08-14, dùng SessionMemory nhiều ô
(llm_prompt_only cần Ollama, finetuned_mt5_lora cần checkpoint - bỏ qua ở đây;
human_oracle là trần trên nên luôn 1.00 theo định nghĩa.)

Chấm cái gì: `predicted_ticker == gold_ticker`. Đây là chỉ số quyết định của
RQ7 - một câu viết lại nghe xuôi tai nhưng gắn sai mã sẽ gửi truy vấn tới sai
công ty, tệ hơn nhiều so với câu viết vụng mà đúng mã.

Chạy:
    python3 scripts/eval_rq7_resolvers.py
Không cần DB/LLM/mạng. Ghi kết quả vào data/eval/runs/rq7_resolvers_<ngày>.json
"""
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dialogue.context_resolver import (  # noqa: E402
    ConversationTurn,
    MemoryAwareResolver,
    NoHistoryResolver,
    RawConcatResolver,
    ResolvedQuery,
    SessionMemory,
    extract_entities,
)

BENCHMARK = ROOT / "data" / "eval" / "rq7_conversational_benchmark.jsonl"
OUT = ROOT / "data" / "eval" / "runs" / f"rq7_resolvers_{date.today().isoformat()}.json"

CATEGORIES = ["pronoun_reference", "ellipsis", "result_reference", "subject_switch",
              "return_to_old_topic", "self_contained", "noisy_irrelevant_history",
              "self_correction"]


def _build(item):
    turns = item["turns"]
    history = [ConversationTurn(turn_idx=t["turn_idx"], speaker=t["speaker"],
                                utterance=t["utterance"]) for t in turns[:-1]]
    memory = SessionMemory(session_id=item["conv_id"])
    for t in history:
        e = extract_entities(t.utterance)
        memory.update_from(
            ResolvedQuery(standalone_question=t.utterance,
                          ticker=e["ticker"], fiscal_year=e["year"]),
            turn_idx=t.turn_idx, utterance=t.utterance,
        )
    return history, memory, turns[-1]["utterance"]


def main():
    items = [json.loads(l) for l in BENCHMARK.open(encoding="utf-8") if l.strip()]
    resolvers = [NoHistoryResolver(), RawConcatResolver(), MemoryAwareResolver()]

    # tally[resolver][category] = [n_correct, n_total]
    tally = {r.name: defaultdict(lambda: [0, 0]) for r in resolvers}
    detail = []

    for it in items:
        history, memory, current = _build(it)
        gold = it["gold_ticker"]
        row = {"conv_id": it["conv_id"], "category": it["error_category"], "gold_ticker": gold}
        for r in resolvers:
            got = r.resolve(history, current, memory).ticker
            ok = (got == gold)
            t = tally[r.name][it["error_category"]]
            t[0] += int(ok); t[1] += 1
            row[r.name] = got
            row[r.name + "_ok"] = ok
        detail.append(row)

    # ---- bảng kết quả ----
    names = [r.name for r in resolvers]
    w = max(len(c) for c in CATEGORIES) + 2
    print("\nRQ7 - độ chính xác giải mã chứng khoán (predicted_ticker == gold_ticker)\n")
    print("Loại lỗi".ljust(w) + "".join(n.rjust(16) for n in names))
    print("-" * (w + 16 * len(names)))
    for c in CATEGORIES:
        line = c.ljust(w)
        for n in names:
            ok, tot = tally[n][c]
            line += f"{ok}/{tot}".rjust(16)
        print(line)
    print("-" * (w + 16 * len(names)))
    line = "TỔNG".ljust(w)
    overall = {}
    for n in names:
        ok = sum(tally[n][c][0] for c in CATEGORIES)
        tot = sum(tally[n][c][1] for c in CATEGORIES)
        overall[n] = round(ok / tot, 4) if tot else None
        line += f"{ok}/{tot} ({overall[n]:.2f})".rjust(16)
    print(line + "\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "date": date.today().isoformat(),
        "note": "RQ7 resolver comparison, offline resolvers only. Chấm theo "
                "predicted_ticker == gold_ticker.",
        "overall_accuracy": overall,
        "by_category": {n: {c: tally[n][c] for c in CATEGORIES} for n in names},
        "detail": detail,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
