"""Đối chiếu keyword_faithfulness() (proxy rẻ, dùng cho toàn bộ 68 câu
`rag_faithfulness` trong run_eval.py) với llm_judge_faithfulness() (RAGAS-
style, dùng chính LLM cục bộ làm giám khảo - src/evaluation/metrics.py,
CHƯA từng được dùng thật trong bất kỳ lần chạy full eval nào tính tới
2026-08-08, xem docs/eval_run1_results.md/eval_run2_results.md's mục
"Hạn chế" - "Faithfulness vẫn là proxy từ khoá, chưa phải LLM-judge").

CHỈ chạy trên một MẪU (không phải toàn bộ 68 câu, để giữ thời gian chạy
hợp lý) - lấy TOÀN BỘ các câu bị keyword-proxy chấm 0 (nghi vấn âm tính giả,
đúng nguyên tắc đã áp dụng cho lần đọc tay eval run #1) CỘNG một mẫu cố
định (seed cố định, tái lập được) trong số các câu chấm 1.0 (nghi vấn dương
tính giả - bảng AUDITOR_ALIASES quá rộng, hoặc keyword trùng ngẫu nhiên).

QUAN TRỌNG: script này SINH LẠI câu trả lời từ đầu (gọi thật
rag_pipeline.answer()), KHÔNG dùng lại câu trả lời đã lưu trong report -
generation có thể sai khác đôi chút so với lần chạy full eval trước (không
có seed cố định cho LLM). Vì vậy với mỗi câu, kw-score và judge-score trong
file này LUÔN được tính trên CÙNG MỘT câu trả lời (nhất quán nội bộ), dù
không nhất thiết trùng khớp con số 0.7899 đã báo cáo ở run #2.

Cách lấy context để chấm: llm_judge_faithfulness(question, answer, context)
cần đúng đoạn trích đã dùng để sinh câu trả lời - nhưng
rag_pipeline.answer()'s return dict CỐ Ý không giữ lại chunk_text (chỉ giữ
metadata trích dẫn, xem generation.answer_with_citation()'s docstring) nên
không có sẵn. Script này monkeypatch `generation.build_prompt` để "chụp"
lại context THẬT đã được dùng, mà KHÔNG đổi hành vi thật của hàm (vẫn gọi
nguyên bản bên trong) - cùng kiểu monkeypatch run_eval.run_rag() đã dùng
cho apply_temporal_anchor (with_temporal_anchor=False).

Chạy (cần Postgres + Ollama thật):
    python3 scripts/llm_judge_faithfulness_check.py [--sample-ones N]

Kết quả ghi vào
data/eval/runs/llm_judge_faithfulness_check_<ngày>.json, không đè report
nào đã có.
"""
import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import keyword_faithfulness, llm_judge_faithfulness  # noqa: E402
from src.evaluation.run_eval import load_benchmark  # noqa: E402

BENCHMARK_PATH = ROOT / "data" / "eval" / "benchmark.jsonl"
RUN2_REPORT_PATH = ROOT / "data" / "eval" / "runs" / "report_2026-08-08_full_run2_final.json"
OUTPUT_PATH = ROOT / "data" / "eval" / "runs" / f"llm_judge_faithfulness_check_{date.today().isoformat()}.json"

SAMPLE_SEED = 42  # cố định để tái lập được mẫu "ones" mỗi lần chạy lại


def _select_sample_ids(n_sample_ones: int) -> tuple:
    """Trả về (zero_ids, one_ids) dựa trên report run #2 đã lưu - toàn bộ
    id bị chấm 0.0, cộng mẫu ngẫu nhiên (seed cố định) n_sample_ones id
    trong số các id chấm 1.0."""
    report = json.loads(RUN2_REPORT_PATH.read_text(encoding="utf-8"))
    detail = [d for d in report["rag"]["detail"] if d.get("category") == "rag_faithfulness"]
    zero_ids = [d["id"] for d in detail if d.get("keyword_faithfulness", {}).get("score") == 0.0]
    one_ids = [d["id"] for d in detail if d.get("keyword_faithfulness", {}).get("score") == 1.0]
    rng = random.Random(SAMPLE_SEED)
    sampled_ones = rng.sample(one_ids, min(n_sample_ones, len(one_ids)))
    return zero_ids, sampled_ones


def _write_report(detail: list) -> None:
    kw_scores = [d["keyword_score"] for d in detail if d.get("keyword_score") is not None]
    judge_scores = [d["judge_score"] for d in detail if d.get("judge_score") is not None]
    disagreements = [
        d for d in detail
        if d.get("keyword_score") is not None and d.get("judge_score") is not None
        and abs(d["keyword_score"] - d["judge_score"]) >= 0.5
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "note": "Đối chiếu keyword_faithfulness (proxy) vs llm_judge_faithfulness "
                        "(RAGAS-style) trên MẪU (14 câu chấm 0 + mẫu câu chấm 1.0), "
                        "SINH LẠI câu trả lời (không dùng lại report run #2). "
                        "Xem docstring đầu file.",
                "date": date.today().isoformat(),
                "n": len(detail),
                "avg_keyword_score": round(sum(kw_scores) / len(kw_scores), 4) if kw_scores else None,
                "avg_judge_score": round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None,
                "n_disagreements_ge_0.5": len(disagreements),
                "disagreement_ids": [d["id"] for d in disagreements],
                "detail": detail,
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-ones", type=int, default=10,
                     help="Số câu chấm 1.0 lấy mẫu thêm ngoài toàn bộ câu chấm 0.0 (mặc định 10)")
    args = ap.parse_args()

    from src.models import generation
    from src.retrieval import rag_pipeline

    zero_ids, one_ids = _select_sample_ids(args.sample_ones)
    sample_ids = set(zero_ids) | set(one_ids)
    print(f"Mẫu: {len(zero_ids)} câu chấm 0.0 (toàn bộ) + {len(one_ids)} câu chấm 1.0 "
          f"(mẫu ngẫu nhiên, seed={SAMPLE_SEED}) = {len(sample_ids)} câu.")

    items = {it["id"]: it for it in load_benchmark(BENCHMARK_PATH) if it["id"] in sample_ids}
    detail = []

    # Monkeypatch build_prompt để "chụp" context thật đã dùng để sinh câu trả
    # lời, không đổi hành vi (vẫn gọi bản gốc bên trong) - captured[0] giữ
    # context của lời gọi GẦN NHẤT (mỗi câu hỏi trong benchmark chỉ gọi
    # rag_pipeline.answer() đúng 1 lần/item nên không lẫn giữa các câu).
    captured = [None]
    original_build_prompt = generation.build_prompt

    def _capturing_build_prompt(question, chunks):
        prompt = original_build_prompt(question, chunks)
        # Tách lại đúng phần context đã build_prompt dùng, không tính toán
        # lại theo cách khác - context nằm giữa 2 marker cố định trong prompt.
        start = prompt.find("Các đoạn trích tài liệu:\n\n") + len("Các đoạn trích tài liệu:\n\n")
        end = prompt.find("\n\n---\n\nCâu hỏi:")
        captured[0] = prompt[start:end] if end > start else prompt
        return prompt

    generation.build_prompt = _capturing_build_prompt

    try:
        for i, (item_id, item) in enumerate(items.items(), start=1):
            print(f"  ... {i}/{len(items)}: {item_id}", end="\r")
            try:
                result = rag_pipeline.answer(item["question"], ticker=item.get("ticker"))
                answer_text = result.get("answer", "")
                context = captured[0] or ""
                kw = keyword_faithfulness(answer_text, item.get("expected_keywords", []))
                judge = llm_judge_faithfulness(item["question"], answer_text, context)
                detail.append({
                    "id": item_id, "question": item["question"],
                    "answer": answer_text,
                    "keyword_score": kw.get("score"),
                    "keyword_missing": kw.get("missing"),
                    "judge_score": judge.get("score"),
                    "judge_raw": judge.get("raw"),
                    "context_char_len": len(context),
                })
            except Exception as e:
                detail.append({"id": item_id, "question": item["question"],
                                "error": f"{type(e).__name__}: {e}"})
            _write_report(detail)  # checkpoint sau MỖI câu - bài học từ ablation script
    finally:
        generation.build_prompt = original_build_prompt

    print()
    kw_scores = [d["keyword_score"] for d in detail if d.get("keyword_score") is not None]
    judge_scores = [d["judge_score"] for d in detail if d.get("judge_score") is not None]
    print("=" * 60)
    print(f"avg_keyword_score = {sum(kw_scores)/len(kw_scores):.4f}" if kw_scores else "avg_keyword_score = N/A")
    print(f"avg_judge_score   = {sum(judge_scores)/len(judge_scores):.4f}" if judge_scores else "avg_judge_score = N/A")
    print("=" * 60)
    print(f"Đã ghi: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
