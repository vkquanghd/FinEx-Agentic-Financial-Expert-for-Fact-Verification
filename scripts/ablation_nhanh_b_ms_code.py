"""Ablation (2026-08-07) - "cấp bảng ms_code vào prompt Nhánh B rồi chạy lại
đúng 27 câu `nhanh_a_vs_b`" (không chạy lại toàn bộ benchmark).

Bối cảnh (docs/eval_run1_results.md §"RQ2 - Nhánh B thất bại THẦM LẶNG
27/27"): lần chạy eval đầy đủ đầu tiên cho thấy Nhánh A = 1.00 accuracy,
Nhánh B = 0.00 - cả 27 câu SQL Nhánh B sinh ra đều hợp lệ, chạy không lỗi,
nhưng trả về 0 dòng vì đoán sai NGUYÊN VĂN chuỗi `item_name` (vd 'Doanh thu
thuần' thay vì tên chính thức 'Doanh thu thuần về bán hàng và cung cấp dịch
vụ'). File đó đã đề xuất ablation này để phân biệt 2 giả thuyết:

  (a) "bảng alias có kiểm soát (Nhánh A dùng) mới là thành phần chịu lực"
      -> nếu Nhánh B cũng được cấp đúng bảng đó, accuracy sẽ tăng mạnh.
  (b) "bản thân LLM tự do sinh SQL kém khả năng, cấp thêm ngữ cảnh cũng
      không cứu được" -> accuracy vẫn thấp dù có bảng ms_code.

Script này CHỈ chạy 27 câu category="nhanh_a_vs_b" (không phải toàn bộ
140+ câu benchmark) với `with_ms_code_hint=True`
(src/structured/text_to_sql_agent.py's answer()), dùng LẠI nguyên
run_nhanh_a_vs_b()/nhanh_a_vs_b_summary() đã dùng cho eval run #1 để số
liệu so sánh được trực tiếp, apples-to-apples - không viết logic chấm điểm
riêng.

Chạy (cần Postgres + Ollama thật, không chạy được trong sandbox này):
    python3 scripts/ablation_nhanh_b_ms_code.py

Kết quả ghi vào data/eval/runs/ablation_ms_code_hint_<ngày>.json, KHÔNG ghi
đè bất kỳ file report nào đã có (cùng nguyên tắc "lưu hết kết quả" đã áp
dụng cho eval run #1 - xem docs/eval_run1_results.md).
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Chạy trực tiếp bằng `python3 scripts/ablation_nhanh_b_ms_code.py` (không
# phải `python3 -m ...`) không tự thêm ROOT vào sys.path -> "ModuleNotFoundError:
# No module named 'src'". Cùng convention với mọi script khác trong thư mục
# này (scripts/init_db.py, scripts/seed_companies.py, ...).
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import nhanh_a_vs_b_summary  # noqa: E402
from src.evaluation.run_eval import load_benchmark, run_nhanh_a_vs_b  # noqa: E402

BENCHMARK_PATH = ROOT / "data" / "eval" / "benchmark.jsonl"
OUTPUT_PATH = ROOT / "data" / "eval" / "runs" / f"ablation_ms_code_hint_{date.today().isoformat()}.json"

# Số liệu baseline (with_ms_code_hint=False) từ eval run #1, 2026-08-07 -
# xem docs/eval_run1_results.md - chép cứng ở đây CHỈ để in ra bảng so sánh
# ngay khi ablation chạy xong, KHÔNG dùng cho bất kỳ phép tính nào.
BASELINE_NHANH_B_ACCURACY = 0.0
BASELINE_SUCCESS_ZERO_ROWS_RATE = 1.0


def _write_report(detail: list, metrics: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "note": "Ablation with_ms_code_hint=True - chỉ 27 câu nhanh_a_vs_b, "
                        "không phải full benchmark. So sánh với eval run #1 "
                        "(docs/eval_run1_results.md), KHÔNG thay thế nó.",
                "date": date.today().isoformat(),
                "with_ms_code_hint": True,
                "metrics": metrics,
                "detail": detail,
            },
            # default=str: cột NUMERIC của Postgres trả về Decimal qua
            # SQLAlchemy (nhanh_a_value/nhanh_b_value/rows) - json không tự
            # serialize được Decimal. Cùng cách run_eval.py's own report
            # writer xử lý (dòng ~682) - giữ nhất quán, không tự chế cách
            # khác.
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )


def main():
    items = load_benchmark(BENCHMARK_PATH)
    n_items = sum(1 for it in items if it["category"] == "nhanh_a_vs_b")
    print(f"Chạy ablation Nhánh B (with_ms_code_hint=True) trên {n_items} câu "
          f"category='nhanh_a_vs_b' ...")

    # Checkpoint sau MỖI câu (không chỉ ghi 1 lần ở cuối) - bài học từ lần
    # chạy đầu: json.dumps() thiếu default=str làm crash NGAY TRƯỚC bước ghi
    # file, mất trắng ~7 phút gọi LLM/DB dù mọi câu đã chạy xong thành công.
    # Cùng nguyên tắc checkpoint_nhanh_a_vs_b() trong run_eval.py's main().
    def on_progress(detail):
        print(f"  ... {len(detail)}/{n_items}", end="\r")
        try:
            _write_report(detail, nhanh_a_vs_b_summary(detail))
        except Exception as e:
            # Không để lỗi ghi checkpoint làm hỏng cả lượt chạy đang dở -
            # bản ghi cuối cùng (sau on_progress cuối, ngoài vòng lặp) vẫn
            # sẽ ghi đè lại nếu checkpoint giữa chừng này lỗi.
            print(f"\n  (checkpoint ghi lỗi, bỏ qua, tiếp tục chạy: {e})")

    result = run_nhanh_a_vs_b(items, on_progress=on_progress, with_ms_code_hint=True)
    print()

    m = result["metrics"]
    _write_report(result["detail"], m)

    print("=" * 60)
    print("SO SÁNH nhanh_b_accuracy:")
    print(f"  Trước ablation (eval run #1, prompt gốc): {BASELINE_NHANH_B_ACCURACY}")
    print(f"  Sau ablation (có bảng ms_code):           {m.get('nhanh_b_accuracy')}")
    print("SO SÁNH success_zero_rows_rate:")
    print(f"  Trước ablation: {BASELINE_SUCCESS_ZERO_ROWS_RATE}")
    print(f"  Sau ablation:   {m.get('success_zero_rows_rate')}")
    print("=" * 60)
    print(f"Đã ghi: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
