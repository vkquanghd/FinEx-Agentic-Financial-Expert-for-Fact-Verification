"""Converts data/eval/answer_quality_audit.jsonl (written by
scripts/audit_answer_quality.py) into a readable .xlsx report - 2026-09-12,
Quang asked for Excel instead of raw JSONL ("khá là khó đọc").

Safe to re-run any time, including WHILE audit_answer_quality.py is still
running in another terminal - reads whatever lines exist right now (each
line was already flushed immediately after that question finished, so a
partial file is never a half-written line), so you can regenerate the
Excel report to see progress without waiting for the full 52-question run
to finish.

Adds two BLANK columns ("Nhãn đánh giá" / "Ghi chú đánh giá") for the manual
3-way judgment (trả lời được / không trả lời được / trả lời mà như không
trả lời) described in docs/answer_quality_audit_2026-09-12.md §4 - this
script does NOT fill those in; that is deliberately a human (or a separate
review pass) reading the actual answer text, not a hardcoded heuristic.

Usage:
    python -m scripts.audit_results_to_excel
    python -m scripts.audit_results_to_excel --in data/eval/answer_quality_audit.jsonl --out data/eval/answer_quality_audit.xlsx
"""
import argparse
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

COLUMNS = [
    ("id", "ID", 6),
    ("category", "Nhóm", 20),
    ("tier", "Tầng dữ liệu", 14),
    ("question", "Câu hỏi", 45),
    ("expected_branch", "Nhánh kỳ vọng", 14),
    ("actual_branch", "Nhánh thực tế", 14),
    ("fallback_from", "Fallback từ", 12),
    ("ok", "Chạy OK?", 9),
    ("looks_like_refusal", "Có vẻ từ chối?", 13),
    ("numeric_verification_label", "Verify số liệu", 16),
    ("elapsed_s", "Thời gian (s)", 11),
    ("answer", "Câu trả lời / Lỗi", 70),
    ("intent_reason", "Lý do phân loại (router)", 45),
    ("note", "Ghi chú thiết kế câu hỏi", 35),
    ("nhan_danh_gia", "Nhãn đánh giá (điền tay)", 22),
    ("ghi_chu_danh_gia", "Ghi chú đánh giá (điền tay)", 40),
]

_FILL_ERROR = PatternFill("solid", start_color="FFF2CC", end_color="FFF2CC")  # nhạt vàng - lỗi chạy (OllamaNotReachable...)
_FILL_REFUSAL = PatternFill("solid", start_color="FCE4D6", end_color="FCE4D6")  # nhạt cam - có vẻ từ chối
_FONT = "Arial"


def _numeric_verification_label(record: dict) -> str:
    nv = record.get("numeric_verification")
    if not nv:
        return ""
    label = nv.get("label", "")
    claims = nv.get("claims") or []
    mismatches = [c for c in claims if not c.get("matched")]
    if label == "unverified_claim_mismatch" and mismatches:
        c = mismatches[0]
        extra = ""
        if c.get("closest_source_value") is not None:
            if c.get("diff_pct") is not None:
                extra = f" (gần nhất trong nguồn: {c['closest_source_value']:,.0f}, lệch {c['diff_pct']:.1f}%)"
            elif c.get("diff_points") is not None:
                extra = f" (gần nhất trong nguồn: {c['closest_source_value']}, lệch {c['diff_points']:.1f} điểm %)"
        else:
            extra = " (không có số cùng loại trong nguồn để so)"
        return f"{label}{extra}"
    return label


def load_records(in_path: Path) -> list:
    records = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda r: r["id"])
    return records


def build_workbook(records: list) -> Workbook:
    wb = Workbook()
    ws = wb.active
    ws.title = "Answer Quality Audit"

    header_font = Font(name=_FONT, bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", start_color="305496", end_color="305496")
    body_font = Font(name=_FONT, size=10)
    wrap = Alignment(vertical="top", wrap_text=True)

    for col_idx, (_, header, width) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.freeze_panes = "A2"

    for row_idx, record in enumerate(records, start=2):
        ok = record.get("ok", False)
        answer_or_error = record.get("answer") if ok else f"[LỖI] {record.get('error', '')}"
        row_values = {
            "id": record.get("id"),
            "category": record.get("category"),
            "tier": record.get("tier"),
            "question": record.get("question"),
            "expected_branch": record.get("expected_branch"),
            "actual_branch": record.get("actual_branch"),
            "fallback_from": record.get("fallback_from"),
            "ok": "OK" if ok else "LỖI",
            "looks_like_refusal": "CÓ" if record.get("looks_like_refusal") else ("" if ok else "n/a"),
            "numeric_verification_label": _numeric_verification_label(record),
            "elapsed_s": record.get("elapsed_s"),
            "answer": answer_or_error,
            "intent_reason": record.get("intent_reason"),
            "note": record.get("note"),
            "nhan_danh_gia": "",
            "ghi_chu_danh_gia": "",
        }
        for col_idx, (key, _, _) in enumerate(COLUMNS, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=row_values.get(key))
            cell.font = body_font
            cell.alignment = wrap
            if not ok:
                cell.fill = _FILL_ERROR
            elif record.get("looks_like_refusal"):
                cell.fill = _FILL_REFUSAL

    # Sheet tóm tắt: đếm theo nhóm/tầng, tách OK/lỗi/có-vẻ-từ-chối - đọc
    # nhanh trước khi vào từng dòng chi tiết.
    summary = wb.create_sheet("Tóm tắt")
    summary["A1"] = "Tổng số câu đã chạy:"
    summary["B1"] = len(records)
    summary["A1"].font = Font(name=_FONT, bold=True)
    summary["A2"] = "Chạy OK:"
    summary["B2"] = sum(1 for r in records if r.get("ok"))
    summary["A3"] = "Lỗi (crash/timeout):"
    summary["B3"] = sum(1 for r in records if not r.get("ok"))
    summary["A4"] = "Có vẻ từ chối (looks_like_refusal, chỉ là gợi ý thô):"
    summary["B4"] = sum(1 for r in records if r.get("ok") and r.get("looks_like_refusal"))
    for r in range(1, 5):
        summary.cell(row=r, column=1).font = Font(name=_FONT, bold=True)
    summary.column_dimensions["A"].width = 45
    summary.column_dimensions["B"].width = 12

    return wb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/eval/answer_quality_audit.jsonl")
    ap.add_argument("--out", dest="out_path", default="data/eval/answer_quality_audit.xlsx")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not in_path.exists():
        raise SystemExit(f"Không tìm thấy {in_path} - chạy scripts/audit_answer_quality.py trước.")

    records = load_records(in_path)
    wb = build_workbook(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    print(f"Đã ghi {len(records)} dòng vào {out_path}")


if __name__ == "__main__":
    main()
