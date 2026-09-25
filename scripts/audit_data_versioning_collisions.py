"""Đếm va chạm phiên bản BCTC thật trong corpus (2026-08-14).

Bối cảnh (docs/RQ_audit_truoc_khi_training_2026-08-14.md §8): `disclosures` và
`financial_line_items` có UNIQUE key KHÔNG chứa `audit_status`
(`uq_disclosure (ticker, fiscal_year, period, consolidation)`,
`uq_line_item (..., statement_type, ms_code, consolidation)`). Nếu cùng một
kỳ báo cáo được nạp hai lần với audit_status khác nhau (vd bản "Soát xét" nạp
trước, bản "Kiểm toán" nạp sau) thì `INSERT ... ON CONFLICT DO UPDATE` sẽ GHI
ĐÈ ÂM THẦM - bản trước biến mất khỏi DAG, dù nhánh RAG (`disclosure_chunks`,
khoá theo `pdf_document_id`) vẫn giữ CẢ HAI file PDF.

Vì `disclosures`/`financial_line_items` chỉ còn giữ trạng thái HIỆN TẠI (bản
bị ghi đè đã mất), ta KHÔNG thể đếm trực tiếp "đã từng ghi đè bao nhiêu lần"
từ hai bảng đó. Cách đếm gián tiếp nhưng kiểm chứng được: nhánh RAG
(`pdf_documents`) không ghi đè theo (ticker, fiscal_year, period) - mỗi
source_file là một hàng riêng - nên nếu CÙNG (ticker, fiscal_year, kỳ suy ra
từ tên file) có NHIỀU hơn 1 file với "mức độ kiểm toán" suy ra từ tên file
khác nhau, đó chính là bằng chứng cụ thể: nếu các file đó từng được nạp vào
DAG, ít nhất một bản audit_status đã bị ghi đè.

Giới hạn đã biết (nêu rõ để không quá tự tin vào con số):
  - Suy luận audit_tier/period_tier từ TÊN FILE, không phải nội dung PDF hay
    cột audit_status thật trong DB - là suy đoán tốt nhất (best-effort),
    không phải sự thật tuyệt đối.
  - Chỉ xét pdf_documents.doctype = 'financial_statement' (do
    scripts/pdf_ingest_router.py's guess_doctype() gán, dựa trên khớp chuỗi
    "baocaotaichinh" trong tên file) - các file tiếng Anh dùng từ khác (vd
    "...finalcialstatements_q2_2025.pdf") KHÔNG khớp, nên bị xếp vào 'other'
    và bị BỎ QUA ở đây -> con số dưới đây là CẬN DƯỚI (undercount), không
    phải toàn bộ.

Chạy (cần Postgres thật, xem src/core/db.py để cấu hình kết nối):
    python3 -m scripts.audit_data_versioning_collisions

Test không cần DB (kiểm tra logic phân loại + logic phát hiện va chạm bằng
dữ liệu giả lập và tên file thật lấy từ data/raw/pdf_disclosures/):
    python3 -m unittest tests.test_scripts.test_audit_data_versioning_collisions -v
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_MD = ROOT / "docs" / "data_versioning_collision_report.md"

# Thứ tự "mức độ kiểm toán" tăng dần - dùng để báo cáo bản nào có khả năng là
# bản "sau cùng, đáng tin hơn" khi có va chạm (không dùng để tự động chọn).
AUDIT_TIER_ORDER = {"unknown": 0, "unaudited_quarterly": 1, "reviewed": 2, "audited": 3}

_AUDITED_RE = re.compile(r"kiem.{0,3}toan|audited", re.IGNORECASE)
_REVIEWED_RE = re.compile(r"soat.{0,3}xet|reviewed", re.IGNORECASE)
_QUARTERLY_RE = re.compile(r"_q[1-4][_.]|_q[1-4]$|quy.?[1-4]", re.IGNORECASE)


def classify_audit_tier(filename: str) -> str:
    """Suy đoán mức độ kiểm toán từ tên file, theo đúng 3 nhãn trong
    infra/postgres/migrations/001_init_schema.sql's disclosures.audit_status
    comment ('Kiểm toán' / 'Soát xét' / 'Chưa kiểm toán'), cộng 'unknown' khi
    tên file không có tín hiệu nào.

    Quy ước thị trường VN (xem ví dụ thật trong
    tests/test_scripts/test_audit_data_versioning_collisions.py): BCTC quý
    hầu như KHÔNG BAO GIỜ được gắn nhãn "kiểm toán"/"soát xét" trong tên file
    - mặc định coi các file quý (đã nhận diện qua classify_period_tier) là
    'unaudited_quarterly' NẾU không có tín hiệu audited/reviewed rõ ràng.
    """
    if _AUDITED_RE.search(filename):
        return "audited"
    if _REVIEWED_RE.search(filename):
        return "reviewed"
    if classify_period_tier(filename) == "quy":
        return "unaudited_quarterly"
    return "unknown"


def classify_period_tier(filename: str) -> str:
    """Suy đoán loại kỳ báo cáo từ tên file: 'quy' (quý) / 'ban_nien' (6
    tháng) / 'nam' (năm, mặc định khi không có tín hiệu quý/6-tháng)."""
    low = filename.lower()
    if _QUARTERLY_RE.search(low):
        return "quy"
    if re.search(r"_6t_|6thang|6_thang|ban.?nien|bannien|_h1_|6months", low):
        return "ban_nien"
    return "nam"


def find_collisions(rows: list[dict]) -> list[dict]:
    """rows: list các dict có 'ticker', 'fiscal_year', 'source_file'.
    Trả về danh sách nhóm (ticker, fiscal_year, period_tier) có >=2 audit_tier
    KHÁC NHAU và ít nhất một tier khác 'unknown' - tách riêng khỏi DB để test
    được bằng dữ liệu giả lập, không cần Postgres.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        tier_a = classify_audit_tier(r["source_file"])
        tier_p = classify_period_tier(r["source_file"])
        key = (r["ticker"], r["fiscal_year"], tier_p)
        groups.setdefault(key, []).append({**r, "audit_tier": tier_a, "period_tier": tier_p})

    collisions = []
    for (ticker, fiscal_year, period_tier), items in groups.items():
        tiers = {i["audit_tier"] for i in items}
        if len(tiers) >= 2 and tiers != {"unknown"}:
            collisions.append({
                "ticker": ticker, "fiscal_year": fiscal_year, "period_tier": period_tier,
                "n_files": len(items), "audit_tiers_found": sorted(tiers, key=AUDIT_TIER_ORDER.get),
                "files": [i["source_file"] for i in items],
            })
    return sorted(collisions, key=lambda c: (c["ticker"], c["fiscal_year"]))


def main():
    import pandas as pd
    from src.core.db import engine

    lines = ["# Báo cáo va chạm phiên bản BCTC (audit_status không nằm trong UNIQUE key)", "",
             "Sinh bởi scripts/audit_data_versioning_collisions.py. "
             "Xem docstring đầu file để hiểu giới hạn của phương pháp đếm.", ""]

    # ---- 1. Bối cảnh: audit_status hiện tại trong `disclosures` (bản CÒN SỐNG,
    # không phải lịch sử - chỉ để đối chiếu với notebooks/01_data_inventory_eda.ipynb) ----
    dist = pd.read_sql(
        "SELECT audit_status, COUNT(*) AS n FROM disclosures GROUP BY audit_status "
        "ORDER BY n DESC", engine)
    lines += ["## 1. Phân bố audit_status hiện tại trong `disclosures` (bối cảnh)", "",
              "| audit_status | Số dòng |", "|---|---|"]
    for _, r in dist.iterrows():
        lines.append(f"| {r.audit_status or '(NULL)'} | {r.n} |")
    lines.append("")

    # ---- 2. Va chạm suy đoán từ tên file PDF trong pdf_documents ----
    pdfs = pd.read_sql(
        "SELECT ticker, fiscal_year, source_file FROM pdf_documents "
        "WHERE doctype = 'financial_statement' AND ticker IS NOT NULL "
        "AND fiscal_year IS NOT NULL", engine)
    rows = pdfs.to_dict("records")
    collisions = find_collisions(rows)

    lines += [f"## 2. Va chạm suy đoán từ tên file (trong {len(rows)} PDF "
              f"doctype='financial_statement')", "",
              f"Số nhóm (ticker, fiscal_year, kỳ) có >=2 mức kiểm toán khác nhau: "
              f"**{len(collisions)}**", ""]
    if collisions:
        lines += ["| Ticker | Năm | Kỳ | Số file | Mức kiểm toán tìm thấy | File |",
                  "|---|---|---|---|---|---|"]
        for c in collisions:
            lines.append(f"| {c['ticker']} | {c['fiscal_year']} | {c['period_tier']} "
                         f"| {c['n_files']} | {', '.join(c['audit_tiers_found'])} "
                         f"| {'; '.join(c['files'])} |")
    else:
        lines.append("Không tìm thấy nhóm nào theo phương pháp suy đoán tên file này "
                     "(xem giới hạn ở §0 docstring - đây là cận dưới, không phải bằng "
                     "chứng 'chưa từng xảy ra').")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nĐã ghi {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
