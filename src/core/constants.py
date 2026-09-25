"""Constants & enums shared across the pipeline: report types, statement
types, sector list, and the curated "core" set of MS codes to prioritize
for QA generation / DAG lookup (see docs/data_plan.md for the full
rationale - derived from inspecting VNM's full 265-line statement).
"""

REPORT_TYPES = ("nam", "ban_nien", "quy")
STATEMENT_TYPES = ("balance_sheet", "income_statement", "cash_flow_indirect", "cash_flow_direct")

SECTORS = (
    "Tiêu dùng thiết yếu",
    "Bất động sản",
    "Công nghiệp chế biến",
    "Tiêu dùng không thiết yếu (Bán lẻ)",
)

# Level-1 aggregate line items only (Roman numeral / lettered subtotals).
# Deliberately excludes level-3 accounting detail (e.g. "- Nguyên giá",
# "- Giá trị hao mòn lũy kế") and the entire cash_flow_direct statement,
# which is confirmed empty for essentially all VN companies (verified on
# VNM: all 34 direct-method rows null - almost everyone reports indirect
# method only). Keeping the schema bounded to this list is what makes the
# DAG's "structured lookup" tractable within the 12-week scope.

CORE_MS_CODES = {
    "balance_sheet": {
        "100": "A. Tài sản ngắn hạn",
        "110": "I. Tiền và các khoản tương đương tiền",
        "120": "II. Đầu tư tài chính ngắn hạn",
        "130": "III. Các khoản phải thu ngắn hạn",
        "140": "IV. Hàng tồn kho",
        "150": "V. Tài sản sinh học ngắn hạn",       # sector-dependent (nông nghiệp)
        "160": "VI. Tài sản ngắn hạn khác",
        "200": "B. Tài sản dài hạn",
        "210": "I. Các khoản phải thu dài hạn",
        "220": "II. Tài sản cố định",
        "230": "III. Tài sản sinh học dài hạn",       # sector-dependent
        "240": "IV. Bất động sản đầu tư",             # sector-dependent (BĐS)
        "250": "V. Tài sản dở dang dài hạn",
        "260": "VI. Đầu tư tài chính dài hạn",
        "270": "VII. Tài sản dài hạn khác",
        "280": "TỔNG CỘNG TÀI SẢN",
        "300": "C. Nợ phải trả",
        "310": "I. Nợ ngắn hạn",
        "330": "II. Nợ dài hạn",
        "400": "D. Vốn chủ sở hữu",
        "440": "TỔNG CỘNG NGUỒN VỐN",
    },
    "income_statement": {
        "01": "Doanh thu bán hàng và cung cấp dịch vụ",
        "10": "Doanh thu thuần về bán hàng và cung cấp dịch vụ",
        "11": "Giá vốn hàng bán",
        "20": "Lợi nhuận gộp về bán hàng và cung cấp dịch vụ",
        "22": "Doanh thu hoạt động tài chính",
        "23": "Chi phí tài chính",
        "25": "Chi phí bán hàng",
        "26": "Chi phí quản lý doanh nghiệp",
        "30": "Lợi nhuận thuần từ hoạt động kinh doanh",
        "31": "Thu nhập khác",
        "32": "Chi phí khác",
        "40": "Lợi nhuận khác",
        "50": "Tổng lợi nhuận kế toán trước thuế",
        "51": "Chi phí thuế TNDN hiện hành",
        "52": "Chi phí thuế TNDN hoãn lại",
        "60": "Lợi nhuận sau thuế thu nhập doanh nghiệp",
        "70": "Lãi cơ bản trên cổ phiếu (EPS)",
        "71": "Lãi suy giảm trên cổ phiếu",
    },
    "cash_flow_indirect": {
        "20": "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
        "30": "Lưu chuyển tiền thuần từ hoạt động đầu tư",
        "40": "Lưu chuyển tiền thuần từ hoạt động tài chính",
        "50": "Lưu chuyển tiền thuần trong kỳ",
        "60": "Tiền và tương đương tiền đầu kỳ",
        "70": "Tiền và tương đương tiền cuối kỳ",
    },
    # cash_flow_direct intentionally omitted - see docstring above.
}

# Company/ticker names that signal a non-standard statement schema
# (banks, securities, insurance) - flagged during the 200-ticker screen.
# Banks confirmed 0/84 on standard indicators; securities/insurance showed
# partial data but should be spot-checked before mixing into the DAG schema.
FINANCE_SUBSECTOR_KEYWORDS = ("Ngân hàng", "Chứng khoán", "Bảo hiểm", "Bank")
