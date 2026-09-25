"""Bounded schema description for the RATIO_LOOKUP path - the financial_ratios
counterpart of src/structured/schema.py (which does the same job for
financial_line_items).

WHY THIS FILE EXISTS (2026-09-15): financial_ratios has existed in the DB
schema since 001_init_schema.sql, and src/ingestion/vietstock_etl.py's
`load_chi_so_tai_chinh()` already parses Vietstock "Chỉ số tài chính"
exports into it correctly (~214 real per-ticker files already sit in
data/raw/vietstock_exports/, confirmed by reading one directly - see
docs/PROGRESS.md's 2026-09-15 entry) - but NOTHING in src/router/rules.py
ever checked this table. The only code that referenced financial_ratios at
all was src/structured/text_to_sql_agent.py, the deliberately-unwired
Nhánh B ablation baseline (see that module's own docstring) - so a typed
question like "P/E của HPG là bao nhiêu?" could never be answered from real
data, even after ingestion. This file + ratio_lookup.py + the new
RATIO_LOOKUP branch in rules.py/query_router.py close that gap the same
way schema.py/query_builder.py/DAG already does for financial_line_items.

The 71 canonical `ratio_name` strings below are copied VERBATIM (not
paraphrased) from a real ingested file
(data/raw/vietstock_exports/VietstockFinance_Chi-so-tai-chinh_20260715-160746.xlsx,
ticker SLS) - `financial_ratios.ratio_name` is a free-text column filled
directly from column A of each Vietstock export row (see
vietstock_etl.parse_chi_so_tai_chinh()), so the alias table's values here
MUST match that exact text or every lookup silently returns "no such row"
even when the data exists. Same "no invented ground truth" discipline as
schema.py - every ratio_name here is copied from an observed real file,
never guessed from the 10-group taxonomy name alone.
"""
import re
import unicodedata

# alias (normalized) -> (ratio_group, ratio_name)
RATIO_ALIASES: dict[str, tuple[str, str]] = {}


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _register(alias: str, group: str, ratio_name: str):
    RATIO_ALIASES.setdefault(_normalize(alias), (group, ratio_name))


# Each entry: (canonical ratio_name AS STORED IN financial_ratios, ratio_group,
# [VI aliases...], [EN aliases...]). The canonical name itself is also
# auto-registered as an alias (so an exact-text question always matches).
_RATIOS = [
    # --- NHÓM CHỈ SỐ ĐỊNH GIÁ (valuation) ---
    # NOTE on "EPS" ambiguity (2026-09-15, found via classify_intent()
    # regression): "EPS" is ALSO a registered alias in
    # src/structured/schema.py -> financial_line_items ms_code 70 ("Lãi cơ
    # bản trên cổ phiếu"), a DIFFERENT number (the officially reported
    # basic EPS for one period) from this ratio's "4 quý gần nhất" (TTM)
    # figure. rules.py's classify_intent() checks DAG (step 6) BEFORE
    # RATIO_LOOKUP (step 6a) and only falls through to the ratio check when
    # DAG's resolve_indicator() found NO ms_code - so a bare "EPS của HPG"
    # question always resolves to the DAG/reported-EPS branch, never this
    # ratio. This is a deliberate precedence choice (the officially
    # reported line item wins over the derived/TTM ratio for an ambiguous
    # bare abbreviation), not a bug - documented here so it isn't
    # rediscovered as a mystery later.
    ("Thu nhập trên mỗi cổ phần của 4 quý gần nhất (EPS)", "NHÓM CHỈ SỐ ĐỊNH GIÁ",
     ["eps", "thu nhap tren moi co phan"], ["eps", "earnings per share"]),
    ("Giá trị sổ sách của cổ phiếu (BVPS)", "NHÓM CHỈ SỐ ĐỊNH GIÁ",
     ["bvps", "gia tri so sach co phieu", "gia tri so sach cua co phieu"],
     ["bvps", "book value per share"]),
    ("Chỉ số giá thị trường trên thu nhập (P/E)", "NHÓM CHỈ SỐ ĐỊNH GIÁ",
     ["p/e", "pe", "chi so gia tren thu nhap"], ["p/e", "price to earnings", "price earnings ratio"]),
    ("Chỉ số giá thị trường trên giá trị sổ sách (P/B)", "NHÓM CHỈ SỐ ĐỊNH GIÁ",
     ["p/b", "pb", "chi so gia tren gia tri so sach"], ["p/b", "price to book"]),
    ("Chỉ số giá thị trường trên doanh thu thuần (P/S)", "NHÓM CHỈ SỐ ĐỊNH GIÁ",
     ["p/s", "ps", "chi so gia tren doanh thu"], ["p/s", "price to sales"]),
    ("Tỷ suất cổ tức", "NHÓM CHỈ SỐ ĐỊNH GIÁ",
     ["ty suat co tuc"], ["dividend yield"]),
    ("Beta", "NHÓM CHỈ SỐ ĐỊNH GIÁ",
     ["beta", "he so beta"], ["beta"]),
    ("Giá trị doanh nghiệp trên lợi nhuận trước thuế và lãi vay (EV/EBIT)", "NHÓM CHỈ SỐ ĐỊNH GIÁ",
     ["ev/ebit", "gia tri doanh nghiep tren ebit"], ["ev/ebit", "enterprise value to ebit"]),

    # --- NHÓM CHỈ SỐ SINH LỢI (profitability) ---
    ("Tỷ suất lợi nhuận gộp biên", "NHÓM CHỈ SỐ SINH LỢI",
     ["ty suat loi nhuan gop bien", "bien loi nhuan gop"], ["gross margin ratio"]),
    ("Tỷ lệ lãi EBIT", "NHÓM CHỈ SỐ SINH LỢI",
     ["ty le lai ebit"], ["ebit margin"]),
    ("Tỷ lệ lãi EBITDA", "NHÓM CHỈ SỐ SINH LỢI",
     ["ty le lai ebitda"], ["ebitda margin"]),
    ("Tỷ suất sinh lợi trên doanh thu thuần", "NHÓM CHỈ SỐ SINH LỢI",
     ["ty suat sinh loi tren doanh thu thuan"], ["net profit margin ratio"]),
    ("Tỷ suất lợi nhuận trên vốn chủ sở hữu bình quân (ROEA)", "NHÓM CHỈ SỐ SINH LỢI",
     ["roea", "roe", "ty suat loi nhuan tren von chu so huu"], ["roea", "roe", "return on equity"]),
    ("Tỷ suất sinh lợi trên vốn dài hạn bình quân (ROCE)", "NHÓM CHỈ SỐ SINH LỢI",
     ["roce", "ty suat sinh loi tren von dai han"], ["roce", "return on capital employed"]),
    ("Tỷ suất sinh lợi trên tổng tài sản bình quân (ROAA)", "NHÓM CHỈ SỐ SINH LỢI",
     ["roaa", "roa", "ty suat sinh loi tren tong tai san"], ["roaa", "roa", "return on assets"]),

    # --- NHÓM CHỈ SỐ TĂNG TRƯỞNG (growth) ---
    ("Tăng trưởng doanh thu thuần", "NHÓM CHỈ SỐ TĂNG TRƯỞNG",
     ["tang truong doanh thu thuan", "tang truong doanh thu"], ["revenue growth", "net revenue growth"]),
    ("Tăng trưởng lợi nhuận gộp", "NHÓM CHỈ SỐ TĂNG TRƯỞNG",
     ["tang truong loi nhuan gop"], ["gross profit growth"]),
    ("Tăng trưởng lợi nhuận trước thuế", "NHÓM CHỈ SỐ TĂNG TRƯỞNG",
     ["tang truong loi nhuan truoc thue"], ["pretax profit growth"]),
    ("Tăng trưởng lợi nhuận sau thuế của CĐ công ty mẹ", "NHÓM CHỈ SỐ TĂNG TRƯỞNG",
     ["tang truong loi nhuan sau thue"], ["net profit growth"]),
    ("Tăng trưởng tổng tài sản", "NHÓM CHỈ SỐ TĂNG TRƯỞNG",
     ["tang truong tong tai san"], ["total assets growth"]),
    ("Tăng trưởng nợ dài hạn", "NHÓM CHỈ SỐ TĂNG TRƯỞNG",
     ["tang truong no dai han"], ["long term debt growth"]),
    ("Tăng trưởng nợ phải trả", "NHÓM CHỈ SỐ TĂNG TRƯỞNG",
     ["tang truong no phai tra"], ["total liabilities growth"]),
    ("Tăng trưởng vốn chủ sở hữu", "NHÓM CHỈ SỐ TĂNG TRƯỞNG",
     ["tang truong von chu so huu"], ["equity growth"]),
    ("Tăng trưởng vốn điều lệ", "NHÓM CHỈ SỐ TĂNG TRƯỞNG",
     ["tang truong von dieu le"], ["charter capital growth"]),

    # --- NHÓM CHỈ SỐ THANH KHOẢN (liquidity) ---
    ("Tỷ số thanh toán bằng tiền mặt", "NHÓM CHỈ SỐ THANH KHOẢN",
     ["ty so thanh toan bang tien mat"], ["cash ratio"]),
    ("Tỷ số thanh toán nhanh", "NHÓM CHỈ SỐ THANH KHOẢN",
     ["ty so thanh toan nhanh"], ["quick ratio"]),
    ("Tỷ số thanh toán nhanh (Đã loại trừ HTK, Phải thu ngắn hạn - Tham khảo)", "NHÓM CHỈ SỐ THANH KHOẢN",
     ["ty so thanh toan nhanh da loai tru htk"], ["quick ratio excluding receivables"]),
    # 2026-09-22: thêm "chi so ..." và dạng TRẦN "thanh toan hien hanh".
    # Phát hiện qua đợt quét cách diễn đạt: "Chỉ số thanh toán hiện hành của
    # CTD năm 2024?" rơi xuống RAG, vì alias duy nhất lúc đó bắt đầu bằng
    # "ty so" (Tỷ số) - người dùng nói "Chỉ số" thì trượt sạch. Dạng trần
    # phủ được cả hai cách gọi lẫn câu không có tiền tố nào, nhờ luật khớp
    # cụm-trọn-từ của resolve_ratio().
    ("Tỷ số thanh toán hiện hành (ngắn hạn)", "NHÓM CHỈ SỐ THANH KHOẢN",
     ["ty so thanh toan hien hanh", "chi so thanh toan hien hanh",
      "thanh toan hien hanh", "kha nang thanh toan hien hanh"],
     ["current ratio"]),
    ("Khả năng thanh toán lãi vay", "NHÓM CHỈ SỐ THANH KHOẢN",
     ["kha nang thanh toan lai vay"], ["interest coverage ratio"]),

    # --- NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG (efficiency) ---
    ("Vòng quay phải thu khách hàng", "NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG",
     ["vong quay phai thu khach hang"], ["receivables turnover"]),
    ("Thời gian thu tiền khách hàng bình quân", "NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG",
     ["thoi gian thu tien khach hang binh quan"], ["days sales outstanding"]),
    ("Vòng quay hàng tồn kho", "NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG",
     ["vong quay hang ton kho"], ["inventory turnover"]),
    ("Thời gian tồn kho bình quân", "NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG",
     ["thoi gian ton kho binh quan"], ["days inventory outstanding"]),
    ("Vòng quay phải trả nhà cung cấp", "NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG",
     ["vong quay phai tra nha cung cap"], ["payables turnover"]),
    ("Thời gian trả tiền khách hàng bình quân", "NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG",
     ["thoi gian tra tien khach hang binh quan"], ["days payable outstanding"]),
    ("Vòng quay tài sản cố định (Hiệu suất sử dụng tài sản cố định)", "NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG",
     ["vong quay tai san co dinh"], ["fixed asset turnover"]),
    ("Vòng quay tổng tài sản (Hiệu suất sử dụng toàn bộ tài sản)", "NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG",
     ["vong quay tong tai san"], ["total asset turnover"]),
    ("Vòng quay vốn chủ sở hữu", "NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG",
     ["vong quay von chu so huu"], ["equity turnover"]),

    # --- NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH (leverage) ---
    ("Tỷ số Nợ ngắn hạn trên Tổng nợ phải trả", "NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH",
     ["ty so no ngan han tren tong no phai tra"], ["short term debt to total liabilities"]),
    ("Tỷ số Nợ vay trên Tổng tài sản", "NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH",
     ["ty so no vay tren tong tai san"], ["debt to assets ratio"]),
    ("Tỷ số Nợ trên Tổng tài sản", "NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH",
     ["ty so no tren tong tai san"], ["total liabilities to assets"]),
    ("Tỷ số Vốn chủ sở hữu trên Tổng tài sản", "NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH",
     ["ty so von chu so huu tren tong tai san"], ["equity to assets ratio"]),
    ("Tỷ số Nợ ngắn hạn trên Vốn chủ sở hữu", "NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH",
     ["ty so no ngan han tren von chu so huu"], ["short term debt to equity"]),
    ("Tỷ số Nợ vay trên Vốn chủ sở hữu", "NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH",
     ["ty so no vay tren von chu so huu"], ["debt to equity ratio"]),
    ("Tỷ số Nợ trên Vốn chủ sở hữu", "NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH",
     ["ty so no tren von chu so huu"], ["total liabilities to equity"]),

    # --- NHÓM CHỈ SỐ DÒNG TIỀN (cash flow) ---
    ("Tỷ lệ dồn tích (Phương pháp Cân đối kế toán)", "NHÓM CHỈ SỐ DÒNG TIỀN",
     ["ty le don tich phuong phap can doi ke toan"], ["accrual ratio balance sheet method"]),
    ("Tỷ lệ dồn tích (Phương pháp Dòng tiền)", "NHÓM CHỈ SỐ DÒNG TIỀN",
     ["ty le don tich phuong phap dong tien"], ["accrual ratio cash flow method"]),
    ("Tỷ số dòng tiền HĐKD trên doanh thu thuần", "NHÓM CHỈ SỐ DÒNG TIỀN",
     ["ty so dong tien hdkd tren doanh thu thuan"], ["operating cash flow to revenue ratio"]),
    ("Dòng tiền từ HĐKD trên Tổng tài sản", "NHÓM CHỈ SỐ DÒNG TIỀN",
     ["dong tien tu hdkd tren tong tai san"], ["operating cash flow to assets"]),
    ("Dòng tiền từ HĐKD trên Vốn chủ sở hữu", "NHÓM CHỈ SỐ DÒNG TIỀN",
     ["dong tien tu hdkd tren von chu so huu"], ["operating cash flow to equity"]),
    ("Dòng tiền từ HĐKD trên Lợi nhuận thuần từ HĐKD", "NHÓM CHỈ SỐ DÒNG TIỀN",
     ["dong tien tu hdkd tren loi nhuan thuan"], ["operating cash flow to operating profit"]),
    ("Khả năng chi trả nợ ngắn hạn từ dòng tiền HĐKD", "NHÓM CHỈ SỐ DÒNG TIỀN",
     ["kha nang chi tra no ngan han tu dong tien hdkd"], ["short term debt coverage from operating cash flow"]),
    ("Khả năng chi trả nợ ngắn hạn từ lưu chuyển tiền thuần trong kỳ", "NHÓM CHỈ SỐ DÒNG TIỀN",
     ["kha nang chi tra no ngan han tu luu chuyen tien thuan"], ["short term debt coverage from net cash flow"]),
    ("Khả năng thanh toán nợ từ dòng tiền HĐKD", "NHÓM CHỈ SỐ DÒNG TIỀN",
     ["kha nang thanh toan no tu dong tien hdkd"], ["debt coverage from operating cash flow"]),
    ("Dòng tiền từ HĐKD trên mỗi cổ phần (CPS)", "NHÓM CHỈ SỐ DÒNG TIỀN",
     ["cps", "dong tien tren moi co phan"], ["cps", "cash flow per share"]),

    # --- CƠ CẤU CHI PHÍ (cost structure) ---
    ("Giá vốn hàng bán/Doanh thu thuần", "CƠ CẤU CHI PHÍ",
     ["gia von hang ban tren doanh thu thuan"], ["cogs to revenue ratio"]),
    ("Chi phí bán hàng/Doanh thu thuần", "CƠ CẤU CHI PHÍ",
     ["chi phi ban hang tren doanh thu thuan"], ["selling expense to revenue ratio"]),
    ("Chi phí quản lý doanh nghiệp/Doanh thu thuần", "CƠ CẤU CHI PHÍ",
     ["chi phi quan ly doanh nghiep tren doanh thu thuan"], ["admin expense to revenue ratio"]),
    ("Chi phí lãi vay/Doanh thu thuần", "CƠ CẤU CHI PHÍ",
     ["chi phi lai vay tren doanh thu thuan"], ["interest expense to revenue ratio"]),

    # --- CƠ CẤU TÀI SẢN NGẮN HẠN (current asset structure) ---
    ("Tài sản ngắn hạn/Tổng tài sản", "CƠ CẤU TÀI SẢN NGẮN HẠN",
     ["ty trong tai san ngan han tren tong tai san"], ["current assets to total assets ratio"]),
    ("Tiền/Tài sản ngắn hạn", "CƠ CẤU TÀI SẢN NGẮN HẠN",
     ["ty trong tien tren tai san ngan han"], ["cash to current assets ratio"]),
    ("Đầu tư tài chính ngắn hạn/Tài sản ngắn hạn", "CƠ CẤU TÀI SẢN NGẮN HẠN",
     ["ty trong dau tu tai chinh ngan han tren tai san ngan han"], ["short term investments to current assets"]),
    ("Phải thu ngắn hạn/Tài sản ngắn hạn", "CƠ CẤU TÀI SẢN NGẮN HẠN",
     ["ty trong phai thu ngan han tren tai san ngan han"], ["receivables to current assets ratio"]),
    ("Hàng tồn kho/Tài sản ngắn hạn", "CƠ CẤU TÀI SẢN NGẮN HẠN",
     ["ty trong hang ton kho tren tai san ngan han"], ["inventory to current assets ratio"]),
    ("Tài sản ngắn hạn khác/Tài sản ngắn hạn", "CƠ CẤU TÀI SẢN NGẮN HẠN",
     ["ty trong tai san ngan han khac tren tai san ngan han"], ["other current assets to current assets"]),

    # --- CƠ CẤU TÀI SẢN DÀI HẠN (long-term asset structure) ---
    ("Tài sản dài hạn/Tổng tài sản", "CƠ CẤU TÀI SẢN DÀI HẠN",
     ["ty trong tai san dai han tren tong tai san"], ["long term assets to total assets ratio"]),
    ("Tài sản cố định/Tổng tài sản", "CƠ CẤU TÀI SẢN DÀI HẠN",
     ["ty trong tai san co dinh tren tong tai san"], ["fixed assets to total assets ratio"]),
    ("Tài sản cố định hữu hình/Tài sản cố định", "CƠ CẤU TÀI SẢN DÀI HẠN",
     ["ty trong tai san co dinh huu hinh tren tai san co dinh"], ["tangible fixed assets to fixed assets"]),
    ("Tài sản thuê tài chính/Tài sản cố định", "CƠ CẤU TÀI SẢN DÀI HẠN",
     ["ty trong tai san thue tai chinh tren tai san co dinh"], ["finance lease assets to fixed assets"]),
    ("Tài sản vô hình/Tài sản cố định", "CƠ CẤU TÀI SẢN DÀI HẠN",
     ["ty trong tai san vo hinh tren tai san co dinh"], ["intangible assets to fixed assets"]),
    ("XDCBDD/Tài sản cố định", "CƠ CẤU TÀI SẢN DÀI HẠN",
     ["xdcbdd tren tai san co dinh"], ["construction in progress to fixed assets"]),
]

for _ratio_name, _group, _vi_aliases, _en_aliases in _RATIOS:
    _register(_ratio_name, _group, _ratio_name)  # exact text always matches
    for _alias in _vi_aliases + _en_aliases:
        _register(_alias, _group, _ratio_name)

# Convenience: every canonical ratio_name, for callers that want the full
# bounded list (e.g. building a benchmark, or an error message listing what
# IS supported).
ALL_RATIO_NAMES = [r[0] for r in _RATIOS]


def resolve_ratio(text: str):
    """Return (ratio_group, ratio_name, matched_alias) for the best match
    against RATIO_ALIASES, or None if nothing matches. Exact normalized
    match first, then the longest whole-token phrase match. Short aliases
    must not match inside unrelated words (for example 'pe' in 'happen')."""
    norm = _normalize(text)
    if norm in RATIO_ALIASES:
        group, name = RATIO_ALIASES[norm]
        return group, name, norm
    best = None
    for alias, (group, name) in RATIO_ALIASES.items():
        if alias and f" {alias} " in f" {norm} " and (best is None or len(alias) > len(best[2])):
            best = (group, name, alias)
    return best
