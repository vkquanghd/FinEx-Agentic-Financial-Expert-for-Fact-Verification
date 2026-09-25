"""Audit answer quality across ~50 real questions spanning every router
branch (RAG/DAG/DAG_TREND/WHATIF/CAUSAL/COMPARE/ABSA/AUDITOR_LOOKUP/nhanh_b)
and BOTH ticker-coverage tiers - 2026-09-12, requested by Quang: "rà soát
các câu trả lời lại 1 lần nữa... xem có câu nào trả lời được câu nào không
trả lời được, câu nào trả lời mà như không trả lời".

Why two tiers matter (see docs/PROGRESS.md 2026-09-12 "GAS không phải mã
ck?" entry for the full finding): CORE_16 (scripts/seed_companies.py) has
RAG coverage (PDFs ingested) but ZERO `financial_line_items` rows (no
Vietstock structured export was ever done for any of the 16) - confirmed by
scanning all 213 valid files in data/raw/vietstock_exports/ and finding none
of them are for a CORE_16 ticker. A SEPARATE ~205-ticker "quality panel"
(downloaded specifically for the causal panel, funded by the Vietstock Pro
subscription mentioned in docs/DSA_v6_Bo_sung_2026-08-04.md) has BOTH RAG
(via the TiniX OCR dataset - see scripts/tinix_ingest_adapter.py's
docstring) AND DAG coverage. Testing only CORE_16 (as most of this
session's earlier ad-hoc questions did) systematically over-samples the
"không đầy đủ" (incomplete) tier and under-samples the "đầy đủ" (complete)
tier - this script deliberately covers both, using tickers confirmed
present in each tier's actual files/DB, not guessed.

TIER_CORE16 = the 16 tickers in scripts.seed_companies.CORE_16 (RAG-only).
TIER_PANEL = a sample of the ~205-ticker quality panel, picked for sector
diversity from the tickers actually found by name-scanning
data/raw/vietstock_exports/*Bao-cao-tai-chinh*.xlsx on 2026-09-12: CTD
(construction - the ONE ticker with confirmed full 4/4 modality coverage as
of 2026-07-20, see scripts/list_partial_coverage_tickers.py), HSG (steel),
BMP (plastics), AAA (plastics), BWE (water utility), MBS (securities), BSR
(oil refining), ACV (airports). Deliberately AVOIDS VCB/VIB even though both
are confirmed loaded (added 2026-07-21) - they're banks, which use a
different statement template (Circular 200 doesn't apply), so
financial_ratios/calculator.py/the causal panel don't find the item_names
they expect for VCB/VIB (see docs/PROGRESS.md 2026-07-21 entry) - using a
non-bank ticker keeps this audit's "panel tier should just work" premise
clean, not muddied by a separate, already-documented bank-template gap.

Needs a live Postgres + Ollama - same as every other real-run script in
this project (run_eval.py, retry_failed_llm_items.py, etc.) - NOT runnable
from the sandbox that wrote this (no DB/Ollama there). Expect this to be
SLOW: every LLM-backed branch (RAG/WHATIF-with-rationale/ABSA/nhanh_b) has
been observed live taking 90-330s per call, and Ollama has been observed
unstable under sustained sequential load (OllamaNotReachable after ~7-8h in
one earlier session today) - budget several hours for a full 50-question
run, or use --only to pilot a handful first.

Resumable (same "safe to re-run" convention as
scripts/retry_failed_llm_items.py): writes one JSON object per line to
--out immediately after each question completes (flushed, not buffered),
and on a re-run SKIPS any id already present in --out - so a crash or an
Ollama timeout partway through never forces redoing already-completed
questions.

This script does NOT attempt to auto-judge "trả lời được / không trả lời
được / trả lời như không trả lời" - that is exactly the nuanced judgment
Quang asked for and it belongs to a human (or a follow-up review pass)
reading the actual answer text, not a hardcoded heuristic pretending to be
ground truth. The one heuristic field this script DOES add
(`looks_like_refusal`) is a coarse, clearly-labeled PRE-SORT aid only
(regex match on known refusal phrases already used verbatim elsewhere in
this codebase, e.g. NO_CONTEXT_MSG) - it flags candidates for the "không trả
lời được" bucket but says nothing about "trả lời như không trả lời", which
requires actually reading the text.

Usage:
    python -m scripts.audit_answer_quality
    python -m scripts.audit_answer_quality --out data/eval/answer_quality_audit.jsonl
    python -m scripts.audit_answer_quality --only 1,2,3          # pilot a few ids first
    python -m scripts.audit_answer_quality --category rag_core16 # run one category only
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# id, category, question, ticker_tier ("core16"|"panel"|"mixed"|"n/a"),
# expected_branch_hint (best guess for what SHOULD happen - not a hard
# assertion, just documents intent so a mismatch is easy to spot), note.
QUESTIONS = [
    # --- A. RAG cơ bản - CORE_16 (RAG-only tier, "không đầy đủ") ---
    dict(id=1, category="rag_core16", tier="core16", expected_branch="rag",
         question="Doanh thu thuần năm 2024 của VNM là bao nhiêu?"),
    dict(id=2, category="rag_core16", tier="core16", expected_branch="rag",
         question="Tổng tài sản của PNJ năm 2025 là bao nhiêu?"),
    dict(id=3, category="rag_core16", tier="core16", expected_branch="rag",
         question="Lợi nhuận sau thuế của MWG năm 2024 là bao nhiêu?"),
    dict(id=4, category="rag_core16", tier="core16", expected_branch="rag",
         question="Vốn chủ sở hữu của SAB năm 2024 là bao nhiêu?"),

    # --- A2. RAG cơ bản - PANEL (đầy đủ tier, đối chứng với A) ---
    dict(id=5, category="rag_panel", tier="panel", expected_branch="dag",
         question="Doanh thu thuần năm 2024 của CTD là bao nhiêu?",
         note="PANEL tier co DAG - ky vong route=dag, khong phai rag"),
    dict(id=6, category="rag_panel", tier="panel", expected_branch="dag",
         question="Tổng tài sản của HSG năm 2024 là bao nhiêu?"),
    dict(id=7, category="rag_panel", tier="panel", expected_branch="dag",
         question="Lợi nhuận sau thuế của BMP năm 2024 là bao nhiêu?"),
    dict(id=8, category="rag_panel", tier="panel", expected_branch="dag",
         question="Vốn chủ sở hữu của BWE năm 2024 là bao nhiêu?",
         note="Doi VCB -> BWE (2026-09-12): VCB la ngan hang, dung template BCTC khac "
              "(Circular 200 khong ap dung) - financial_ratios/causal khong tin cay cho VCB "
              "(xem PROGRESS.md 2026-07-21). BWE la DN thuong, an toan hon cho test DAG co ban."),

    # --- B. DAG explicit (nam+quy cu the) - PANEL (ky vong route=dag sach) ---
    dict(id=9, category="dag_panel", tier="panel", expected_branch="dag",
         question="Tổng tài sản của CTD quý 1 năm 2025 là bao nhiêu?"),
    dict(id=10, category="dag_panel", tier="panel", expected_branch="dag",
         question="Doanh thu năm 2024 của AAA là bao nhiêu?"),
    dict(id=11, category="dag_panel", tier="panel", expected_branch="dag",
         question="Lợi nhuận sau thuế của MBS năm 2024 là bao nhiêu?"),
    dict(id=12, category="dag_panel", tier="panel", expected_branch="dag",
         question="Nợ phải trả của BSR năm 2024 là bao nhiêu?"),

    # --- B2. DAG explicit - CORE_16 (ky vong DAG miss -> fallback RAG) ---
    dict(id=13, category="dag_core16_fallback", tier="core16", expected_branch="rag",
         question="Tổng tài sản của HPG quý 1 năm 2025 là bao nhiêu?",
         note="Da xac nhan song 2026-09-12: DAG khong co du lieu HPG, fallback RAG."),
    dict(id=14, category="dag_core16_fallback", tier="core16", expected_branch="rag",
         question="Nợ phải trả của GAS năm 2024 là bao nhiêu?"),
    dict(id=15, category="dag_core16_fallback", tier="core16", expected_branch="rag",
         question="Vốn chủ sở hữu của DGW năm 2024 là bao nhiêu?"),

    # --- C. DAG_TREND (khong neu nam, ngon ngu xu huong) - PANEL ---
    dict(id=16, category="trend_panel", tier="panel", expected_branch="dag_trend",
         question="Doanh thu của CTD biến động qua các năm gần đây thế nào?"),
    dict(id=17, category="trend_panel", tier="panel", expected_branch="dag_trend",
         question="Lợi nhuận sau thuế của ACV qua các năm ra sao?"),

    # --- C2. DAG_TREND - CORE_16 (doi chung, co the khong khop ms_code vi thieu DAG) ---
    dict(id=18, category="trend_core16", tier="core16", expected_branch="dag_trend or rag",
         question="Doanh thu của VNM biến động qua các năm gần đây thế nào?"),
    dict(id=19, category="trend_core16", tier="core16", expected_branch="dag_trend or rag",
         question="Tổng tài sản của HPG qua các năm ra sao?"),

    # --- D. WHATIF - PANEL (ky vong CO tinh duoc) ---
    dict(id=20, category="whatif_panel", tier="panel", expected_branch="whatif",
         question="Nếu COGS của CTD tăng 5% trong năm 2025 thì lợi nhuận gộp thay đổi thế nào?"),
    dict(id=21, category="whatif_panel", tier="panel", expected_branch="whatif",
         question="Giả sử doanh thu của HSG giảm 10% năm 2025, lợi nhuận gộp bị ảnh hưởng ra sao?"),
    dict(id=22, category="whatif_panel", tier="panel", expected_branch="whatif",
         question="Nếu giá thép đầu vào tăng 8%, biên lợi nhuận của HSG thay đổi thế nào?"),

    # --- D2. WHATIF - CORE_16 (doi chung, ky vong "khong tinh duoc") ---
    dict(id=23, category="whatif_core16", tier="core16", expected_branch="whatif",
         question="Nếu COGS của HPG tăng 5% năm 2025 thì lợi nhuận gộp thay đổi ra sao?",
         note="Da xac nhan song 2026-09-12: 'Khong tinh duoc: Thieu du lieu revenue/cogs'."),
    dict(id=24, category="whatif_core16", tier="core16", expected_branch="whatif",
         question="Giả sử giá vàng tăng 10%, biên lợi nhuận của PNJ bị ảnh hưởng thế nào?",
         note="Da xac nhan song 2026-09-12: cung loi 'khong tinh duoc'."),

    # --- E. CAUSAL (nhieu cap khai niem, gom ca negative-control) ---
    dict(id=25, category="causal", tier="panel(ca ca 349 quan sat)", expected_branch="causal",
         question="Đòn bẩy tài chính ảnh hưởng đến biên lợi nhuận ròng như thế nào?",
         note="Retest sau ban va 2026-09-12 (_format_causal them cau tra loi tu nhien)."),
    dict(id=26, category="causal", tier="panel", expected_branch="causal",
         question="Đòn bẩy tài chính có tác động đến vòng quay tài sản không?"),
    dict(id=27, category="causal_negative_control", tier="panel", expected_branch="causal",
         question="Khả năng thanh toán hiện hành có ảnh hưởng đến biên lợi nhuận gộp không?",
         note="Cap negative-control - ky vong KHONG bi gan nham trich dan Myers/Abdeljawad."),
    dict(id=28, category="causal", tier="panel", expected_branch="causal",
         question="Dòng tiền hoạt động trên doanh thu có bị ảnh hưởng bởi đòn bẩy tài chính không?"),

    # --- F. COMPARE (da dang tier) ---
    dict(id=29, category="compare", tier="mixed", expected_branch="compare",
         question="So sánh hiệu suất giá của VNM và HPG trong năm qua"),
    dict(id=30, category="compare", tier="mixed", expected_branch="compare",
         question="So sánh CTD với HSG, mã nào tăng giá tốt hơn?",
         note="Cau hoi CO hoi 'ma nao tot hon' - kiem tra co tra loi thang cau hoi nay khong hay "
              "chi in %. (Doi VCB->HSG 2026-09-12, xem ghi chu o cau 8.)"),
    dict(id=31, category="compare_gold", tier="core16", expected_branch="compare",
         question="So sánh PNJ và CTD, có kèm giá vàng"),

    # --- G. ABSA ---
    dict(id=32, category="absa", tier="core16", expected_branch="absa",
         question="Phân tích cảm xúc về kết quả kinh doanh và rủi ro của HPG"),
    dict(id=33, category="absa", tier="panel", expected_branch="absa",
         question="Phân tích cảm xúc về quản trị doanh nghiệp của CTD"),
    dict(id=34, category="absa", tier="core16", expected_branch="absa",
         question="Đánh giá cảm xúc tích cực hay tiêu cực về triển vọng tăng trưởng của VNM"),

    # --- H. AUDITOR_LOOKUP (nguoc: tu ten hang kiem toan ra danh sach cong ty) ---
    dict(id=35, category="auditor_lookup", tier="n/a", expected_branch="auditor_lookup",
         question="KPMG kiểm toán các công ty nào?"),
    dict(id=36, category="auditor_lookup", tier="n/a", expected_branch="auditor_lookup",
         question="Deloitte kiểm toán những công ty nào qua các năm?"),

    # --- I. Nhanh B / text-to-SQL (governance/AGM - vung da biet co bug nhanh_b_accuracy=0.0) ---
    dict(id=37, category="nhanh_b", tier="core16", expected_branch="dag or rag",
         question="Nghị quyết ĐHĐCĐ năm 2025 của VNM thông qua những nội dung gì?"),
    dict(id=38, category="nhanh_b", tier="core16", expected_branch="dag or rag",
         question="Cổ tức năm 2024 của SAB được ĐHĐCĐ quyết định là bao nhiêu?"),
    dict(id=39, category="nhanh_b", tier="panel", expected_branch="dag or rag",
         question="Hội đồng quản trị của CTD gồm những ai?"),

    # --- J. Cau hoi ghep nhieu phan (P3 - _extract_ticker/_PERCENT_RE chi bat 1 match) ---
    dict(id=40, category="compound_p3_bug", tier="mixed", expected_branch="?",
         question="So sánh doanh thu của VNM và HPG năm 2024, mã nào cao hơn?",
         note="Cau hoi ghep so sanh+dinh luong - dung de xem router chon nhanh nao."),
    dict(id=41, category="compound_p3_bug", tier="panel", expected_branch="whatif",
         question="Nếu COGS tăng 5% và doanh thu giảm 3% cùng lúc cho CTD năm 2025 thì lợi nhuận gộp ra sao?",
         note="2 kich ban % cung luc - kiem tra co chi bat % dau tien khong (P3 bug)."),
    dict(id=42, category="compound_p3_bug", tier="core16", expected_branch="?",
         question="Tổng tài sản của cả VNM, HPG và PNJ năm 2024 là bao nhiêu?",
         note="3 ma trong 1 cau - _extract_ticker chi bat mot mã dau tien theo thiet ke hien tai."),

    # --- K. Song ngu (English mirror cua vai cau tren) ---
    dict(id=43, category="bilingual_en", tier="panel", expected_branch="dag",
         question="What was CTD's revenue in 2024?"),
    dict(id=44, category="bilingual_en", tier="panel", expected_branch="causal",
         question="How does financial leverage affect net profit margin?",
         note="Nhanh causal hien CHUA co bilingual (chi tieng Viet) - ky vong tra loi Viet hoac loi."),
    dict(id=45, category="bilingual_en", tier="mixed", expected_branch="compare",
         question="Compare VNM and HPG's stock performance over the past year"),
    dict(id=46, category="bilingual_en", tier="core16", expected_branch="rag",
         question="What is HPG's total assets in Q1 2025?"),

    # --- L. Dinh tinh/rui ro phap ly (sparse coverage, da tier) ---
    dict(id=47, category="qualitative", tier="core16", expected_branch="rag",
         question="Rủi ro pháp lý của VNM là gì?"),
    dict(id=48, category="qualitative", tier="panel", expected_branch="rag",
         question="Rủi ro pháp lý của CTD là gì?"),
    dict(id=49, category="qualitative", tier="core16", expected_branch="rag",
         question="Chiến lược kinh doanh năm 2025 của MWG là gì?"),

    # --- M. Ngoai pham vi / adversarial (kiem tra tu choi dung cach) ---
    dict(id=50, category="out_of_scope", tier="n/a", expected_branch="out_of_scope",
         question="Giá cổ phiếu VNM ngày mai sẽ là bao nhiêu?",
         note="Du doan tuong lai - khong co can cu du lieu, ky vong tu choi chu khong doan. "
              "expected_branch sua tu 'rag' -> 'out_of_scope' 2026-09-14 sau khi them nhanh "
              "OUT_OF_SCOPE rieng (xem src/router/rules.py FUTURE_PREDICTION_MARKERS) - "
              "truoc do cau nay roi vao RAG va nhan huong dan sai ('thu neu ro ky/chi tieu')."),
    dict(id=51, category="out_of_scope", tier="n/a", expected_branch="out_of_scope",
         question="Warren Buffett nghĩ gì về HPG?",
         note="Hoan toan ngoai pham vi tai lieu da nap. expected_branch sua tu 'rag' -> "
              "'out_of_scope' 2026-09-14 (xem THIRD_PARTY_OPINION_PATTERNS) - phan hoi gio "
              "giai thich ro Warren Buffett la ai va vi sao khong lien quan HPG, thay vi "
              "cau 'khong tim thay trong tai lieu' chung chung truoc day."),

    # --- N. Ma khong ton tai trong he thong ---
    dict(id=52, category="unknown_ticker", tier="n/a", expected_branch="rag or dag",
         question="Doanh thu năm 2024 của XYZ123 là bao nhiêu?",
         note="Ma khong hop le - ky vong tu choi ro rang, KHONG bia so."),

    # --- O. Mo rong 2026-09-13 (52 -> 60 cau, theo yeu cau chay 1 lan tong the) ---
    # 8 cau moi, lap day dung nhung khoang trong ro rang cua bo 52 cau dau -
    # dung ticker DA XAC NHAN co trong du lieu (CORE_16/PANEL da dung o tren),
    # khong bia ticker/du lieu moi.
    dict(id=53, category="trend_panel_quarterly", tier="panel", expected_branch="dag",
         question="Doanh thu theo từng quý của BSR trong năm 2024 diễn biến ra sao?",
         note="Trend O CAP DO QUY trong CUNG 1 nam - khac voi id=16/17 (trend theo NAM, nhieu nam). "
              "SUA 2026-09-14: expected_branch tu 'dag_trend' -> 'dag' sau khi doc lai code - day "
              "la NHAM LAN cua chinh nguoi viet cau hoi (toi), khong phai bug he thong. rules.py's "
              "DAG_TREND CHI kich hoat khi KHONG co nam cu the ('not parsed.fiscal_year') - cau nay "
              "NEU RO nam 2024 nen ve ban chat khong bao gio khop dag_trend, VA DUNG NHU VAY: neu ep "
              "vao dag_trend thi se MAT do phan giai theo quy (answer_recent_years() chi dung QUY MO "
              "NAM). Phan giai thich theo quy that ra da co san qua duong 'dag' -> trend.enrich() "
              "(kich hoat khi period=='YEAR', tu dong noi them tung dong Q1..Q4 kem %QoQ/%YoY - xem "
              "trend.py's _quarter_lines()), hoac qua trend.answer_from_quarters() neu BSR khong co "
              "dong YEAR gop san. Tuc la NOI DUNG cau tra loi rat co the da du (co breakdown theo "
              "quy that), chi la NHAN branch 'dag' dung, khong phai 'dag_trend' nhu toi ghi nham - "
              "can doc answer that de xac nhan enrich() co thuc su chay (BSR co the thieu du lieu "
              "quy dang tin cay)."),
    dict(id=54, category="rag_core16", tier="core16", expected_branch="rag",
         question="Biên lợi nhuận gộp của MSN năm 2024 là bao nhiêu?",
         note="MSN la ticker CORE_16 CHUA duoc test trong 52 cau dau (chi VNM/PNJ/MWG/SAB/HPG/GAS/DGW "
              "da dung) - mo rong phu CORE_16, dong thoi 'bien loi nhuan gop' la chi tieu PHAI TU TINH "
              "(khong phai dong bao cao truc tiep) nen thu thach RAG hon 1 con so don thuan."),
    dict(id=55, category="causal_negative_control", tier="panel", expected_branch="causal",
         question="Vòng quay tài sản có ảnh hưởng đến khả năng thanh toán hiện hành không?",
         note="Cap negative-control THU HAI (khac cap o id=27: current_ratio->gross_margin) - "
              "them 1 cap khong ky vong co quan he nhan qua that su, tang do tin cay cho viec kiem "
              "tra confounding chua duoc mo hinh hoa trong toan bo pipeline causal."),
    dict(id=56, category="compare", tier="core16", expected_branch="compare",
         question="So sánh hiệu suất giá của VNM, MSN và KDC trong năm qua",
         note="So sanh 3 MA CUNG LUC (khong chi 2 nhu id=29/30/31) - kiem tra nhanh compare co xu ly "
              "dung >2 ticker mot luc khong, hay chi lay 2 ma dau/bo sot ma thu 3."),
    dict(id=57, category="bilingual_en", tier="core16", expected_branch="rag",
         question="What are HPG's legal risks?",
         note="To hop MOI chua test: song ngu EN + cau hoi dinh tinh/rui ro phap ly (id=43-46 chi "
              "test EN cho dag/causal/compare/rag-so-lieu, KHONG co EN cho cau hoi dinh tinh)."),
    dict(id=58, category="out_of_scope", tier="n/a", expected_branch="out_of_scope",
         question="Bạn nghĩ tôi có nên mua cổ phiếu HPG bây giờ không?",
         note="LOAI out_of_scope MOI: xin loi khuyen DAU TU CA NHAN (khac id=50 - du doan tuong lai; "
              "id=51 - hoi y kien nguoi khac) - day la ranh gioi an toan/phap ly quan trong (tu van "
              "dau tu ca nhan), ky vong he thong TU CHOI dua loi khuyen chu khong chi noi 'khong du "
              "du lieu' - can doc ky cau tra loi that de phan biet 2 kieu tu choi nay."),
    dict(id=59, category="unknown_ticker", tier="n/a", expected_branch="rag or dag",
         question="Tổng tài sản năm 2024 của VNMX là bao nhiêu?",
         note="Ma GAN GIONG ma that (VNM + hau to la 'X') - khac id=52 (ma hoan toan khong lien quan, "
              "XYZ123) - kiem tra he thong co NHAM sang VNM that (fuzzy match sai) hay tu choi dung "
              "vi VNMX khong ton tai."),
    dict(id=60, category="auditor_lookup", tier="n/a", expected_branch="auditor_lookup",
         question="EY (Ernst & Young) kiểm toán những công ty nào?",
         note="Hang kiem toan Big4 THU BA (khac KPMG id=35, Deloitte id=36) - da dang hoa auditor_lookup "
              "tu 2 len 3 muc, dong bo voi cac nhom cau hoi khac."),
]

_REFUSAL_PATTERNS = re.compile(
    r"không tìm thấy|không tính được|không ước lượng được|không phân tích được|"
    r"no information found|not found in|không xác định được|không đủ dữ liệu",
    re.IGNORECASE,
)


def _looks_like_refusal(answer_text: str) -> bool:
    """Coarse PRE-SORT heuristic only - see module docstring. Matches
    phrases already used verbatim by this codebase's own refusal messages
    (NO_CONTEXT_MSG, WHATIF's "Không tính được:", etc.), so a hit here is a
    reasonably reliable signal of an explicit refusal - but a MISS does NOT
    mean the answer is good; that still needs a human to read the text
    (exactly the "trả lời như không trả lời" case Quang flagged)."""
    return bool(_REFUSAL_PATTERNS.search(answer_text or ""))


def _branch_matches_expected(actual_branch: str, expected_branch: str) -> bool:
    """2026-09-14 - mechanical comparator added for scripts/score_audit_results.py
    (Quang: "bộ benchmark cần có... câu trả lời dự kiến rõ ràng để đối chiếu").
    `expected_branch` is sometimes an "X or Y" hint (e.g. "rag or dag" for
    unknown_ticker, "dag or rag" for nhanh_b - see QUESTIONS above) written
    BEFORE the actual behavior was verified live - split on " or " so either
    named branch counts as a match. This is a MECHANICAL routing check only
    (did the router pick the branch the question was designed to exercise) -
    it says nothing about whether the answer TEXT is actually good, which is
    the separate, still-manual "trả lời được / không trả lời được / trả lời
    như không trả lời" rubric in docs/answer_quality_audit_2026-09-12.md."""
    if not actual_branch or not expected_branch:
        return False
    allowed = {b.strip() for b in expected_branch.split(" or ")}
    return actual_branch in allowed


def run_one(q: dict) -> dict:
    from src.router.query_router import route

    t0 = time.time()
    record = {"id": q["id"], "category": q["category"], "tier": q["tier"],
              "question": q["question"], "expected_branch": q["expected_branch"],
              "note": q.get("note")}
    try:
        result = route(q["question"], with_rationale=False)
        elapsed = round(time.time() - t0, 1)
        answer = result.get("answer", "")
        raw = result.get("raw") or {}
        actual_branch = result.get("branch")
        record.update({
            "ok": True,
            "actual_branch": actual_branch,
            "fallback_from": result.get("fallback_from"),
            "branch_match": _branch_matches_expected(actual_branch, q["expected_branch"]),
            "intent_reason": result.get("intent_reason"),
            "answer": answer,
            "numeric_verification": raw.get("numeric_verification") if isinstance(raw, dict) else None,
            "elapsed_s": elapsed,
            "looks_like_refusal": _looks_like_refusal(answer),
        })
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        record.update({
            "ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": elapsed,
        })
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/answer_quality_audit.jsonl")
    ap.add_argument("--only", default=None, help="Comma-separated question ids to run, e.g. 1,2,3")
    ap.add_argument("--category", default=None, help="Run only questions in this category")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            print(f"[resume] {len(done_ids)} question(s) already in {out_path} - will skip them.")

    todo = QUESTIONS
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        todo = [q for q in todo if q["id"] in wanted]
    if args.category:
        todo = [q for q in todo if q["category"] == args.category]
    todo = [q for q in todo if q["id"] not in done_ids]

    print(f"Running {len(todo)} question(s), writing to {out_path} (append mode, flushed per question)...")
    with out_path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] id={q['id']} ({q['category']}): {q['question'][:70]}...")
            record = run_one(q)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            status = "ERROR" if not record.get("ok") else (
                "REFUSAL?" if record.get("looks_like_refusal") else "answered"
            )
            print(f"    -> {status} ({record['elapsed_s']}s)")

    print(f"\nDone. Results in {out_path} - read it back for the actual review pass "
          f"(trả lời được / không trả lời được / trả lời như không trả lời).")


if __name__ == "__main__":
    main()
