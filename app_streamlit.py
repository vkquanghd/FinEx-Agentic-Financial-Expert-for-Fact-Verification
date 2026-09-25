"""Streamlit demo UI for the DAG (structured lookup) path - System S2 in the
thesis's 3-system comparison (S1 RAG-only / S2 DAG-only / S3 Router).

Run with:
    streamlit run app_streamlit.py

Five tabs:
  - "Hoi bang cau hoi": free-text question, shows exactly what got parsed
    (ticker/year/period/indicator) alongside the answer - useful for the
    thesis's error analysis (why a question does/doesn't resolve).
  - "Chon truc tiep": guided dropdowns (company/period/indicator),
    guaranteed to hit a valid lookup if the data is loaded - useful for a
    live demo where free-text parsing might miss.
  - "Du lieu da nap": overview of what's actually in the DB right now
    (row counts per ticker), so you can see at a glance which companies
    are ready to query.
  - "What-if / Decomposition": exact bridge decomposition + ceteris-paribus
    sensitivity (src/structured/calculator.py) - includes a macro-grounded
    variant using real industry indices (src/causal/macro_lookup.py).
  - "Phan tich nhan qua (tham do)": cross-sectional T-learner ATE estimate
    across the company panel (src/causal/treatment_effect.py), with
    optional macro covariates and a sample-size-driven confidence label.
  - "Router (S3 - Tu dong)": System S3 - one free-text box, no dropdowns.
    src/router/query_router.py classifies the question and dispatches to
    whichever of the 4 branches above actually answers it, showing the
    classifier's reasoning alongside the answer. This is the thesis's
    actual "Adaptive Retrieval Routing" contribution - the other tabs
    exercise each branch directly for demoing/debugging one at a time.
  - "So sanh ma & Vang": multi-ticker price comparison (rebased to 100)
    against domestic gold price / VN-Index / WTI oil, plus a
    return-correlation table (gold, VN-Index, oil, USD/VND) and an
    event-window comparison around a chosen date - built on
    src/causal/market_factors.py (2026-07-19). Also includes a gold
    what-if for PNJ (COGS shock) and an oil what-if for GAS (revenue
    shock, 2026-07-19 addition - see docs/market_factor_case_study_gold_2025.md
    for why gold/oil use opposite shock mechanisms). See
    docs/market_factor_case_study_gold_2025.md for the sourced 2025
    gold/PNJ case study this tab was originally built to support.
  - "RQ7 - Context Resolver" (2026-07-29): thử cả 5 baseline resolver
    (src/dialogue/context_resolver.py) trên 1 hội thoại - hoặc nhập tay,
    hoặc nạp sẵn 1 hội thoại thật từ
    data/eval/rq7_conversational_benchmark.jsonl - rồi (tuỳ chọn) đẩy câu
    hỏi đã giải quyết ngữ cảnh qua Router (S3) ở trên để xem có định
    tuyến đúng nhánh không. FineTunedContextResolver cần checkpoint thật
    (chưa train trong sandbox) - báo lỗi rõ ràng nếu chưa có, không phải
    demo giả.
  - "Web Fallback (ngoài phạm vi RQ)" (2026-07-29): giao diện cho
    scripts/web_fallback_lookup.py - CHỦ Ý tách biệt khỏi Router/benchmark
    (xem docstring file đó) - luôn hiện banner cảnh báo, không dùng để
    thay thế nhánh RAG đang được đánh giá.
  - "💬 Chat (RQ1-RQ7)" (2026-07-29, rating thêm 2026-08-04): giao diện hội
    thoại thật (st.chat_message/st.chat_input, giống Claude/ChatGPT) - MỖI
    tin nhắn: (1) RQ7 Context Resolver giải quyết ngữ cảnh dựa trên lịch sử
    hội thoại trong phiên này (st.session_state, mất khi tải lại trang -
    không phải backend thật), (2) Router (S3) phân loại + gọi đúng nhánh
    (DAG/RAG/WHATIF/CAUSAL/COMPARE/ABSA). Khác các tab riêng lẻ ở trên (mỗi
    tab test 1 nhánh/1 RQ độc lập) - tab này là nơi DUY NHẤT kết hợp toàn
    bộ hệ thống thành 1 trải nghiệm hội thoại liền mạch.
    HITL (2026-08-04, xem docs/DSA_v6_Bo_sung_2026-08-04.md muc 7 cho
    citation đầy đủ): mỗi câu trả lời ghi 1 dòng `feedback.answer_runs`
    (session_id/question/branch/evidence - PostgreSQL, sống qua reload,
    khác chat_history trong session_state) NGAY khi trả lời xong - không
    chờ rating, vì skip cũng là dữ liệu (denominator của response-rate).
    Widget rating (1-5 sao + reason_code + free text tuỳ chọn, hoặc nút Bỏ
    qua) hiện dưới MỖI câu trả lời CHƯA rate; rating <=2 hoặc DAG->RAG
    fallback tự động vào `feedback.review_queue_lite` (uncertainty
    sampling, Settles 2009) để review sau - KHÔNG có gì tự động retrain.
"""
import json
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.causal.macro_lookup import TICKER_INDUSTRY_MAP, industry_index_for_ticker
from src.causal.market_factors import (
    KNOWN_2025_GOLD_EVENTS,
    compare_tickers,
    correlate_ticker_with_forex,
    correlate_ticker_with_gold,
    correlate_ticker_with_oil,
    correlate_ticker_with_vnindex,
    event_window_comparison,
    forex_series,
    gold_price_series,
    oil_price_series,
    vnindex_series,
)
from src.causal.panel_builder import build_panel
from src.causal.treatment_effect import estimate_ate
from src.core.constants import CORE_MS_CODES
from src.core.db import get_session
from src.structured.calculator import (
    Period, decompose_gross_margin_change, sensitivity_what_if, sensitivity_what_if_gold,
    sensitivity_what_if_macro, sensitivity_what_if_oil,
)
from src.router.query_router import route
from src.core.feedback_store import (
    enqueue_review, extract_evidence, list_pending_review, log_answer_run, log_feedback,
    mark_review_status, maybe_enqueue_low_similarity,
)
from src.structured.dag_pipeline import answer
from src.structured.query_builder import ParsedQuery, execute, parse_question
from src.dialogue.context_resolver import (
    ConversationTurn, FineTunedContextResolver, HumanOracleResolver,
    LLMPromptOnlyResolver, NoHistoryResolver, RawConcatResolver,
)

st.set_page_config(page_title="DAG - Structured Lookup Demo", layout="wide")
st.title("DAG - Structured Lookup Demo (System S2)")
st.caption(
    "Tra cứu trực tiếp từ bảng financial_line_items - không qua LLM, "
    "bounded schema (xem src/structured/schema.py). Không phải NL2SQL tổng quát."
)


@st.cache_data(ttl=30)
def load_companies():
    with get_session() as session:
        rows = session.execute(text(
            "SELECT ticker, company_name, exchange, sector FROM companies ORDER BY ticker"
        )).fetchall()
    return rows


@st.cache_data(ttl=30)
def load_periods(ticker: str):
    with get_session() as session:
        rows = session.execute(text(
            "SELECT DISTINCT fiscal_year, period FROM financial_line_items "
            "WHERE ticker = :ticker ORDER BY fiscal_year, period"
        ), {"ticker": ticker}).fetchall()
    return rows


@st.cache_data(ttl=30)
def load_row_counts():
    with get_session() as session:
        rows = session.execute(text(
            "SELECT c.ticker, c.company_name, c.sector, COUNT(f.id) AS n_rows "
            "FROM companies c LEFT JOIN financial_line_items f ON f.ticker = c.ticker "
            "GROUP BY c.ticker, c.company_name, c.sector ORDER BY n_rows DESC"
        )).fetchall()
    return rows


@st.cache_data(ttl=30)
def load_disclosure_chunk_coverage():
    """Which tickers actually have disclosure_chunks rows RIGHT NOW (Bước
    9-10 output landed in Postgres) - the question 'which tickers are
    done' can't be answered from data/raw/pdf_disclosures or
    data/processed/chunks.jsonl alone (those exist the moment a PDF is
    downloaded/chunked, before vector_store.py's skip_existing-aware
    upsert has actually written anything to the DB) - only this table is
    the real ground truth of what's queryable by RAG/ABSA right now."""
    with get_session() as session:
        rows = session.execute(text(
            "SELECT ticker, COUNT(*) AS n_chunks, COUNT(DISTINCT source_file) AS n_files, "
            "COUNT(DISTINCT doctype) AS n_doctypes, "
            "SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS n_embedded "
            "FROM disclosure_chunks GROUP BY ticker ORDER BY ticker"
        )).fetchall()
    return rows


(tab_nl, tab_form, tab_data, tab_whatif, tab_causal, tab_router, tab_market,
 tab_rq7, tab_webfallback, tab_chat) = st.tabs([
    "Hỏi bằng câu hỏi", "Chọn trực tiếp", "Dữ liệu đã nạp",
    "What-if / Decomposition", "Phân tích nhân quả (thăm dò)", "Router (S3 - Tự động)",
    "So sánh mã & Vàng", "RQ7 - Context Resolver", "Web Fallback (ngoài phạm vi RQ)",
    "💬 Chat (RQ1-RQ7)",
])

with tab_nl:
    st.subheader("Hỏi bằng câu hỏi tự nhiên")
    question = st.text_input(
        "Câu hỏi",
        value="Tổng tài sản của CTD năm 2024 là bao nhiêu?",
    )
    if st.button("Hỏi", key="ask_nl"):
        parsed = parse_question(question)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Đã nhận diện được từ câu hỏi:**")
            st.json({
                "Mã CK": parsed.ticker,
                "Năm": parsed.fiscal_year,
                "Kỳ": parsed.period,
                "Báo cáo": parsed.statement_type,
                "Mã MS": parsed.ms_code,
                "Chỉ tiêu khớp": parsed.matched_alias,
            })
        with col2:
            st.markdown("**Trả lời:**")
            result_text = answer(question)
            if result_text.startswith("Không trả lời được"):
                st.error(result_text)
            else:
                st.success(result_text)

with tab_form:
    st.subheader("Chọn trực tiếp (đảm bảo tra được nếu dữ liệu đã nạp)")
    companies = load_companies()

    if not companies:
        st.warning("Chưa có công ty nào trong bảng companies. Chạy `python scripts/seed_companies.py` trước.")
    else:
        ticker_options = [f"{c.ticker} - {c.company_name}" for c in companies]
        selected = st.selectbox("Công ty", ticker_options)
        ticker = selected.split(" - ")[0]

        periods = load_periods(ticker)
        fiscal_year, period = None, None
        if periods:
            period_labels = [f"{p.fiscal_year} - {p.period}" for p in periods]
            selected_period = st.selectbox("Kỳ có dữ liệu", period_labels)
            year_str, period = selected_period.split(" - ")
            fiscal_year = int(year_str)
        else:
            st.warning(f"Chưa có dữ liệu financial_line_items cho {ticker}.")

        statement_type = st.selectbox("Báo cáo", list(CORE_MS_CODES.keys()))
        indicator_options = CORE_MS_CODES[statement_type]
        ms_code = st.selectbox(
            "Chỉ tiêu",
            list(indicator_options.keys()),
            format_func=lambda c: f"{c} - {indicator_options[c]}",
        )

        if st.button("Tra cứu", key="ask_form"):
            if fiscal_year is None:
                st.error("Chưa chọn được kỳ có dữ liệu cho mã này.")
            else:
                parsed = ParsedQuery(
                    ticker=ticker, fiscal_year=fiscal_year, period=period,
                    statement_type=statement_type, ms_code=ms_code, matched_alias=None,
                )
                result = execute(parsed)
                if result["ok"]:
                    st.success(
                        f"{result['value']:,} {result['unit']} "
                        f"({result['item_name']}, {result['consolidation']})"
                    )
                else:
                    st.error(result["error"])

with tab_data:
    st.subheader("Tổng quan dữ liệu đã nạp")
    row_counts = load_row_counts()
    if row_counts:
        df = pd.DataFrame(
            [(r.ticker, r.company_name, r.sector, r.n_rows) for r in row_counts],
            columns=["Mã CK", "Tên công ty", "Ngành", "Số dòng chỉ tiêu (financial_line_items)"],
        )
        st.metric("Tổng số công ty trong companies", len(df))
        st.metric("Số công ty đã có dữ liệu (>0 dòng)", int((df.iloc[:, 3] > 0).sum()))
        st.dataframe(df, use_container_width=True, height=500)
    else:
        st.info("Chưa có công ty nào. Chạy `python scripts/seed_companies.py` trước.")

    st.markdown("---")
    st.subheader("Độ phủ disclosure_chunks (RAG/ABSA - Bước 9-10)")
    st.caption(
        "Đây là bảng DUY NHẤT trả lời chính xác 'mã nào đã xong' cho RAG/ABSA - có file "
        "trong data/raw/pdf_disclosures/ hay dòng trong chunks.jsonl KHÔNG có nghĩa là đã "
        "vào được Postgres (vector_store.py mới là bước ghi thật, và skip_existing chỉ bỏ "
        "qua embed lại, không phải bỏ qua ghi DB lần đầu)."
    )
    chunk_rows = load_disclosure_chunk_coverage()
    if chunk_rows:
        chunk_df = pd.DataFrame(
            [(r.ticker, r.n_chunks, r.n_files, r.n_doctypes, r.n_embedded) for r in chunk_rows],
            columns=["Mã CK", "Số chunk", "Số file nguồn", "Số loại tài liệu", "Số chunk đã embed"],
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Số mã đã có chunk trong disclosure_chunks", len(chunk_df))
        c2.metric("Tổng số chunk", int(chunk_df["Số chunk"].sum()))
        c3.metric("Tổng số chunk đã embed (dùng được cho RAG)", int(chunk_df["Số chunk đã embed"].sum()))
        st.dataframe(chunk_df, use_container_width=True, height=400)
    else:
        st.warning(
            "disclosure_chunks đang RỖNG - chạy Bước 9 (`pdf_ingest_router.py`) rồi Bước 10 "
            "(`text_splitter.py` + `vector_store.py`) trước, xem README."
        )

with tab_whatif:
    st.subheader("What-if / Decomposition (công thức chính xác, không phải ước lượng ML)")
    st.caption(
        "Trả lời câu hỏi kiểu 'biên lợi nhuận thay đổi bao nhiêu % do doanh thu, "
        "bao nhiêu % do giá vốn' hoặc 'nếu giá vốn tăng X% thì lợi nhuận gộp còn "
        "bao nhiêu' - dùng đẳng thức kế toán đúng 100%, không train mô hình."
    )
    wi_ticker = st.text_input("Mã CK", value="HSG", key="wi_ticker")

    st.markdown("**Bridge decomposition giữa 2 kỳ**")
    c1, c2, c3, c4 = st.columns(4)
    y1 = c1.number_input("Năm kỳ 1", value=2024, step=1, key="wi_y1")
    per1 = c2.text_input("Kỳ 1 (YEAR/Q1..Q4)", value="YEAR", key="wi_per1")
    y2 = c3.number_input("Năm kỳ 2", value=2025, step=1, key="wi_y2")
    per2 = c4.text_input("Kỳ 2 (YEAR/Q1..Q4)", value="YEAR", key="wi_per2")

    if st.button("Decompose", key="wi_decompose"):
        res = decompose_gross_margin_change(wi_ticker, Period(int(y1), per1), Period(int(y2), per2))
        if res["ok"]:
            st.success(
                f"Δ Lợi nhuận gộp = {res['delta_gross_profit']:,.0f} "
                f"(hiệu ứng doanh thu: {res['revenue_growth_effect']:,.0f} "
                f"[{res['revenue_growth_effect_share_pct']}%], "
                f"hiệu ứng biên lợi nhuận/giá vốn: {res['margin_rate_effect']:,.0f} "
                f"[{res['margin_rate_effect_share_pct']}%])"
            )
            st.json(res)
        else:
            st.error(res["error"])

    st.markdown("**Sensitivity what-if (ceteris paribus)**")
    c5, c6, c7 = st.columns(3)
    wi_year = c5.number_input("Năm", value=2025, step=1, key="wi_sy")
    wi_period = c6.text_input("Kỳ", value="YEAR", key="wi_sp")
    cogs_pct = c7.number_input("Giá vốn thay đổi (%)", value=5.0, step=1.0, key="wi_cogs")

    if st.button("Tính what-if", key="wi_run"):
        res = sensitivity_what_if(wi_ticker, Period(int(wi_year), wi_period), cogs_pct_change=cogs_pct)
        if res["ok"]:
            st.success(
                f"Lợi nhuận gộp: {res['gross_profit_before']:,.0f} -> {res['gross_profit_after']:,.0f} "
                f"(biên LN gộp: {res['gross_margin_before_pct']}% -> {res['gross_margin_after_pct']}%)"
            )
            st.caption(f"Giả định: {res['assumption']}")
            st.json(res)
        else:
            st.error(res["error"])

    st.markdown("**Sensitivity what-if grounded bằng dữ liệu vĩ mô thật** (không tự nhập %)")
    st.caption(
        f"Chỉ hoạt động với ticker đã có ánh xạ ngành trong TICKER_INDUSTRY_MAP: "
        f"{', '.join(sorted(TICKER_INDUSTRY_MAP))}."
    )
    if st.button("Tính what-if từ macro thật", key="wi_macro_run"):
        mapped = industry_index_for_ticker(wi_ticker)
        if mapped is None:
            st.error(f"{wi_ticker} chưa có ánh xạ ngành - dùng ô nhập tay ở trên thay thế.")
        else:
            res = sensitivity_what_if_macro(wi_ticker, Period(int(wi_year), wi_period))
            if res["ok"]:
                src = res["macro_source"]
                st.success(
                    f"Lợi nhuận gộp: {res['gross_profit_before']:,.0f} -> {res['gross_profit_after']:,.0f} "
                    f"(dùng % thay đổi {src['indicator_name']} thực tế: "
                    f"{src['pct_change_used_as_cogs_shock']:.2f}%, {src['period_from']} -> {src['period_to']})"
                )
                st.json(res)
            else:
                st.error(res["error"])

with tab_causal:
    st.subheader("Phân tích nhân quả chéo công ty (thăm dò)")
    st.caption(
        "Ước lượng treatment effect (T-learner) trên TOÀN BỘ panel công ty x kỳ - "
        "không phải trên chuỗi thời gian của 1 công ty. Đây là dạng dữ liệu 'nhiều đơn vị "
        "chéo, ít kỳ mỗi đơn vị' đúng với thiết kế của CausalML/EconML. Nhãn độ tin cậy "
        "được tính tự động theo cỡ mẫu thực tế - sẽ tự nâng cấp khi nạp thêm dữ liệu."
    )

    panel_preview = build_panel()
    st.metric("Số dòng panel (ticker x kỳ, period=YEAR)", len(panel_preview))

    c1, c2 = st.columns(2)
    treatment_col = c1.selectbox(
        "Biến 'treatment'", ["leverage", "current_ratio", "asset_turnover", "ocf_to_revenue"], index=0,
    )
    outcome_col = c2.selectbox(
        "Biến kết quả (outcome)", ["net_margin", "gross_margin", "asset_turnover"], index=0,
    )
    extra_covariates = st.multiselect(
        "Biến kiểm soát vĩ mô bổ sung (tùy chọn, ngoài sector + fiscal_year mặc định)",
        ["cpi_annual_avg", "gdp_growth_annual_avg", "gold_price_annual_avg"],
        default=[],
        help="Giá trị thật từ macro_indicators, join theo fiscal_year - cùng 1 công ty trong "
             "cùng 1 năm nhận cùng giá trị macro, đúng bản chất của 1 biến confounder chung. "
             "gold_price_annual_avg (mới, 2026-07-19): dữ liệu giá vàng hiện RẤT thưa (chỉ có "
             "từ ngày DAG thesis_market_snapshot bắt đầu chạy) - hầu hết các năm sẽ ra NaN và bị "
             "loại khỏi ước lượng cho đến khi tích luỹ đủ dữ liệu qua nhiều tháng/năm chạy DAG.",
    )
    covariates = ("sector", "fiscal_year", *extra_covariates)

    if st.button("Ước lượng ATE", key="causal_run"):
        res = estimate_ate(treatment_col=treatment_col, outcome_col=outcome_col, covariates=covariates, panel=panel_preview)
        if res["ok"]:
            st.success(
                f"ATE ({treatment_col} cao vs thấp -> {outcome_col}): "
                f"{res['ate_t_learner']:.4f} (95% CI [{res['ate_95ci'][0]:.4f}, {res['ate_95ci'][1]:.4f}])"
            )
            st.warning(f"Độ tin cậy: {res['confidence']}")
            st.json(res)
        else:
            st.error(res["error"])

with tab_router:
    st.subheader("Router (System S3) - một ô hỏi, tự động chọn nhánh")
    st.caption(
        "Không có dropdown - src/router/query_router.py tự phân loại câu hỏi (DAG / RAG / "
        "What-if / Causal, xem src/router/rules.py) và gọi đúng nhánh. Nếu DAG nhận diện đúng "
        "dạng câu hỏi nhưng không có dữ liệu, router tự động chuyển sang RAG (fallback) thay vì "
        "trả lỗi - đây là phần 'Adaptive' trong tên đề tài."
    )
    router_examples = [
        "Tổng tài sản của VNM quý 1 năm 2026 là bao nhiêu?",
        "Rủi ro pháp lý của công ty là gì?",
        "Nếu COGS của CTD tăng 5% trong 2025, lợi nhuận gộp sẽ ra sao?",
        "So sánh doanh thu của VNM giữa 2024 và 2025",
        "Đòn bẩy có ảnh hưởng đến biên lợi nhuận ròng không?",
        "So sánh PNJ và CTD, có kèm giá vàng",
        "Nếu giá vàng tăng thì PNJ sẽ ra sao?",
        # 2026-09-08 (sửa lại theo yêu cầu người dùng cùng ngày) - bilingual
        # demo tập trung vào 4 "mã chính" (AAV/KTS/AAH/ACV, xoáy sâu
        # 2026-09-07 - nằm trong cả nhóm "đủ dữ liệu" và TiniX-xác nhận), mỗi
        # mã có cặp VI/EN CÙNG nội dung để so sánh thời gian trả lời công
        # bằng (xem docs/PROGRESS.md, data/eval/bilingual_latency_pairs.jsonl).
        "Rủi ro pháp lý của AAV là gì?",
        "What are AAV's legal risks?",
        "Hoạt động kinh doanh chính của KTS là gì?",
        "What is KTS's main business activity?",
        "Rủi ro pháp lý của AAH là gì?",
        "What are AAH's legal risks?",
        "Hoạt động kinh doanh chính của ACV là gì?",
        "What is ACV's main business activity?",
    ]
    router_q = st.selectbox("Câu hỏi mẫu (hoặc tự nhập bên dưới) / Sample question (VI/EN)", ["(tự nhập)"] + router_examples)
    router_question = st.text_input(
        "Câu hỏi",
        value="" if router_q == "(tự nhập)" else router_q,
        key="router_q_input",
    )
    if st.button("Hỏi (Router)", key="router_run") and router_question:
        # 2026-09-08 - bộ đếm thời gian hiển thị TRỰC TIẾP trên UI (trước đó
        # Router tab hoàn toàn không đo thời gian; Chat tab có đo latency_ms
        # nhưng chỉ ghi vào log_answer_run, không hiện lên màn hình - cả 2
        # đều là gap thật khi demo trực diện VI/EN cần thấy ngay số giây).
        _t0 = time.perf_counter()
        with st.spinner("Đang xử lý (Router + nhánh tương ứng)... / Processing..."):
            result = route(router_question)
        latency_s = time.perf_counter() - _t0
        from src.core.lang_detect import detect_language
        q_lang = detect_language(router_question)
        st.metric(
            "⏱ Thời gian trả lời / Response time",
            f"{latency_s:.2f}s",
            help=f"Ngôn ngữ câu hỏi phát hiện được / detected question language: {q_lang.upper()}",
        )
        badge = {
            "dag": "🔵 DAG", "rag": "🟢 RAG", "whatif": "🟡 What-if", "causal": "🟣 Causal",
            "compare": "🟠 Compare", "absa": "🟤 ABSA",
            "whatif_unsupported": "⚪ Whatif (chưa đủ dữ liệu để tự động)",
            "causal_unsupported": "⚪ Causal (chưa đủ dữ liệu để tự động)",
            "compare_unsupported": "⚪ Compare (chưa đủ dữ liệu để tự động)",
            "absa_unsupported": "⚪ ABSA (chưa nhận diện được mã)",
        }.get(result["branch"], result["branch"])
        st.markdown(f"**Nhánh được chọn:** {badge}" + (
            f"  _(fallback từ {result['fallback_from']})_" if result["fallback_from"] else ""
        ))
        st.caption(f"Lý do: {result['intent_reason']}")
        st.success(result["answer"])
        is_web_fallback = result["branch"] == "rag" and result["raw"] and result["raw"].get("source_tier") == "web_fallback"
        if is_web_fallback:
            st.warning("Trả lời từ báo chí/nguồn web - KHÔNG phải tài liệu công bố chính thức (BCTC/báo cáo thường niên).")
        if result["branch"] == "rag" and result["raw"] and result["raw"].get("sources"):
            st.markdown("**Nguồn:**")
            for i, s in enumerate(result["raw"]["sources"], start=1):
                if is_web_fallback:
                    date_part = f" - ngày {s['published_date']}" if s.get("published_date") else ""
                    st.caption(f"[{i}] {s['domain']}{date_part} - {s['source_url']} (similarity={s['similarity']})")
                    continue
                st.caption(
                    f"[{i}] {s['ticker']} - {s['doctype']} - {s['source_file']} "
                    f"trang {s['page']} (similarity={s['similarity']})"
                )
        with st.expander("Kết quả thô (raw)"):
            st.json(result["raw"] if result["raw"] is not None else {})

with tab_market:
    st.subheader("So sánh nhiều mã & giá vàng (Step 11 + Step 14)")
    st.caption(
        "Dựa trên dữ liệu daily ingest qua Airflow (scripts/market_snapshot.py) - VN-Index, "
        "giá vàng SJC, giá đóng cửa từng mã (company_price_snapshot). Xem "
        "src/causal/market_factors.py và docs/market_factor_case_study_gold_2025.md."
    )

    gold_df = gold_price_series()
    vni_df = vnindex_series()
    forex_df = forex_series()
    oil_df = oil_price_series()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số ngày có dữ liệu giá vàng", len(gold_df))
    c2.metric("Số ngày có dữ liệu VN-Index", len(vni_df))
    c3.metric("Số ngày có dữ liệu USD/VND", len(forex_df))
    c4.metric("Số ngày có dữ liệu dầu WTI", len(oil_df))
    if len(gold_df) < 30 or len(forex_df) < 30:
        st.info(
            "Dữ liệu giá vàng và USD/VND còn thưa (fetch_gold()/fetch_forex() chỉ ghi được giá "
            "NGÀY HÔM ĐÓ mỗi lần chạy - xem docstring trong scripts/market_snapshot.py) - biểu đồ/"
            "tương quan bên dưới sẽ đầy đủ hơn khi DAG thesis_market_snapshot chạy tích luỹ qua "
            "nhiều ngày/tháng."
        )
    if oil_df.empty:
        st.info(
            "Chưa có dữ liệu giá dầu WTI - fetch_oil() cần FRED_API_KEY trong .env để backfill "
            "(xem scripts/market_snapshot.py)."
        )

    st.markdown("**So sánh hiệu suất giá nhiều mã (rebase = 100 tại ngày đầu)**")
    default_tickers = "PNJ,CTD,VNM"
    tickers_input = st.text_input("Danh sách mã, phân cách bởi dấu phẩy", value=default_tickers, key="mkt_tickers")
    c_ov1, c_ov2 = st.columns(2)
    show_gold_overlay = c_ov1.checkbox("Chồng thêm đường giá vàng (rebase)", value=True, key="mkt_show_gold")
    show_oil_overlay = c_ov2.checkbox("Chồng thêm đường giá dầu WTI (rebase)", value=False, key="mkt_show_oil")

    if st.button("Vẽ biểu đồ so sánh", key="mkt_compare_run"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        cmp_result = compare_tickers(tickers)
        if not cmp_result["ok"]:
            st.error(cmp_result["error"])
        else:
            chart_df = cmp_result["wide"].set_index("trade_date")
            if show_gold_overlay and not gold_df.empty:
                gold_rebased = gold_df.set_index("trade_date")["gold_price"]
                gold_rebased = gold_rebased / gold_rebased.iloc[0] * 100
                chart_df = chart_df.join(gold_rebased.rename("Vàng SJC (rebase)"), how="left")
            if show_oil_overlay and not oil_df.empty:
                oil_rebased = oil_df.set_index("trade_date")["oil_price"]
                oil_rebased = oil_rebased / oil_rebased.iloc[0] * 100
                chart_df = chart_df.join(oil_rebased.rename("Dầu WTI (rebase)"), how="left")
            st.line_chart(chart_df)
            if cmp_result["missing"]:
                st.warning(f"Không có dữ liệu cho: {cmp_result['missing']}")

            st.markdown("**Tương quan lợi suất ngày với vàng / VN-Index / dầu WTI / USD-VND**")
            rows = []
            for t in tickers:
                if t in cmp_result["missing"]:
                    continue
                rg = correlate_ticker_with_gold(t)
                rv = correlate_ticker_with_vnindex(t)
                ro = correlate_ticker_with_oil(t)
                rf = correlate_ticker_with_forex(t)
                notes = [r.get("error") for r in (rg, rv, ro, rf) if not r["ok"] and r.get("error")]
                rows.append({
                    "Mã": t,
                    "Tương quan với vàng": round(rg["correlation"], 3) if rg["ok"] else None,
                    "Tương quan với VN-Index": round(rv["correlation"], 3) if rv["ok"] else None,
                    "Tương quan với dầu WTI": round(ro["correlation"], 3) if ro["ok"] else None,
                    "Tương quan với USD/VND": round(rf["correlation"], 3) if rf["ok"] else None,
                    "Ghi chú": "OK" if not notes else notes[0],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("**Kịch bản: so sánh quanh một sự kiện giá vàng đã biết (2025)**")
    st.caption(
        "Không phải ước lượng nhân quả (không có nhóm đối chứng/kiểm soát biến gây nhiễu - "
        "dùng src/causal/treatment_effect.py cho việc đó) - đây chỉ là mức thay đổi giá quanh "
        "một mốc thời gian cụ thể, để minh hoạ."
    )
    event_key = st.selectbox(
        "Sự kiện", list(KNOWN_2025_GOLD_EVENTS.keys()),
        format_func=lambda k: f"{KNOWN_2025_GOLD_EVENTS[k]['date']} - {k}",
        key="mkt_event",
    )
    event = KNOWN_2025_GOLD_EVENTS[event_key]
    st.caption(event["description"])
    event_tickers = st.text_input(
        "Mã muốn xem quanh sự kiện này", value=",".join(event["affected_tickers"] + ["VNM"]), key="mkt_event_tickers",
    )
    if st.button("Xem thay đổi quanh sự kiện", key="mkt_event_run"):
        ts = [t.strip().upper() for t in event_tickers.split(",") if t.strip()]
        ev_result = event_window_comparison(ts, event["date"], days_before=10, days_after=10)
        st.json(ev_result)

    st.markdown("**Kịch bản what-if: giá vàng thực tế -> lợi nhuận gộp PNJ**")
    st.caption(
        "sensitivity_what_if_gold() - dùng biến động giá vàng SJC THỰC TẾ gần nhất (không phải % "
        "tự nhập) làm cú sốc COGS, chỉ áp dụng cho GOLD_SENSITIVE_TICKERS (hiện là PNJ)."
    )
    gold_wi_year = st.number_input("Năm", value=2025, step=1, key="mkt_gold_wi_year")
    if st.button("Tính kịch bản giá vàng cho PNJ", key="mkt_gold_wi_run"):
        res = sensitivity_what_if_gold("PNJ", Period(int(gold_wi_year), "YEAR"))
        if res["ok"]:
            st.success(
                f"Lợi nhuận gộp: {res['gross_profit_before']:,.0f} -> {res['gross_profit_after']:,.0f} "
                f"(biên LN gộp: {res['gross_margin_before_pct']}% -> {res['gross_margin_after_pct']}%)"
            )
            st.caption(f"Nguồn giá vàng: {res['gold_source']}")
            st.json(res)
        else:
            st.error(res["error"])

    st.markdown("**Kịch bản what-if: giá dầu WTI thực tế -> doanh thu GAS**")
    st.caption(
        "sensitivity_what_if_oil() - dùng biến động giá dầu WTI THỰC TẾ gần nhất (không phải % "
        "tự nhập) làm cú sốc DOANH THU (không phải COGS - GAS/PV Gas bán khí theo công thức giá "
        "gắn với dầu, khác cơ chế với PNJ+vàng), chỉ áp dụng cho OIL_SENSITIVE_TICKERS (hiện là "
        "GAS). Cần FRED_API_KEY để có dữ liệu giá dầu - xem cảnh báo phía trên nếu chưa có."
    )
    oil_wi_year = st.number_input("Năm", value=2025, step=1, key="mkt_oil_wi_year")
    if st.button("Tính kịch bản giá dầu cho GAS", key="mkt_oil_wi_run"):
        res = sensitivity_what_if_oil("GAS", Period(int(oil_wi_year), "YEAR"))
        if res["ok"]:
            st.success(
                f"Lợi nhuận gộp: {res['gross_profit_before']:,.0f} -> {res['gross_profit_after']:,.0f} "
                f"(biên LN gộp: {res['gross_margin_before_pct']}% -> {res['gross_margin_after_pct']}%)"
            )
            st.caption(f"Nguồn giá dầu: {res['oil_source']}")
            st.json(res)
        else:
            st.error(res["error"])

RQ7_DATASET_PATH = Path("data/eval/rq7_conversational_benchmark.jsonl")
RQ7_BASELINES = {
    "no_history": "1. Không dùng lịch sử (baseline sàn)",
    "raw_concat": "2. Nối thô k lượt gần nhất",
    "llm_prompt_only": "3. LLM viết lại (Qwen3/Ollama, không fine-tune)",
    "finetuned_mt5_lora": "4. mT5+LoRA đã fine-tune (cần checkpoint thật)",
    "human_oracle": "5. Human oracle (chỉ dùng được với hội thoại mẫu có gold)",
}


@st.cache_data(ttl=60)
def load_rq7_sample_conversations():
    if not RQ7_DATASET_PATH.exists():
        return []
    convs = []
    with RQ7_DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(json.loads(line))
    return convs


with tab_rq7:
    st.subheader("RQ7 - Context Resolver (thử 5 baseline)")
    st.caption(
        "Đặt TRƯỚC Router (src/dialogue/context_resolver.py) - giải quyết câu hỏi phụ "
        "thuộc ngữ cảnh ('nó', 'còn... thì sao', tự sửa lời...) thành câu hỏi độc lập, "
        "rồi (tuỳ chọn) đẩy tiếp qua Router (S3) ở tab trên để xem có định tuyến đúng "
        "nhánh không. Xem docs/Ghi_chu_thao_luan_phan_bien_2026-07-26.md mục 8.10."
    )

    sample_convs = load_rq7_sample_conversations()
    mode = st.radio("Nguồn hội thoại", ["Tự nhập tay", "Nạp hội thoại mẫu (có gold)"],
                     horizontal=True, key="rq7_mode")

    history_text, current_utterance, selected_item = "", "", None
    if mode == "Nạp hội thoại mẫu (có gold)":
        if not sample_convs:
            st.warning(
                f"Chưa thấy {RQ7_DATASET_PATH} - chạy `python3 scripts/build_rq7_dataset.py` trước."
            )
        else:
            labels = [f"{c['conv_id']} ({c['error_category']})" for c in sample_convs]
            idx = st.selectbox("Hội thoại mẫu", range(len(sample_convs)),
                                format_func=lambda i: labels[i], key="rq7_sample_idx")
            selected_item = sample_convs[idx]
            turns = selected_item["turns"]
            history_text = "\n".join(t["utterance"] for t in turns[:-1])
            current_utterance = turns[-1]["utterance"]
            st.json({"turns": turns, "gold_standalone_question": selected_item["gold_standalone_question"],
                     "gold_route": selected_item["gold_route"], "gold_ticker": selected_item["gold_ticker"]})

    st.markdown("**Lịch sử hội thoại** (mỗi dòng 1 lượt trước đó, có thể để trống)")
    history_input = st.text_area("Lịch sử", value=history_text, key="rq7_history", height=100)
    current_input = st.text_input("Câu hỏi cuối cùng cần giải quyết", value=current_utterance, key="rq7_current")

    baseline_key = st.selectbox(
        "Baseline", list(RQ7_BASELINES.keys()), format_func=lambda k: RQ7_BASELINES[k], key="rq7_baseline",
    )
    concat_k = 3
    if baseline_key == "raw_concat":
        concat_k = st.slider("k (số lượt gần nhất nối vào)", 1, 5, 3, key="rq7_concat_k")
    then_route = st.checkbox("Sau đó đẩy câu hỏi đã giải quyết qua Router (S3)", value=True, key="rq7_then_route")

    if st.button("Giải quyết ngữ cảnh", key="rq7_run"):
        history_turns = [ConversationTurn(i, "user", line) for i, line in
                          enumerate(l for l in history_input.split("\n") if l.strip())]
        try:
            if baseline_key == "no_history":
                resolved = NoHistoryResolver().resolve(history_turns, current_input)
            elif baseline_key == "raw_concat":
                resolved = RawConcatResolver(k=concat_k).resolve(history_turns, current_input)
            elif baseline_key == "llm_prompt_only":
                resolved = LLMPromptOnlyResolver().resolve(history_turns, current_input)
            elif baseline_key == "finetuned_mt5_lora":
                resolved = FineTunedContextResolver().resolve(history_turns, current_input)
            elif baseline_key == "human_oracle":
                if selected_item is None:
                    st.error("Human oracle chỉ dùng được khi nạp hội thoại mẫu (có gold) ở trên.")
                    resolved = None
                else:
                    resolved = HumanOracleResolver().resolve_from_item(selected_item)
            else:
                resolved = None
        except FileNotFoundError as e:
            st.error(f"Chưa có checkpoint fine-tune thật: {e}")
            resolved = None
        except Exception as e:
            st.error(f"Lỗi khi gọi baseline '{baseline_key}': {e}")
            resolved = None

        if resolved is not None:
            st.success(f"Câu hỏi độc lập: {resolved.standalone_question}")
            st.json({
                "resolver": resolved.resolver_name, "ticker": resolved.ticker,
                "fiscal_year": resolved.fiscal_year, "confidence": resolved.confidence,
            })
            if selected_item is not None:
                st.caption(
                    f"So với gold: '{selected_item['gold_standalone_question']}' "
                    f"(mã gold: {selected_item['gold_ticker']}, nhánh gold: {selected_item['gold_route']})"
                )
            if then_route:
                st.markdown("**Kết quả sau khi đẩy qua Router (S3):**")
                router_result = route(resolved.standalone_question)
                st.markdown(f"Nhánh: `{router_result['branch']}`")
                st.caption(f"Lý do: {router_result['intent_reason']}")
                st.info(router_result["answer"])

with tab_webfallback:
    st.subheader("Web Fallback - tiện ích tra cứu web (NGOÀI PHẠM VI RQ1-RQ7)")
    st.warning(
        "Kết quả ở tab này đến từ tìm kiếm web trực tiếp, KHÔNG PHẢI RAG đã verify - "
        "không tái lập được, không nằm trong benchmark.jsonl, không được chấm điểm bởi "
        "bất kỳ RQ nào. Chỉ dùng để tra cứu tiện ích khi dữ liệu chưa kịp nạp vào "
        "disclosure_chunks. Xem docstring scripts/web_fallback_lookup.py."
    )
    wf_question = st.text_input("Câu hỏi", value="", key="wf_question")
    wf_ticker_in = st.text_input("Mã CK (tuỳ chọn)", value="", key="wf_ticker")
    wf_sites = st.text_input(
        "Danh sách domain giới hạn tìm kiếm (phân cách bởi dấu phẩy, để trống dùng mặc định)",
        value="", key="wf_sites",
    )
    wf_no_fetch = st.checkbox("Chỉ dùng snippet, không tải nội dung trang top-1", value=False, key="wf_no_fetch")

    if st.button("Tìm trên web", key="wf_run") and wf_question:
        try:
            from scripts.web_fallback_lookup import DEFAULT_SITES, answer_with_web_fallback
        except ImportError as e:
            st.error(f"Thiếu thư viện tuỳ chọn: {e}. Chạy `pip install ddgs` trước.")
        else:
            sites = [s.strip() for s in wf_sites.split(",") if s.strip()] or None
            result = answer_with_web_fallback(
                wf_question, ticker=wf_ticker_in or None, sites=sites,
                fetch_top_page=not wf_no_fetch,
            )
            if result["note"]:
                st.error(result["note"])
            else:
                st.info(result["answer"])
                st.markdown("**Các nguồn đã tìm được:**")
                for url in result["sources"]:
                    st.caption(url)
            st.caption(f"Domain đã giới hạn: {sites or DEFAULT_SITES}")

CHAT_RESOLVER_OPTIONS = {
    "no_history": "Không dùng lịch sử (nhanh nhất, không gọi LLM)",
    "raw_concat": "Nối thô 3 lượt gần nhất (nhanh, không gọi LLM)",
    "llm_prompt_only": "LLM viết lại (Qwen3/Ollama - chậm hơn, hiểu ngữ cảnh tốt hơn)",
}


def _resolve_chat_context(history_turns, current_utterance, resolver_choice):
    """Wraps the 3 RQ7 baselines that don't need a trained checkpoint (the
    chat tab is a live demo, not a benchmark run - FineTunedContextResolver
    is deliberately NOT offered here since it would just raise
    FileNotFoundError every time until a real checkpoint exists, see the
    RQ7 tab above for testing that one specifically). Falls back to the
    raw utterance, with a visible warning, on ANY resolver error rather than
    breaking the whole chat turn."""
    try:
        if resolver_choice == "no_history":
            return NoHistoryResolver().resolve(history_turns, current_utterance).standalone_question
        if resolver_choice == "raw_concat":
            return RawConcatResolver(k=3).resolve(history_turns, current_utterance).standalone_question
        return LLMPromptOnlyResolver().resolve(history_turns, current_utterance).standalone_question
    except Exception as e:
        st.warning(f"RQ7 resolver lỗi ({e}) - dùng nguyên câu hỏi gốc, không giải quyết ngữ cảnh.")
        return current_utterance


FEEDBACK_REASON_LABELS = {
    "good": "Đúng, hài lòng", "wrong_number": "Sai số liệu", "wrong_branch": "Sai nhánh routing",
    "missing_evidence": "Thiếu nguồn/bằng chứng", "hallucination": "Bịa đặt (hallucination)",
    "outdated": "Dữ liệu cũ", "unclear": "Trả lời không rõ", "too_slow": "Quá chậm", "other": "Khác",
}


def _render_feedback_widget(msg: dict, msg_idx: int) -> None:
    """1-5 sao + reason_code + free text tuỳ chọn, hoặc nút Bỏ qua - ghi vào
    feedback.answer_feedback. Chỉ hiện nếu answer_run_id có thật (log_answer_run
    không lỗi) và message này CHƯA được rate trong phiên này. Skip cũng ghi
    log (is_skip=True) - xem docstring module này + src/core/feedback_store.py
    cho lý do (denominator của response-rate, Wu et al. 2022)."""
    if not msg.get("answer_run_id") or msg.get("feedback_done"):
        return
    with st.form(key=f"feedback_form_{msg_idx}", border=False):
        cols = st.columns([2, 3, 1])
        rating = cols[0].feedback("stars", key=f"rating_{msg_idx}")
        reason = cols[1].selectbox(
            "Lý do (tuỳ chọn)", [None] + list(FEEDBACK_REASON_LABELS.keys()),
            format_func=lambda k: "-" if k is None else FEEDBACK_REASON_LABELS[k],
            key=f"reason_{msg_idx}", label_visibility="collapsed",
        )
        # form_submit_button() (streamlit==1.39.0 pinned in requirements.txt)
        # has NO `key` param - uniqueness across the 2 buttons here comes
        # from distinct labels within this st.form's own key
        # (feedback_form_{msg_idx}, already unique per message).
        submitted = cols[2].form_submit_button("Gửi")
        skip = st.form_submit_button("Bỏ qua đánh giá câu này")
        free_text = st.text_input("Ghi chú thêm (tuỳ chọn)", key=f"free_text_{msg_idx}")

        if submitted or skip:
            try:
                if skip or rating is None:
                    log_feedback(answer_run_id=msg["answer_run_id"], is_skip=True)
                else:
                    log_feedback(
                        answer_run_id=msg["answer_run_id"], rating=int(rating) + 1,  # st.feedback trả 0-4
                        reason_code=reason, free_text=free_text or None,
                    )
                msg["feedback_done"] = True
                st.rerun()
            except Exception as e:
                st.warning(f"Không ghi được feedback ({e}) - DB có thể chưa chạy migration 013.")


with tab_chat:
    st.subheader("💬 Chat tổng hợp - kết hợp RQ1-RQ7 trong 1 hội thoại")
    st.caption(
        "Mỗi tin nhắn: (1) RQ7 Context Resolver giải quyết câu hỏi phụ thuộc ngữ cảnh dựa "
        "trên lịch sử hội thoại TRONG PHIÊN NÀY (session_state - mất khi tải lại trang, "
        "không phải bộ nhớ lâu dài/backend thật), rồi (2) Router (S3, RQ1) phân loại + gọi "
        "đúng nhánh DAG/RAG/WHATIF/CAUSAL/COMPARE/ABSA (RQ2/RQ3/RQ4/RQ5/RQ6). Đây là bản "
        "demo trước khi có backend thật (src/api/ vẫn NOT STARTED) - không phải sản phẩm "
        "cuối cùng. Mỗi câu trả lời được LOG vào PostgreSQL (feedback.answer_runs) ngay "
        "khi trả lời xong - hãy rate hoặc bấm 'Bỏ qua' bên dưới mỗi câu, đây là dữ liệu "
        "human-evaluation thật cho thesis (docs/DSA_v6_Bo_sung_2026-08-04.md mục 7)."
    )

    chat_resolver_choice = st.selectbox(
        "Bộ giải quyết ngữ cảnh (RQ7)", list(CHAT_RESOLVER_OPTIONS.keys()),
        index=2, format_func=lambda k: CHAT_RESOLVER_OPTIONS[k], key="chat_resolver_choice",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_session_id" not in st.session_state:
        # Groups every answer_run in this browser session together
        # (feedback.answer_runs.session_id) - regenerated on page reload,
        # same lifecycle as chat_history itself (thesis-now has no login).
        st.session_state.chat_session_id = uuid.uuid4().hex

    if st.button("🗑️ Xoá hội thoại", key="chat_clear"):
        st.session_state.chat_history = []
        st.rerun()

    with st.expander("🗂️ Hàng đợi review (HITL) — việc bạn cần xem lại"):
        st.caption(
            "Tự động vào đây khi: rating ≤2, DAG→RAG fallback, hoặc RAG similarity <0,45 "
            "(uncertainty sampling, Settles 2009). Đây LÀ giao diện HITL của chính bạn — "
            "không có gì tự động retrain, chỉ có mục này để bạn quyết định."
        )
        try:
            pending = list_pending_review(limit=20)
        except Exception as e:
            pending = []
            st.caption(f"⚠️ Không đọc được queue ({e}).")
        if not pending:
            st.caption("Trống — không có mục nào đang chờ review.")
        for item in pending:
            with st.container(border=True):
                st.caption(
                    f"#{item['review_id']} · nguồn: `{item['source']}` · ưu tiên {item['priority']} "
                    f"· nhánh: `{item['branch']}`" + (f" (fallback từ {item['fallback_from']})" if item['fallback_from'] else "")
                )
                st.markdown(f"**Q:** {item['question_text']}")
                st.markdown(f"**A:** {item['answer_text']}")
                note = st.text_input("Ghi chú quyết định", key=f"queue_note_{item['review_id']}")
                qc1, qc2, qc3 = st.columns(3)
                if qc1.button("Đã xử lý", key=f"queue_resolved_{item['review_id']}"):
                    mark_review_status(review_id=item["review_id"], status="resolved", resolution_note=note or None)
                    st.rerun()
                if qc2.button("Đã xem (chưa xong)", key=f"queue_reviewed_{item['review_id']}"):
                    mark_review_status(review_id=item["review_id"], status="reviewed", resolution_note=note or None)
                    st.rerun()
                if qc3.button("Bỏ qua mục này", key=f"queue_dismissed_{item['review_id']}"):
                    mark_review_status(review_id=item["review_id"], status="dismissed", resolution_note=note or None)
                    st.rerun()

    def _md_newlines(text: str) -> str:
        # Markdown nuốt xuống dòng đơn -> câu trả lời nhiều dòng (trend theo
        # quý, scope hint...) bị dồn thành 1 dòng dài rất khó đọc (feedback
        # 2026-08-06). Hai dấu cách cuối dòng = hard line break trong
        # markdown, giữ nguyên các đoạn đã cách nhau bằng dòng trống.
        return text.replace("\n", "  \n")

    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(_md_newlines(msg["content"]))
            if msg.get("branch"):
                st.caption(f"Nhánh: `{msg['branch']}`" + (
                    f" | Câu hỏi đã giải quyết ngữ cảnh: _{msg['resolved_question']}_"
                    if msg.get("resolved_question") and msg["resolved_question"] != msg.get("original_question")
                    else ""
                ))
                st.caption(f"Lý do: {msg.get('reason', '')}")
            if msg["role"] == "assistant":
                if msg.get("feedback_done"):
                    st.caption("✅ Đã ghi nhận đánh giá.")
                else:
                    _render_feedback_widget(msg, i)

    _last_assistant = next(
        (m for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"), None
    )
    _awaiting_feedback = bool(
        _last_assistant and _last_assistant.get("answer_run_id") and not _last_assistant.get("feedback_done")
    )
    if _awaiting_feedback:
        st.info("⬆️ Vui lòng rate hoặc bấm 'Bỏ qua' cho câu trả lời trên trước khi hỏi tiếp.")
    user_msg = st.chat_input(
        "Nhập câu hỏi (có thể hỏi tiếp theo ngữ cảnh câu trước)...",
        disabled=_awaiting_feedback,
    )
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        history_turns = [
            ConversationTurn(i, "user", m["content"])
            for i, m in enumerate(st.session_state.chat_history[:-1])
            if m["role"] == "user"
        ]
        # 2026-08-07: khai triển viết tắt tài chính (TTS -> "tổng tài sản"...)
        # TRƯỚC khi câu đi qua LLM context-resolver - phát hiện qua Chat rằng
        # resolver có thể đoán bừa viết tắt nó không biết ("TTS" -> "Tỷ giá
        # trung tâm"). Từ điển deterministic đi trước, LLM chỉ xử lý phần
        # ngữ cảnh hội thoại. Xem src/structured/abbreviations.py.
        from src.structured.abbreviations import expand_display
        expanded_msg = expand_display(user_msg)
        resolved_question = _resolve_chat_context(history_turns, expanded_msg, chat_resolver_choice)

        with st.chat_message("assistant"):
            with st.spinner("Đang xử lý (Router + nhánh tương ứng)..."):
                _t0 = time.perf_counter()
                result = route(resolved_question)
                latency_ms = int((time.perf_counter() - _t0) * 1000)
            st.markdown(_md_newlines(result["answer"]))
            branch_label = result["branch"] + (
                f" (fallback từ {result['fallback_from']})" if result["fallback_from"] else ""
            )
            if resolved_question != user_msg:
                st.caption(f"Nhánh: `{branch_label}` | Câu hỏi đã giải quyết ngữ cảnh: _{resolved_question}_")
            else:
                st.caption(f"Nhánh: `{branch_label}`")
            st.caption(f"Lý do: {result['intent_reason']}")
            # 2026-09-08 - latency_ms was already measured above (for
            # log_answer_run below) but never shown on screen - added so a
            # live demo (VI or EN question) visibly shows response time,
            # not just the DB log.
            from src.core.lang_detect import detect_language as _detect_lang_chat
            _q_lang_chat = _detect_lang_chat(user_msg)
            st.caption(f"⏱ {latency_ms} ms ({latency_ms / 1000:.2f}s) - ngôn ngữ câu hỏi: {_q_lang_chat.upper()}")
            is_web_fallback = result["branch"] == "rag" and result["raw"] and result["raw"].get("source_tier") == "web_fallback"
            if is_web_fallback:
                st.warning("Trả lời từ báo chí/nguồn web - KHÔNG phải tài liệu công bố chính thức (BCTC/báo cáo thường niên).")
            if result["branch"] in ("rag", "absa") and result["raw"] and result["raw"].get("sources"):
                with st.expander("Nguồn"):
                    for i, s in enumerate(result["raw"]["sources"], start=1):
                        if is_web_fallback:
                            date_part = f" - ngày {s['published_date']}" if s.get("published_date") else ""
                            st.caption(f"[{i}] {s['domain']}{date_part} - {s['source_url']}")
                            continue
                        st.caption(f"[{i}] {s['ticker']} - {s['doctype']} - {s['source_file']} trang {s['page']}")

            # Log the answer_run right away - independent of whether the user
            # ever rates it (skip is data too, see FEEDBACK_REASON_LABELS
            # widget below). Never let a logging failure break the chat turn.
            answer_run_id = None
            try:
                evidence = extract_evidence(result["raw"], result["branch"])
                answer_run_id = log_answer_run(
                    session_id=st.session_state.chat_session_id,
                    question_text=user_msg, resolved_question=resolved_question,
                    branch=result["branch"], fallback_from=result["fallback_from"],
                    answer_text=result["answer"], evidence=evidence, latency_ms=latency_ms,
                )
                if result["fallback_from"]:
                    # DAG->RAG fallback is an uncertainty proxy known at
                    # answer time (Settles 2009) - enqueue immediately,
                    # don't wait for a low rating that may never come.
                    enqueue_review(answer_run_id=answer_run_id, source="fallback", priority=2)
                maybe_enqueue_low_similarity(answer_run_id=answer_run_id, evidence=evidence)
            except Exception as e:
                st.caption(f"⚠️ Không ghi được log vào DB ({e}) - chat vẫn hoạt động bình thường.")

        st.session_state.chat_history.append({
            "role": "assistant", "content": result["answer"],
            "branch": branch_label, "reason": result["intent_reason"],
            "original_question": user_msg, "resolved_question": resolved_question,
            "answer_run_id": answer_run_id, "feedback_done": False,
        })
        st.rerun()  # re-render from chat_history so the new message gets its feedback widget
