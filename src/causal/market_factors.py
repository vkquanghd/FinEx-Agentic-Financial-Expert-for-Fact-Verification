"""Query/analysis helpers over the "real-time market factor" data ingested
by scripts/market_snapshot.py (Step 11) - domestic gold price, VN-Index,
USD/VND exchange rate, WTI oil price, and per-company daily price
(`company_price_snapshot`). Built to answer 3 concrete questions raised
about extending the thesis system with this data (2026-07-19), then
extended again the same day after noticing forex/oil were being INGESTED
by the Airflow DAG but never queried anywhere - see git history / the
"orphaned indicators" note below:

1. How did gold price moves in 2025 relate to the market and to specific
   gold-linked tickers, with real evidence? -> gold_price_series(),
   ticker_price_series(), correlate_ticker_with_gold(), and
   KNOWN_2025_GOLD_EVENTS (dated, sourced facts - see that constant's
   docstring for citations) for event_window_comparison().
2. Compare multiple tickers against each other, with a few scenarios?
   -> compare_tickers() (normalized price series for charting) and
   event_window_comparison() (return around a specific dated event).
3. Should there be a comparison chart? -> yes, see app_streamlit.py's new
   "So sánh mã & Vàng" tab and scripts/market_factor_chart.py for a static
   PNG export.

Orphaned-indicators follow-up: `scripts/market_snapshot.py` also ingests
USD/VND exchange rate (MKT_FOREX_VCB) and WTI oil price (MKT_OIL_WTI,
skipped if no FRED_API_KEY) into `macro_indicators`, alongside gold/
VN-Index - but until this pass, nothing in the codebase ever READ those
two categories back out; they were being stored and never used. Added:
`forex_series()`/`oil_price_series()` (same query pattern as gold/
VN-Index), `correlate_ticker_with_forex()`/`correlate_ticker_with_oil()`,
`forex_annual_aggregate()`/`oil_annual_aggregate()` (feeding new optional
covariates in src/causal/panel_builder.py), and
`calculator.sensitivity_what_if_oil()` for GAS (Tổng Công ty Khí Việt Nam -
PV Gas). Deliberately did NOT add a forex-sensitivity ticker mapping the
way GOLD_SENSITIVE_TICKERS/OIL_SENSITIVE_TICKERS work - real USD exposure
varies company to company and isn't something to guess at without
checking actual disclosures first (same conservative philosophy as
TICKER_INDUSTRY_MAP in macro_lookup.py leaving real-estate developers
unmapped) - forex is available as a correlation/causal-covariate tool,
just not pre-bound to specific "forex-sensitive" tickers.

Design notes:
- Gold data has NO single canonical `indicator_name` - fetch_gold() in
  scripts/market_snapshot.py writes one row per (branch/type, buy/sell)
  combination returned by vnstock's sjc_gold_price() (e.g. "SJC 1L, 10L,
  1KG - sell_price"), and those exact strings were never confirmed against
  a real query result (the environment that built this can't reach the
  project owner's live DB). Rather than hardcode a guessed exact name,
  gold_price_series() averages across ALL rows matching `%{price_type}`
  for each day - the same "blend sub-readings into one representative
  daily value" pattern already used by macro_lookup.annual_aggregate().
- Correlations are computed on DAILY RETURNS (pct_change), never on raw
  price levels - correlating two trending level series is a classic
  spurious-correlation trap (two things that both went up all year will
  show near-1.0 correlation regardless of any real relationship). Returns
  correlation is the standard finance convention and the actually
  meaningful comparison.
- Real gold data coverage is currently SPARSE: fetch_gold() only ever
  writes TODAY's snapshot (SJC's API has no historical range endpoint),
  so the row count only grows by ~1 day's worth of entries per Airflow DAG
  run (confirmed: 24 rows after 24 days of runs per docs/PROGRESS.md) -
  NOT a proper full-2025 historical gold series yet. Every function below
  works correctly against however much is actually loaded and returns
  ok=False with an explicit "not enough data" message rather than
  fabricating a result - do not expect a full-year gold chart until the
  DAG has been running for a full year.
- Same sparsity caveat applies to forex (also "today only", same as gold).
  Oil is DIFFERENT: fetch_oil() DOES backfill a real date range from FRED
  per run (`--days N`), so oil_price_series() can have genuine historical
  depth immediately - but ONLY if FRED_API_KEY is set in .env; entirely
  empty otherwise (see README Step 11). Check which situation you're in
  before assuming "empty result" means "no oil data will ever load."
"""
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import text

from src.core.db import get_session

# Real, sourced 2025 gold/market events - NOT invented, used as reference
# points for event_window_comparison() and the case-study writeup. Each
# entry cites the fact it's grounded in (see
# docs/market_factor_case_study_gold_2025.md for full source list/URLs).
KNOWN_2025_GOLD_EVENTS = {
    "sbv_aml_inspection_selloff": {
        "date": date(2025, 5, 30),
        "description": (
            "SBV (Ngân hàng Nhà nước) công bố kết luận thanh tra phát hiện vi phạm "
            "phòng chống rửa tiền tại SJC, Bảo Tín Minh Châu, DOJI và PNJ. Cổ phiếu "
            "PNJ giảm 7/10 phiên, từ 84.000 -> 76.000 VND (-9.5%) kể từ phiên 30/5. "
            "PNJ Q1/2025: doanh thu thuần giảm 23.5% YoY, doanh thu vàng miếng 24K "
            "giảm 65.8% YoY do khó khăn nguồn nguyên liệu - một cú sốc QUY ĐỊNH, "
            "không phải cú sốc GIÁ vàng."
        ),
        "affected_tickers": ["PNJ"],
    },
    "decree_232_gold_monopoly_end": {
        "date": date(2025, 10, 10),
        "description": (
            "Nghị định 232/2025/NĐ-CP có hiệu lực, bãi bỏ độc quyền nhà nước về sản "
            "xuất vàng miếng. PNJ, DOJI, SJC là số ít doanh nghiệp đủ vốn điều lệ "
            "(>=1,000 tỷ VND) để xin giấy phép sản xuất vàng miếng - một cú sốc "
            "CHÍNH SÁCH tích cực, mở ra dòng doanh thu mới cho PNJ."
        ),
        "affected_tickers": ["PNJ"],
    },
}


def _daily_period(d: date) -> tuple[int, str]:
    """Matches scripts/market_snapshot.py's _daily_period() encoding exactly."""
    return d.year, f"D{d.month:02d}{d.day:02d}"


def _period_to_date(fiscal_year: int, period: str) -> Optional[date]:
    """Inverse of _daily_period() - 'D0715' + 2025 -> date(2025, 7, 15). Returns
    None for non-DAILY period strings (defensive - callers should already be
    filtering to period_type='DAILY')."""
    if not period.startswith("D") or len(period) != 5:
        return None
    try:
        return date(fiscal_year, int(period[1:3]), int(period[3:5]))
    except ValueError:
        return None


_GOLD_SERIES_SQL = text("""
    SELECT fiscal_year, period, AVG(value) AS value, COUNT(*) AS n_readings
    FROM macro_indicators
    WHERE category = 'MKT_GOLD_SJC' AND indicator_name LIKE :name_pattern
    GROUP BY fiscal_year, period
    ORDER BY fiscal_year, period
""")


def gold_price_series(price_type: str = "sell_price") -> pd.DataFrame:
    """Daily domestic gold price, blended across every SJC branch/type
    reported that day (see module docstring for why - no single canonical
    indicator_name exists). `price_type`: "sell_price" or "buy_price".
    Returns columns: trade_date, gold_price, n_readings (how many
    branch/type rows were averaged - low n means thin data for that day,
    worth checking before trusting it)."""
    with get_session() as session:
        rows = session.execute(_GOLD_SERIES_SQL, {"name_pattern": f"%{price_type}"}).fetchall()
    out = []
    for fy, period, value, n in rows:
        d = _period_to_date(int(fy), str(period))
        if d is not None and value is not None:
            out.append({"trade_date": d, "gold_price": float(value), "n_readings": int(n)})
    df = pd.DataFrame(out, columns=["trade_date", "gold_price", "n_readings"])
    return df.sort_values("trade_date").reset_index(drop=True)


def gold_annual_aggregate(fiscal_year: int, price_type: str = "sell_price") -> Optional[float]:
    """Average all gold_price_series() readings within one calendar year
    into a single scalar - the gold-data equivalent of
    macro_lookup.annual_aggregate(), used as an optional causal-panel
    covariate (src/causal/panel_builder.py). Can't reuse annual_aggregate()
    directly since it expects one exact (category, indicator_name) pair,
    and gold has no single canonical indicator_name (see module docstring).
    Returns None if nothing is loaded for that year - expect this for
    every year except whichever one(s) the Airflow DAG has actually been
    running in, given fetch_gold()'s "today only" data-availability limit."""
    series = gold_price_series(price_type)
    if series.empty:
        return None
    year_values = series[pd.to_datetime(series["trade_date"]).dt.year == fiscal_year]["gold_price"]
    return float(year_values.mean()) if not year_values.empty else None


_VNINDEX_SERIES_SQL = text("""
    SELECT fiscal_year, period, value
    FROM macro_indicators
    WHERE category = 'MKT_VNINDEX' AND indicator_name = 'VN-Index (close)'
    ORDER BY fiscal_year, period
""")


def vnindex_series() -> pd.DataFrame:
    """Daily VN-Index close. Columns: trade_date, vnindex_close."""
    with get_session() as session:
        rows = session.execute(_VNINDEX_SERIES_SQL).fetchall()
    out = []
    for fy, period, value in rows:
        d = _period_to_date(int(fy), str(period))
        if d is not None and value is not None:
            out.append({"trade_date": d, "vnindex_close": float(value)})
    df = pd.DataFrame(out, columns=["trade_date", "vnindex_close"])
    return df.sort_values("trade_date").reset_index(drop=True)


_FOREX_SERIES_SQL = text("""
    SELECT fiscal_year, period, AVG(value) AS value, COUNT(*) AS n_readings
    FROM macro_indicators
    WHERE category = 'MKT_FOREX_VCB' AND indicator_name LIKE :name_pattern
    GROUP BY fiscal_year, period
    ORDER BY fiscal_year, period
""")


def forex_series(currency: str = "USD", rate_type: str = "sell") -> pd.DataFrame:
    """Daily VCB exchange rate, blended the same way gold_price_series() is
    (fetch_forex() in scripts/market_snapshot.py writes one row per
    (currency, rate-column) combination - "buy"/"sell"/"transfer" columns
    whose exact names weren't confirmed against a real query result, same
    caveat as gold - see module docstring). `currency`: e.g. "USD".
    `rate_type`: substring matched against the column name, e.g. "sell".
    Returns columns: trade_date, fx_rate, n_readings."""
    with get_session() as session:
        rows = session.execute(_FOREX_SERIES_SQL, {"name_pattern": f"%{currency}%{rate_type}%"}).fetchall()
    out = []
    for fy, period, value, n in rows:
        d = _period_to_date(int(fy), str(period))
        if d is not None and value is not None:
            out.append({"trade_date": d, "fx_rate": float(value), "n_readings": int(n)})
    df = pd.DataFrame(out, columns=["trade_date", "fx_rate", "n_readings"])
    return df.sort_values("trade_date").reset_index(drop=True)


def forex_annual_aggregate(fiscal_year: int, currency: str = "USD", rate_type: str = "sell") -> Optional[float]:
    """Gold_annual_aggregate()'s counterpart for forex - see that function's
    docstring for why this can't just reuse macro_lookup.annual_aggregate()."""
    series = forex_series(currency, rate_type)
    if series.empty:
        return None
    year_values = series[pd.to_datetime(series["trade_date"]).dt.year == fiscal_year]["fx_rate"]
    return float(year_values.mean()) if not year_values.empty else None


_OIL_SERIES_SQL = text("""
    SELECT fiscal_year, period, value
    FROM macro_indicators
    WHERE category = 'MKT_OIL_WTI' AND indicator_name = 'WTI crude (Cushing, OK)'
    ORDER BY fiscal_year, period
""")


def oil_price_series() -> pd.DataFrame:
    """Daily WTI crude price (FRED). Columns: trade_date, oil_price.
    Unlike gold/forex, fetch_oil() writes ONE clean indicator_name, so no
    blending is needed here - a plain exact-match query, same shape as
    vnindex_series()."""
    with get_session() as session:
        rows = session.execute(_OIL_SERIES_SQL).fetchall()
    out = []
    for fy, period, value in rows:
        d = _period_to_date(int(fy), str(period))
        if d is not None and value is not None:
            out.append({"trade_date": d, "oil_price": float(value)})
    df = pd.DataFrame(out, columns=["trade_date", "oil_price"])
    return df.sort_values("trade_date").reset_index(drop=True)


def oil_annual_aggregate(fiscal_year: int) -> Optional[float]:
    """Unlike gold/forex's _annual_aggregate() siblings, this COULD reuse
    macro_lookup.annual_aggregate() directly (oil has one clean
    indicator_name) - kept as its own function anyway for a consistent
    call shape across all 3 market-factor covariates in panel_builder.py."""
    series = oil_price_series()
    if series.empty:
        return None
    year_values = series[pd.to_datetime(series["trade_date"]).dt.year == fiscal_year]["oil_price"]
    return float(year_values.mean()) if not year_values.empty else None


_TICKER_PRICE_SQL = text("""
    SELECT trade_date, close_price, volume
    FROM company_price_snapshot
    WHERE ticker = :ticker
      AND (:start_date IS NULL OR trade_date >= :start_date)
      AND (:end_date IS NULL OR trade_date <= :end_date)
    ORDER BY trade_date
""")


def ticker_price_series(ticker: str, start_date: date = None, end_date: date = None) -> pd.DataFrame:
    """One ticker's daily close price/volume. Columns: trade_date, close_price, volume."""
    with get_session() as session:
        rows = session.execute(_TICKER_PRICE_SQL, {
            "ticker": ticker, "start_date": start_date, "end_date": end_date,
        }).fetchall()
    df = pd.DataFrame(rows, columns=["trade_date", "close_price", "volume"])
    df["close_price"] = df["close_price"].astype(float)
    return df


_LATEST_TWO_SQL = text("""
    SELECT trade_date, open_price, high_price, low_price, close_price, volume
    FROM company_price_snapshot
    WHERE ticker = :ticker
    ORDER BY trade_date DESC
    LIMIT 2
""")


def latest_price_snapshot(ticker: str) -> dict:
    """Most recent EOD OHLCV row for `ticker` + the change vs the row before
    it, for the router's "price_snapshot" branch (rules.py - a user asking
    "biến động/giá X hôm nay thế nào" type questions).

    2026-09-21, added per explicit request after confirming NO existing
    branch answered this: compare_tickers() needs >=2 tickers (rebases to
    100, meant for relative comparison charts, not "what did ticker X do"),
    and trend.py explicitly does not touch price data at all (see its
    module docstring). text_to_sql_agent.py (Branch B, free-form LLM SQL)
    could technically reach this table but that branch is the one this
    project's own Chapter 4 evaluation found 0% reliable for structured
    lookups (0/26, see docs/PROGRESS.md 2026-09-19 discussion) - so this is
    a small, dedicated, zero-LLM-call reader instead, same shape as
    auditor_lookup.py/bank_rate_lookup.

    IMPORTANT - what this CANNOT answer: `company_price_snapshot` is
    END-OF-DAY data, written ONCE PER DAY by scripts/market_snapshot.py
    (Airflow DAG `thesis_market_snapshot`) - there is no intraday/tick-level
    price anywhere in this project. So "biến động trong ngày hôm nay" in
    the sense of live/intraday movement is NOT answerable - only "so với
    phiên giao dịch gần nhất có dữ liệu". The `is_stale` flag below tells
    the caller whether the latest row IS today (calendar) or an earlier
    trading day, so the formatted answer can say so honestly instead of
    implying real-time freshness it doesn't have.
    """
    with get_session() as session:
        rows = session.execute(_LATEST_TWO_SQL, {"ticker": ticker.upper()}).mappings().all()
    if not rows:
        return {
            "ok": False,
            "error": (f"Không có dòng nào trong company_price_snapshot cho mã {ticker.upper()} - "
                      f"có thể chưa được nạp qua Airflow DAG thesis_market_snapshot cho mã này."),
        }
    latest = dict(rows[0])
    prev = dict(rows[1]) if len(rows) > 1 else None
    pct_change = None
    abs_change = None
    if prev and prev.get("close_price") not in (None, 0) and latest.get("close_price") is not None:
        abs_change = float(latest["close_price"] - prev["close_price"])
        pct_change = float(abs_change / float(prev["close_price"]) * 100)
    days_stale = (date.today() - latest["trade_date"]).days
    return {
        "ok": True,
        "ticker": ticker.upper(),
        "trade_date": latest["trade_date"],
        "open_price": float(latest["open_price"]) if latest["open_price"] is not None else None,
        "high_price": float(latest["high_price"]) if latest["high_price"] is not None else None,
        "low_price": float(latest["low_price"]) if latest["low_price"] is not None else None,
        "close_price": float(latest["close_price"]) if latest["close_price"] is not None else None,
        "volume": int(latest["volume"]) if latest["volume"] is not None else None,
        "prev_trade_date": prev["trade_date"] if prev else None,
        "abs_change": abs_change,
        "pct_change": pct_change,
        "days_stale": days_stale,  # 0 = latest row IS today; >0 = data lags by that many calendar days
        "is_stale": days_stale > 1,  # >1 not >0: a weekend/holiday gap of 1-2 days is normal for EOD data, not a pipeline problem
    }


_PRICE_WINDOW_SQL = text("""
    SELECT trade_date, open_price, high_price, low_price, close_price, volume
    FROM company_price_snapshot
    WHERE ticker = :ticker
    ORDER BY trade_date DESC
    LIMIT :limit
""")


def price_window_series(ticker: str, days: int = 7) -> dict:
    """The last `days` EOD rows for `ticker`, newest first, with per-session
    and whole-window changes computed.

    2026-09-22, added after a real failure: "Cho tôi biết giá cả của VNZ
    trong vài ngày gần đây" was routed to RAG (26s, then an Ollama fallback
    message) even though the data sat in company_price_snapshot the whole
    time. latest_price_snapshot() above only ever returns ONE session, so it
    could not answer a question about a window - and rules.py's
    PRICE_TODAY_MARKERS only matched "hôm nay"/"today", so a multi-day
    question never even reached this module. This function fills the data
    half of that gap; rules.py's PRICE_RECENT_MARKERS fills the routing half.

    LIMIT counts ROWS (trading sessions), not calendar days - deliberately.
    Asking for 7 "days" over a weekend would otherwise silently return 5
    rows and look like missing data; returning the last 7 SESSIONS is both
    what a finance user means and what the table can honestly provide. The
    returned `first_trade_date`/`last_trade_date` say exactly which calendar
    span those sessions cover, so the answer never has to guess.

    Same honesty contract as latest_price_snapshot(): EOD only, no intraday
    anywhere in this project, and `is_stale` reports whether the newest row
    is actually recent.
    """
    limit = max(2, min(int(days), 60))  # >=2 để tính được thay đổi; <=60 để không đổ một mớ vào câu trả lời chat
    with get_session() as session:
        rows = session.execute(
            _PRICE_WINDOW_SQL, {"ticker": ticker.upper(), "limit": limit},
        ).mappings().all()
    if not rows:
        return {
            "ok": False,
            "error": (f"Không có dòng nào trong company_price_snapshot cho mã {ticker.upper()} - "
                      f"có thể chưa được nạp qua Airflow DAG thesis_market_snapshot cho mã này."),
        }

    # rows: mới -> cũ. Đảo lại thành cũ -> mới để tính %thay đổi theo thời gian
    # thuận, rồi mới trình bày ngược lại ở formatter (người đọc muốn thấy
    # phiên gần nhất trước).
    ordered = [dict(r) for r in reversed(rows)]
    sessions = []
    prev_close = None
    for r in ordered:
        close = float(r["close_price"]) if r["close_price"] is not None else None
        pct = None
        if prev_close not in (None, 0) and close is not None:
            pct = (close - prev_close) / prev_close * 100
        sessions.append({
            "trade_date": r["trade_date"],
            "open_price": float(r["open_price"]) if r["open_price"] is not None else None,
            "high_price": float(r["high_price"]) if r["high_price"] is not None else None,
            "low_price": float(r["low_price"]) if r["low_price"] is not None else None,
            "close_price": close,
            "volume": int(r["volume"]) if r["volume"] is not None else None,
            "pct_change": pct,
        })
        if close is not None:
            prev_close = close

    closes = [s["close_price"] for s in sessions if s["close_price"] is not None]
    highs = [s["high_price"] for s in sessions if s["high_price"] is not None]
    lows = [s["low_price"] for s in sessions if s["low_price"] is not None]
    net_abs = net_pct = None
    if len(closes) >= 2 and closes[0]:
        net_abs = closes[-1] - closes[0]
        net_pct = net_abs / closes[0] * 100

    newest = sessions[-1]["trade_date"]
    days_stale = (date.today() - newest).days
    return {
        "ok": True,
        "ticker": ticker.upper(),
        "n_sessions": len(sessions),
        "requested_days": limit,
        "first_trade_date": sessions[0]["trade_date"],
        "last_trade_date": newest,
        # newest first - thứ tự người đọc muốn thấy
        "sessions": list(reversed(sessions)),
        "net_abs_change": net_abs,
        "net_pct_change": net_pct,
        "window_high": max(highs) if highs else (max(closes) if closes else None),
        "window_low": min(lows) if lows else (min(closes) if closes else None),
        "days_stale": days_stale,
        "is_stale": days_stale > 1,
    }


_PRICE_RANGE_SQL = text("""
    SELECT trade_date, open_price, high_price, low_price, close_price, volume
    FROM company_price_snapshot
    WHERE ticker = :ticker
      AND trade_date >= :start_date
      AND trade_date <= :end_date
    ORDER BY trade_date
""")


def price_range_series(ticker: str, start_date: date, end_date: date) -> dict:
    """Mọi phiên của `ticker` trong khoảng ngày TƯỜNG MINH, kèm tóm tắt tháng.

    2026-09-24, vá nốt nửa còn lại của lỗ giá: price_window_series() ở trên
    trả lời được "vài ngày gần đây" nhưng không trả lời được "từ tháng
    10/2025 đến tháng 1/2026" - câu đó vẫn rơi xuống RAG rồi chết ở Ollama.
    rules.py's _price_date_range() bóc mốc ngày; hàm này đọc dữ liệu.

    Trả về CẢ hai mức chi tiết và để formatter chọn: một khoảng 4 tháng là
    ~80 phiên, in hết ra chat thì không ai đọc nổi, nên gộp theo tháng mới
    dùng được; nhưng một khoảng 2 tuần thì liệt kê từng phiên lại đúng hơn.
    Quyết định đó thuộc về cách TRÌNH BÀY, nên để ở formatter, không chôn
    vào tầng đọc dữ liệu.

    Cùng hợp đồng trung thực như hai hàm giá kia: EOD, không hề có dữ liệu
    trong ngày, và `requested_start`/`requested_end` giữ nguyên mốc người
    dùng hỏi để câu trả lời nói được "bạn hỏi tới 31/01 nhưng dữ liệu chỉ
    có tới 24/01" thay vì im lặng trả ít hơn.
    """
    with get_session() as session:
        rows = session.execute(_PRICE_RANGE_SQL, {
            "ticker": ticker.upper(), "start_date": start_date, "end_date": end_date,
        }).mappings().all()
    if not rows:
        return {
            "ok": False,
            "requested_start": start_date, "requested_end": end_date,
            "error": (f"Không có phiên giao dịch nào của {ticker.upper()} trong khoảng "
                      f"{start_date} - {end_date} trong company_price_snapshot."),
        }

    sessions = []
    prev_close = None
    for r in rows:                                  # đã ORDER BY cũ -> mới
        close = float(r["close_price"]) if r["close_price"] is not None else None
        pct = None
        if prev_close not in (None, 0) and close is not None:
            pct = (close - prev_close) / prev_close * 100
        sessions.append({
            "trade_date": r["trade_date"],
            "open_price": float(r["open_price"]) if r["open_price"] is not None else None,
            "high_price": float(r["high_price"]) if r["high_price"] is not None else None,
            "low_price": float(r["low_price"]) if r["low_price"] is not None else None,
            "close_price": close,
            "volume": int(r["volume"]) if r["volume"] is not None else None,
            "pct_change": pct,
        })
        if close is not None:
            prev_close = close

    # gộp theo tháng, giữ thứ tự thời gian
    months: dict[tuple, dict] = {}
    for s in sessions:
        key = (s["trade_date"].year, s["trade_date"].month)
        m = months.setdefault(key, {
            "year": key[0], "month": key[1], "n_sessions": 0,
            "first_close": None, "last_close": None,
            "high": None, "low": None, "volume_sum": 0,
        })
        m["n_sessions"] += 1
        if s["close_price"] is not None:
            if m["first_close"] is None:
                m["first_close"] = s["close_price"]
            m["last_close"] = s["close_price"]
        for src, key2, fn in (("high_price", "high", max), ("low_price", "low", min)):
            v = s[src] if s[src] is not None else s["close_price"]
            if v is not None:
                m[key2] = v if m[key2] is None else fn(m[key2], v)
        if s["volume"] is not None:
            m["volume_sum"] += s["volume"]
    for m in months.values():
        m["pct_change"] = (
            (m["last_close"] - m["first_close"]) / m["first_close"] * 100
            if m["first_close"] not in (None, 0) and m["last_close"] is not None else None
        )

    closes = [s["close_price"] for s in sessions if s["close_price"] is not None]
    highs = [s["high_price"] for s in sessions if s["high_price"] is not None]
    lows = [s["low_price"] for s in sessions if s["low_price"] is not None]
    net_abs = net_pct = None
    if len(closes) >= 2 and closes[0]:
        net_abs = closes[-1] - closes[0]
        net_pct = net_abs / closes[0] * 100

    return {
        "ok": True,
        "ticker": ticker.upper(),
        "requested_start": start_date,
        "requested_end": end_date,
        "first_trade_date": sessions[0]["trade_date"],
        "last_trade_date": sessions[-1]["trade_date"],
        "n_sessions": len(sessions),
        "sessions": list(reversed(sessions)),        # mới -> cũ, đúng thứ tự người đọc muốn
        "months": [months[k] for k in sorted(months)],
        "net_abs_change": net_abs,
        "net_pct_change": net_pct,
        "window_high": max(highs) if highs else (max(closes) if closes else None),
        "window_low": min(lows) if lows else (min(closes) if closes else None),
        # Dữ liệu có phủ hết khoảng người dùng hỏi không - để nói thẳng ra
        # thay vì lặng lẽ trả về ít hơn.
        "covers_start": sessions[0]["trade_date"] <= start_date + timedelta(days=7),
        "covers_end": sessions[-1]["trade_date"] >= end_date - timedelta(days=7),
    }


def compare_tickers(tickers: list[str], start_date: date = None, end_date: date = None) -> dict:
    """Rebase each ticker's close price to 100 at its first available date
    in range, for direct visual comparison regardless of absolute price
    level (a 500,000 VND stock and a 10,000 VND stock plotted on raw price
    would be visually meaningless together). Returns:
    {"ok", "wide": DataFrame[trade_date, <TICKER1>, <TICKER2>, ...], "missing": [tickers with no data]}
    """
    frames = {}
    missing = []
    for t in tickers:
        df = ticker_price_series(t, start_date, end_date)
        if df.empty:
            missing.append(t)
            continue
        base = df["close_price"].iloc[0]
        if base == 0:
            missing.append(t)
            continue
        s = df.set_index("trade_date")["close_price"] / base * 100
        frames[t] = s

    if not frames:
        return {"ok": False, "error": f"Không có dữ liệu giá cho bất kỳ mã nào trong {tickers}.", "missing": missing}

    wide = pd.DataFrame(frames)
    wide = wide.sort_index().reset_index().rename(columns={"index": "trade_date"})
    return {"ok": True, "wide": wide, "missing": missing}


def _returns(series: pd.Series) -> pd.Series:
    """Daily % change - see module docstring for why correlations use
    returns, not raw levels."""
    return series.pct_change().dropna()


def correlate_series(a: pd.DataFrame, a_value_col: str, b: pd.DataFrame, b_value_col: str) -> dict:
    """Pearson correlation of two DAILY RETURN series, joined on trade_date
    (inner join - only dates present in BOTH series count). Both `a`/`b`
    must have a `trade_date` column. Returns ok=False if fewer than 5
    overlapping dates - not enough points for a correlation to mean
    anything."""
    merged = pd.merge(a[["trade_date", a_value_col]], b[["trade_date", b_value_col]], on="trade_date", how="inner")
    merged = merged.sort_values("trade_date")
    if len(merged) < 6:
        return {"ok": False, "error": f"Chỉ có {len(merged)} ngày trùng khớp giữa 2 chuỗi - cần >=6 để tính tương quan có ý nghĩa."}

    ra = _returns(merged.set_index("trade_date")[a_value_col])
    rb = _returns(merged.set_index("trade_date")[b_value_col])
    joined = pd.concat([ra, rb], axis=1, keys=["a", "b"]).dropna()
    if len(joined) < 5:
        return {"ok": False, "error": f"Chỉ có {len(joined)} cặp lợi suất ngày trùng khớp - không đủ để tính tương quan."}

    corr = float(joined["a"].corr(joined["b"]))
    return {
        "ok": True, "correlation": corr, "n_days": len(joined),
        "date_range": [str(joined.index.min()), str(joined.index.max())],
        "note": "Tương quan tính trên LỢI SUẤT NGÀY (% thay đổi), không phải mức giá thô.",
    }


def correlate_ticker_with_gold(ticker: str, start_date: date = None, end_date: date = None,
                                price_type: str = "sell_price") -> dict:
    prices = ticker_price_series(ticker, start_date, end_date)
    if prices.empty:
        return {"ok": False, "error": f"Không có dữ liệu giá cho {ticker}."}
    gold = gold_price_series(price_type)
    if gold.empty:
        return {"ok": False, "error": "Chưa có dữ liệu giá vàng (MKT_GOLD_SJC) trong macro_indicators."}
    result = correlate_series(prices, "close_price", gold, "gold_price")
    if result["ok"]:
        result["ticker"] = ticker
        result["compared_with"] = f"gold_{price_type}"
    return result


def correlate_ticker_with_vnindex(ticker: str, start_date: date = None, end_date: date = None) -> dict:
    prices = ticker_price_series(ticker, start_date, end_date)
    if prices.empty:
        return {"ok": False, "error": f"Không có dữ liệu giá cho {ticker}."}
    vni = vnindex_series()
    if vni.empty:
        return {"ok": False, "error": "Chưa có dữ liệu VN-Index (MKT_VNINDEX) trong macro_indicators."}
    result = correlate_series(prices, "close_price", vni, "vnindex_close")
    if result["ok"]:
        result["ticker"] = ticker
        result["compared_with"] = "vnindex"
    return result


def correlate_ticker_with_oil(ticker: str, start_date: date = None, end_date: date = None) -> dict:
    """Same pattern as correlate_ticker_with_gold() - most relevant for
    GAS (see calculator.OIL_SENSITIVE_TICKERS)."""
    prices = ticker_price_series(ticker, start_date, end_date)
    if prices.empty:
        return {"ok": False, "error": f"Không có dữ liệu giá cho {ticker}."}
    oil = oil_price_series()
    if oil.empty:
        return {"ok": False, "error": "Chưa có dữ liệu giá dầu (MKT_OIL_WTI) trong macro_indicators - "
                                       "cần đặt FRED_API_KEY trong .env, xem README Step 11."}
    result = correlate_series(prices, "close_price", oil, "oil_price")
    if result["ok"]:
        result["ticker"] = ticker
        result["compared_with"] = "oil_wti"
    return result


def correlate_ticker_with_forex(ticker: str, start_date: date = None, end_date: date = None,
                                 currency: str = "USD", rate_type: str = "sell") -> dict:
    """Same pattern as correlate_ticker_with_gold(). Relevant for companies
    with meaningful USD exposure (importers, USD-denominated debt/costs) -
    deliberately NOT pre-mapped to specific tickers the way
    GOLD_SENSITIVE_TICKERS/OIL_SENSITIVE_TICKERS are, since real forex
    exposure varies a lot by company and isn't something to guess at
    without checking each company's actual disclosures first."""
    prices = ticker_price_series(ticker, start_date, end_date)
    if prices.empty:
        return {"ok": False, "error": f"Không có dữ liệu giá cho {ticker}."}
    fx = forex_series(currency, rate_type)
    if fx.empty:
        return {"ok": False, "error": "Chưa có dữ liệu tỷ giá (MKT_FOREX_VCB) trong macro_indicators."}
    result = correlate_series(prices, "close_price", fx, "fx_rate")
    if result["ok"]:
        result["ticker"] = ticker
        result["compared_with"] = f"forex_{currency}_{rate_type}"
    return result


def event_window_comparison(tickers: list[str], event_date: date, days_before: int = 5, days_after: int = 5) -> dict:
    """% change in each ticker's close price from the last available trade
    date <= (event_date - days_before) to the last available trade date <=
    (event_date + days_after) - a simple event-study window, NOT a causal
    claim (no control group, no confounder adjustment - see
    src/causal/treatment_effect.py for the project's actual causal
    estimation tool). Useful for "what happened to these tickers around
    this specific dated event" scenario questions.
    """
    start_window = event_date - timedelta(days=days_before)
    end_window = event_date + timedelta(days=days_after)

    results = {}
    for t in tickers:
        df = ticker_price_series(t, start_window, end_window)
        if df.empty:
            results[t] = {"ok": False, "error": "Không có dữ liệu giá trong khung thời gian này."}
            continue
        before = df[df["trade_date"] <= event_date]
        after = df[df["trade_date"] > event_date]
        if before.empty or after.empty:
            results[t] = {"ok": False, "error": "Thiếu dữ liệu trước hoặc sau ngày sự kiện trong khung đã chọn."}
            continue
        p1, p2 = float(before["close_price"].iloc[-1]), float(after["close_price"].iloc[-1])
        pct = (p2 - p1) / p1 * 100 if p1 else None
        results[t] = {
            "ok": True, "price_before": p1, "price_after": p2, "pct_change": pct,
            "date_before": str(before["trade_date"].iloc[-1]), "date_after": str(after["trade_date"].iloc[-1]),
        }
    return {"event_date": str(event_date), "window": [str(start_window), str(end_window)], "results": results}


if __name__ == "__main__":
    import json
    import sys

    tickers = sys.argv[1:] or ["PNJ", "GAS"]
    print(f"=== Gold price series (blended, sell_price) ===")
    g = gold_price_series()
    print(f"{len(g)} daily readings loaded" + (f", latest: {g.iloc[-1].to_dict()}" if not g.empty else " (none yet)"))

    print(f"\n=== Forex (USD/VND, sell) series ===")
    fx = forex_series()
    print(f"{len(fx)} daily readings loaded" + (f", latest: {fx.iloc[-1].to_dict()}" if not fx.empty else " (none yet)"))

    print(f"\n=== Oil (WTI) series ===")
    oil = oil_price_series()
    print(f"{len(oil)} daily readings loaded" + (f", latest: {oil.iloc[-1].to_dict()}" if not oil.empty else " (none yet - check FRED_API_KEY in .env)"))

    for t in tickers:
        print(f"\n=== {t} vs gold (returns correlation) ===")
        print(json.dumps(correlate_ticker_with_gold(t), indent=2, ensure_ascii=False))
        print(f"\n=== {t} vs VN-Index (returns correlation) ===")
        print(json.dumps(correlate_ticker_with_vnindex(t), indent=2, ensure_ascii=False))
        print(f"\n=== {t} vs oil (returns correlation) ===")
        print(json.dumps(correlate_ticker_with_oil(t), indent=2, ensure_ascii=False))

    print(f"\n=== Compare {tickers} (rebased to 100) ===")
    cmp = compare_tickers(tickers)
    if cmp["ok"]:
        print(cmp["wide"].tail(10).to_string(index=False))
    else:
        print(cmp["error"])
