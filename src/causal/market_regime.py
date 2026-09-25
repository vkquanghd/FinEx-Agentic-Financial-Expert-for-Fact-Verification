"""RQ3 (Trend-Aligned Dynamic Reranking) - a market-regime signal computed
from real VN-Index data already loaded via market_snapshot.py, consumed by
src/retrieval/rag_pipeline.rerank_for_regime() to adjust RAG chunk ranking.

Academic basis: "History Rhymes: Macro-Contextual Retrieval for Robust
Financial Forecasting" (arXiv:2511.09754) embeds macro indicators (CPI,
GDP, yields...) in the SAME embedding space as text so retrieval adapts to
the prevailing macro regime, improving robustness across regime shifts.
A full joint macro-text embedding is out of scope for the remaining weeks
(disclosure_chunks has no per-chunk macro tagging, and building one would
need a new embedding pipeline this late in the schedule) - this module
implements the same underlying idea (regime-conditioned retrieval) via a
much cheaper mechanism: classify the CURRENT market regime from VN-Index's
own recent trend/volatility, and let that classification tighten the
RECENCY weight applied when reranking RAG candidates (see
rag_pipeline.rerank_for_regime()). This is exactly RQ3's own wording in
Thesis_Direction_Redesign.pdf section 3 - "trọng số routing điều chỉnh
theo chu kỳ thị trường, dựa trên macro_indicators".

Why VN-Index and not gold/oil/forex: VN-Index is the only series with
broad daily coverage - see market_factors.py's data-sparsity caveats for
gold/forex ("today only" snapshots that only grow ~1 row per Airflow run).
A regime classifier needs enough trailing history to compute a trend over
`window_days`, which only vnindex_series() can currently supply.

Deliberately rule-based, not ML/fine-tuned: consistent with section 9 of
Thesis_Direction_Redesign.pdf ("Toàn bộ hệ thống dùng model
pretrained/prompt-based, không fine-tune bất kỳ thành phần nào") and with
market_factors.py's existing "not enough data -> ok=False, don't fabricate"
philosophy.

**Percentile-based classification (added 2026-07-26), not fixed thresholds
anymore when enough history exists.** The original version (2026-07-24)
compared trend/volatility against hardcoded absolute thresholds (+-5% /
2.5%), justified at the time as "not enough VN-Index history loaded to fit
anything against". That justification turned out to be stale, not
re-checked against what was actually already loaded: a real backfill on
2026-07-18 (`scripts/market_snapshot.py --days 365 --source KBS`, see
docs/PROGRESS.md) had already landed 249 real VN-Index trading days six
days before this module was first written - enough to build a real
trailing empirical distribution of rolling trend/volatility, not enough to
justify a hardcoded absolute cutoff as the *only* option. This version
computes rolling `window_days` trend%/volatility% over the FULL VN-Index
history available as of `as_of` (or today), and classifies the current
window by its PERCENTILE RANK against that trailing distribution (e.g.
"today's volatility is higher than 85% of all `window_days`-day stretches
seen so far") rather than a fixed absolute number - this is what actually
makes the classifier "adaptive" in the sense the thesis title claims: as
more VN-Index history accumulates via the Airflow DAG, the classification
boundary moves with it instead of staying frozen at a number chosen once
in July 2026.

Percentile, not z-score: z-score assumes the trend/volatility distribution
is roughly normal, which daily-return volatility in particular is not
(well-documented fat tails/clustering in financial return series) -
percentile rank (a plain empirical CDF, no distributional assumption) is
the more defensible choice for a sample this size (a few hundred trading
days, one calendar year, not multiple bull/bear cycles).

Still not fit/tuned in the ML sense: the PERCENTILE CUTOFFS themselves
(BULL_PCTL/BEAR_PCTL/VOLATILE_PCTL below, e.g. "top 20%") are still a
documented judgment call, same honesty as before - what changed is that the
number being compared against that cutoff is now the data's own trailing
distribution instead of an arbitrary absolute percentage.

Falls back to the original fixed-threshold logic when there isn't enough
rolling-window history yet to trust a percentile (see MIN_HISTORY_WINDOWS)
- e.g. a fresh deployment that just started the `thesis_market_snapshot`
DAG and hasn't run the one-time 365-day backfill yet. This fallback is not
a lesser-tested code path bolted on afterward: it is the exact same logic
this module used exclusively before 2026-07-26, so it inherits that
history's verification (see docs/PROGRESS.md, Week 5 entry).
"""
from datetime import date
from typing import Optional

from src.causal.market_factors import vnindex_series

# Same "not enough points to mean anything" floor correlate_series() uses.
MIN_TRADING_DAYS = 6

# Minimum number of PRIOR rolling-window observations required before the
# percentile method is trusted over the fixed-threshold fallback. 60 is a
# round, documented judgment call (roughly 3 trading months of rolling
# windows) - enough that a percentile rank means something more than "where
# does today rank among the last dozen days", not fit against any labelled
# regime data (none exists for this project, see module docstring).
MIN_HISTORY_WINDOWS = 60

# Percentile cutoffs used once enough history exists (see MIN_HISTORY_WINDOWS).
# Documented judgment calls, not fit to data - see module docstring for why
# these replace, rather than are derived from, the original fixed
# thresholds. "Volatile" and "bull" both use the SAME 0.80 cutoff
# deliberately (both mean "unusually far from the ordinary" on their
# respective distribution, just in different directions) - "bear" is
# reflected around 0.20 as the natural mirror of "bull", not an
# independent choice.
BULL_PCTL = 0.80
BEAR_PCTL = 0.20
VOLATILE_PCTL = 0.80

# Fixed-threshold fallback, used only when history is too short for the
# percentile method (MIN_HISTORY_WINDOWS not met) - this is the module's
# original (2026-07-24) logic, kept verbatim as a documented fallback, not
# re-derived. Round, documented judgment calls, NOT fit to data. Chosen to
# roughly match commonly-cited VN-Index regime language: a >=5% move over
# the window is a real directional trend, >=2.5% daily-return std is a
# genuinely choppy/volatile stretch.
BULL_TREND_PCT = 5.0
BEAR_TREND_PCT = -5.0
VOLATILE_STD_PCT = 2.5


def _classify_by_fixed_threshold(trend_pct: float, volatility_pct: float) -> str:
    """The module's original (2026-07-24) logic, factored out unchanged so
    classify_market_regime() can fall back to it when there isn't enough
    rolling-window history for the percentile method (see
    MIN_HISTORY_WINDOWS). Volatility checked BEFORE trend direction: a
    choppy stretch that happens to end up net-flat should still read as
    "volatile", not "stable" - a flat net trend can hide a wild round-trip
    within the window, which is exactly the situation where trusting older
    chunks is riskiest."""
    if volatility_pct >= VOLATILE_STD_PCT:
        return "volatile"
    if trend_pct >= BULL_TREND_PCT:
        return "bull"
    if trend_pct <= BEAR_TREND_PCT:
        return "bear"
    return "stable"


def classify_market_regime(window_days: int = 20, as_of: Optional[date] = None) -> dict:
    """Classify the current VN-Index regime from its own trailing
    trend/volatility. Returns:
    {"ok", "regime", "trend_pct", "volatility_pct", "n_days", "window_days",
    "method", "reason"} plus "trend_percentile"/"volatility_percentile"
    when `method == "percentile"`.

    `regime` in {"bull", "bear", "volatile", "stable"} when ok=True, else
    "unknown" with ok=False and an explicit `reason` - mirrors
    market_factors.py's "don't fabricate on thin data" pattern rather than
    guessing a regime off too few trading days.

    `method` is "percentile" (current window ranked against the full
    trailing distribution of rolling `window_days` windows, see module
    docstring) when at least MIN_HISTORY_WINDOWS prior rolling windows
    exist, else "fixed_threshold_fallback" (the original absolute-percent
    thresholds) when history is too short to trust a percentile - always
    surfaced explicitly in the return value so callers/logs never silently
    mix results computed two different ways."""
    series = vnindex_series()
    if series.empty:
        return {
            "ok": False, "regime": "unknown",
            "reason": "Chưa có dữ liệu VN-Index (MKT_VNINDEX) trong macro_indicators.",
        }

    if as_of is not None:
        series = series[series["trade_date"] <= as_of]
    window = series.tail(window_days)
    if len(window) < MIN_TRADING_DAYS:
        return {
            "ok": False, "regime": "unknown",
            "reason": (
                f"Chỉ có {len(window)} phiên VN-Index gần nhất - cần >= "
                f"{MIN_TRADING_DAYS} phiên để phân loại chu kỳ thị trường."
            ),
        }

    closes = window["vnindex_close"]
    trend_pct = float((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100) if closes.iloc[0] else 0.0
    daily_returns = closes.pct_change().dropna()
    volatility_pct = float(daily_returns.std() * 100) if len(daily_returns) >= 2 else 0.0

    # Build the trailing distribution of rolling window_days trend%/vol%
    # over the FULL history available (up to as_of), to rank the current
    # window against it - same window_days as the current-window
    # calculation above, so "today" and "history" are measured the same way.
    all_closes = series["vnindex_close"]
    rolling_trend = (all_closes / all_closes.shift(window_days - 1) - 1) * 100
    rolling_vol = all_closes.pct_change().rolling(window_days).std() * 100
    rolling_trend = rolling_trend.dropna()
    rolling_vol = rolling_vol.dropna()

    # Exclude the current window itself from the baseline it's ranked
    # against - ranking a value against a distribution that already
    # contains it is a mild tautology, avoidable for free here, and keeps
    # this consistent with "classify today using only what was knowable
    # before today" (matters if as_of is used for backtesting).
    trend_baseline = rolling_trend.iloc[:-1] if len(rolling_trend) else rolling_trend
    vol_baseline = rolling_vol.iloc[:-1] if len(rolling_vol) else rolling_vol

    if len(trend_baseline) >= MIN_HISTORY_WINDOWS and len(vol_baseline) >= MIN_HISTORY_WINDOWS:
        trend_percentile = float((trend_baseline < trend_pct).mean())
        vol_percentile = float((vol_baseline < volatility_pct).mean())

        if vol_percentile >= VOLATILE_PCTL:
            regime = "volatile"
        elif trend_percentile >= BULL_PCTL:
            regime = "bull"
        elif trend_percentile <= BEAR_PCTL:
            regime = "bear"
        else:
            regime = "stable"

        return {
            "ok": True, "regime": regime, "trend_pct": round(trend_pct, 2),
            "volatility_pct": round(volatility_pct, 2), "n_days": len(window),
            "window_days": window_days, "method": "percentile",
            "trend_percentile": round(trend_percentile, 4),
            "volatility_percentile": round(vol_percentile, 4),
            "n_history_windows": len(trend_baseline),
            "reason": (
                f"VN-Index {window_days} phiên gần nhất: trend={trend_pct:+.2f}% "
                f"(percentile {trend_percentile:.0%} so với {len(trend_baseline)} cửa sổ "
                f"lịch sử), biến động ngày={volatility_pct:.2f}% (percentile "
                f"{vol_percentile:.0%})."
            ),
        }

    # Not enough rolling-window history yet - fall back to the original
    # fixed-threshold logic, explicitly labelled as such.
    regime = _classify_by_fixed_threshold(trend_pct, volatility_pct)
    return {
        "ok": True, "regime": regime, "trend_pct": round(trend_pct, 2),
        "volatility_pct": round(volatility_pct, 2), "n_days": len(window),
        "window_days": window_days, "method": "fixed_threshold_fallback",
        "n_history_windows": len(trend_baseline),
        "reason": (
            f"VN-Index {window_days} phiên gần nhất: trend={trend_pct:+.2f}%, "
            f"biến động ngày={volatility_pct:.2f}%. Chỉ có {len(trend_baseline)} "
            f"cửa sổ lịch sử (< {MIN_HISTORY_WINDOWS}) nên dùng ngưỡng cố định "
            f"thay vì percentile."
        ),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(classify_market_regime(), indent=2, ensure_ascii=False))
