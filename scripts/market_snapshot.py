"""Daily "real-time market factor" ingestion (Step 11): VN-Index, domestic
gold price, USD/VND exchange rate, and per-company daily price/volume for
every ticker in `companies` - all via vnstock (no login). Oil price (WTI)
via FRED, only if FRED_API_KEY is set (see .env.example). SBV policy
interest rate is NOT implemented yet - see the fetch_sbv_rate() stub below
for why.

IMPORTANT - unlike every other script in this repo, this one was written
WITHOUT being run against the real vnstock package first: the sandbox that
wrote it can't reach PyPI (proxy blocks pypi.org), so `pip install vnstock`
failed here. The code below matches vnstock's documented API
(https://github.com/thinh-vu/vnstock, vnstock.explorer.misc), but treat the
FIRST real run on your machine as a test, not a confirmed-working script -
report back whatever error you see and it'll get fixed, same as
pdf_ingest_router.py's poppler/dotenv/venv issues earlier.

Design (matches macro_etl.py's resilience pattern): each data source is
fetched independently and wrapped in try/except in main() - if vnstock's
API has drifted for one source (e.g. gold), the others still run and you
get a clear per-source SKIPPED/ERROR message instead of the whole script
dying on the first problem.

VN-Index / gold / forex / oil go into the EXISTING macro_indicators table
(period_type='DAILY'), using NEW category codes prefixed MKT_ so they never
collide with the manually-loaded Vietstock export categories already in
that table (e.g. USDVND_CENTRAL_RATE from macro_etl.py is a DIFFERENT
category from MKT_FOREX_USDVND here - same real-world quantity, different
source, kept separate on purpose so you can compare them later if you want).
Per-company price goes into the NEW company_price_snapshot table (see
infra/mysql/migrations/003_market_snapshot.sql - apply it by hand first).

Usage:
    python -m scripts.market_snapshot                  # last 5 trading days, all sources
    python -m scripts.market_snapshot --days 365        # backfill (first run)
    python -m scripts.market_snapshot --skip-prices      # macro factors only (fast)
    python -m scripts.market_snapshot --tickers ACB,VNM  # subset, for testing
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.core.config import FRED_API_KEY
from src.core.db import get_session

UPSERT_MACRO = text("""
    INSERT INTO macro_indicators
        (category, indicator_name, unit, period_type, fiscal_year, period, value, source_file)
    VALUES
        (:category, :indicator_name, :unit, 'DAILY', :fiscal_year, :period, :value, :source_file)
    ON CONFLICT (category, indicator_name, period_type, fiscal_year, period)
    DO UPDATE SET
        unit = EXCLUDED.unit, value = EXCLUDED.value, source_file = EXCLUDED.source_file
""")

UPSERT_PRICE = text("""
    INSERT INTO company_price_snapshot
        (ticker, trade_date, open_price, high_price, low_price, close_price, volume, source)
    VALUES
        (:ticker, :trade_date, :open_price, :high_price, :low_price, :close_price, :volume, :source)
    ON CONFLICT (ticker, trade_date)
    DO UPDATE SET
        open_price = EXCLUDED.open_price, high_price = EXCLUDED.high_price,
        low_price = EXCLUDED.low_price, close_price = EXCLUDED.close_price,
        volume = EXCLUDED.volume, source = EXCLUDED.source
""")

INSERT_LOG = text("""
    INSERT INTO ingestion_log (file_name, file_type, ticker, row_count, notes)
    VALUES (:file_name, 'market_snapshot', NULL, :row_count, :notes)
""")


def _daily_period(d: date):
    """Same DAILY period encoding macro_etl.py uses: fiscal_year=year, period='DMMDD'."""
    return d.year, f"D{d.month:02d}{d.day:02d}"


def get_tickers() -> list:
    with get_session() as session:
        rows = session.execute(text("SELECT ticker FROM companies ORDER BY ticker")).fetchall()
    return [r[0] for r in rows]


def fetch_vnindex(days: int, source: str = "KBS") -> int:
    """VN-Index daily close -> macro_indicators category=MKT_VNINDEX.

    Uses `Quote` directly, NOT `Vnstock().stock(...)` - the latter is
    vnstock's deprecated class-based wrapper (EOL 31/08/2026 per the
    library's own runtime deprecation notice) and, on a real run, calling
    it with a non-stock symbol (VNINDEX is an index, not a stock) printed
    "Not a stock. Company and finance data unavailable" and then
    SEGFAULTED - the wrapper's extra company/finance-data lookup logic
    appears to crash hard on index symbols instead of raising a normal
    Python exception. `Quote(symbol, source).history(...)` is the same
    data with none of that wrapper logic - see vnstock's current docs
    (https://vnstocks.com/docs/vnstock/thong-ke-gia-lich-su)."""
    from vnstock import Quote

    start = (date.today() - timedelta(days=days)).isoformat()
    end = date.today().isoformat()
    df = Quote(symbol="VNINDEX", source=source).history(start=start, end=end, interval="1D")

    n = 0
    with get_session() as session:
        for _, row in df.iterrows():
            d = row["time"].date() if hasattr(row["time"], "date") else date.fromisoformat(str(row["time"])[:10])
            fy, period = _daily_period(d)
            session.execute(UPSERT_MACRO, {
                "category": "MKT_VNINDEX", "indicator_name": "VN-Index (close)",
                "unit": "điểm", "fiscal_year": fy, "period": period,
                "value": float(row["close"]), "source_file": "vnstock_vci",
            })
            n += 1
    return n


def fetch_gold() -> int:
    """Domestic gold price (SJC) -> macro_indicators category=MKT_GOLD_SJC.
    sjc_gold_price() returns today's snapshot (buy/sell by branch/type), not
    a date-range history, so this always writes today's date only."""
    from vnstock.explorer.misc import sjc_gold_price

    df = sjc_gold_price()
    d = date.today()
    fy, period = _daily_period(d)

    n = 0
    with get_session() as session:
        for _, row in df.iterrows():
            # Column names per vnstock docs: 'name' (branch/type), 'buy_price', 'sell_price'.
            # Defensive .get()-style access in case the exact column set differs by version.
            name = str(row.get("name", row.iloc[0]))
            for col in df.columns:
                if "price" not in col.lower():
                    continue
                val = row[col]
                if val is None or str(val).strip() == "":
                    continue
                session.execute(UPSERT_MACRO, {
                    "category": "MKT_GOLD_SJC", "indicator_name": f"{name} - {col}",
                    "unit": "VND", "fiscal_year": fy, "period": period,
                    "value": float(val), "source_file": "vnstock_sjc",
                })
                n += 1
    return n


def fetch_forex() -> int:
    """USD/VND (and other currency) rates from Vietcombank -> macro_indicators
    category=MKT_FOREX_VCB."""
    from vnstock.explorer.misc import vcb_exchange_rate

    d = date.today()
    df = vcb_exchange_rate(date=d.isoformat())
    fy, period = _daily_period(d)

    n = 0
    with get_session() as session:
        for _, row in df.iterrows():
            currency = str(row.get("currency_code", row.iloc[0]))
            for col in df.columns:
                if "rate" not in col.lower() and "sell" not in col.lower() and "buy" not in col.lower():
                    continue
                val = row[col]
                if val is None or str(val).strip() == "":
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                session.execute(UPSERT_MACRO, {
                    "category": "MKT_FOREX_VCB", "indicator_name": f"{currency} - {col}",
                    "unit": "VND", "fiscal_year": fy, "period": period,
                    "value": val, "source_file": "vnstock_vcb",
                })
                n += 1
    return n


def fetch_oil(days: int) -> int:
    """WTI crude (FRED series DCOILWTICO) -> macro_indicators category=MKT_OIL_WTI.
    Skipped entirely if FRED_API_KEY isn't set - see .env.example."""
    if not FRED_API_KEY:
        print("  fetch_oil: SKIPPED - FRED_API_KEY not set in .env "
              "(register free at https://fred.stlouisfed.org/docs/api/api_key.html)")
        return 0

    import requests

    start = (date.today() - timedelta(days=days)).isoformat()
    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": "DCOILWTICO",
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start,
        },
        timeout=30,
    )
    resp.raise_for_status()
    obs = resp.json().get("observations", [])

    n = 0
    with get_session() as session:
        for o in obs:
            if o["value"] == ".":  # FRED's marker for no data that day (market holiday)
                continue
            d = date.fromisoformat(o["date"])
            fy, period = _daily_period(d)
            session.execute(UPSERT_MACRO, {
                "category": "MKT_OIL_WTI", "indicator_name": "WTI crude (Cushing, OK)",
                "unit": "USD/bbl", "fiscal_year": fy, "period": period,
                "value": float(o["value"]), "source_file": "fred_dcoilwtico",
            })
            n += 1
    return n


def fetch_sbv_rate() -> int:
    """SBV policy interest rate - NOT IMPLEMENTED.

    The real data lives at https://dttktt.sbv.gov.vn/webcenter/portal/vi/menu/rm/ls/bls
    (Oracle WebCenter portal) - a plain HTTP fetch of that URL returns an
    empty page, meaning the content is rendered client-side (JavaScript),
    not present in the initial HTML. Scraping it would need either
    rendering the page with a real browser (Selenium/Playwright) or finding
    the underlying JSON endpoint the page calls - neither confirmed yet.
    Left as an explicit stub (matches the "skip with a clear message, don't
    silently fake it" pattern used throughout this pipeline) rather than
    guessing at a scraping approach that might silently break or violate
    the portal's terms. See docs/PROGRESS.md for status."""
    print("  fetch_sbv_rate: SKIPPED - not implemented yet, see function docstring "
          "in scripts/market_snapshot.py")
    return 0


def fetch_ticker_prices(tickers: list, days: int, source: str = "KBS", sleep_s: float = 3.5) -> int:
    # Quote(), not Vnstock().stock(...) - see fetch_vnindex()'s docstring for
    # why (the deprecated wrapper segfaulted on VNINDEX; using the same
    # wrapper here for 216 individual stock tickers too risks the same crash).
    #
    # Real run (2026-07-18): vnstock enforces a hard 20 requests/minute limit
    # for unauthenticated ("Guest") usage - confirmed from its own error
    # message (source=KBS hit it too, not just VCI). At the old default of
    # --sleep 1.5 (40 req/min), it got throttled at ticker ~20 every time.
    # Default bumped to 3.5s (~17 req/min) to stay under that with margin.
    # WORSE: when the limit is hit, vnstock doesn't raise a normal Python
    # exception - it prints "Process terminated." and calls something that
    # behaves like sys.exit(), which `except Exception` does NOT catch
    # (SystemExit is a BaseException, not an Exception) - that's why the
    # whole script died outright on the first real --source KBS run instead
    # of skipping just that one ticker. Caught explicitly below now: sleep
    # the ~60s the limit needs to reset, then retry the same ticker instead
    # of losing the rest of the run. A free "Community" API key (60 req/min,
    # self-register at https://vnstocks.com/login - same "your own account,
    # not something I can create for you" situation as FRED_API_KEY) would
    # make this much faster if you want it later.
    from vnstock import Quote

    start = (date.today() - timedelta(days=days)).isoformat()
    end = date.today().isoformat()

    n_total = 0
    n_failed = 0
    i = 1
    while i <= len(tickers):
        ticker = tickers[i - 1]
        if i % 20 == 0 or i == 1:
            print(f"  ... {i}/{len(tickers)} tickers", flush=True)
        try:
            df = Quote(symbol=ticker, source=source).history(
                start=start, end=end, interval="1D"
            )
        except SystemExit:
            print(f"  {ticker}: rate limit hard-stop from vnstock - sleeping 65s and retrying this ticker")
            time.sleep(65)
            continue  # retry the SAME ticker (i not incremented)
        except Exception as e:
            n_failed += 1
            print(f"  {ticker}: SKIPPED - {e}")
            # Back off harder after a failure (likely a rate-limit/timeout,
            # not a one-off) instead of immediately hammering the next
            # ticker while the API is still refusing requests.
            time.sleep(max(sleep_s * 3, 5))
            i += 1
            continue

        if df is None or df.empty:
            i += 1
            continue

        with get_session() as session:
            for _, row in df.iterrows():
                d = row["time"].date() if hasattr(row["time"], "date") else date.fromisoformat(str(row["time"])[:10])
                session.execute(UPSERT_PRICE, {
                    "ticker": ticker, "trade_date": d,
                    "open_price": float(row["open"]), "high_price": float(row["high"]),
                    "low_price": float(row["low"]), "close_price": float(row["close"]),
                    "volume": int(row["volume"]), "source": f"vnstock_{source.lower()}",
                })
                n_total += 1
        i += 1
        time.sleep(sleep_s)  # be polite to the underlying API - 216 tickers back-to-back with no delay risks rate-limiting

    if n_failed:
        print(f"  {n_failed}/{len(tickers)} tickers failed - see SKIPPED lines above")
    return n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5,
                     help="How many calendar days of history to (re-)fetch per run. "
                          "Default 5 covers weekends/holidays for a daily incremental "
                          "refresh. Use a larger number (e.g. 365) for the first backfill run.")
    ap.add_argument("--tickers", type=str, default=None,
                     help="Comma-separated ticker subset, e.g. ACB,VNM (default: all companies in DB)")
    ap.add_argument("--skip-prices", action="store_true",
                     help="Skip per-ticker price loop (macro factors only - much faster, useful for testing)")
    ap.add_argument("--source", type=str, default="KBS", choices=["VCI", "KBS"],
                     help="vnstock data source, for VN-Index and per-ticker price. "
                          "Default KBS - confirmed on a real 216-ticker/365-day backfill "
                          "(2026-07-18): VCI's connections silently hang (30s timeouts, "
                          "no clean error) on a real fraction of requests, most likely "
                          "an anti-bot layer on VCI's side holding non-browser "
                          "connections instead of rejecting them; KBS completed the "
                          "same backfill with zero timeouts. --source VCI is available "
                          "if you want to try it (e.g. it may have more complete data "
                          "for some symbols) but expect timeouts.")
    ap.add_argument("--sleep", type=float, default=3.5,
                     help="Seconds to wait between per-ticker price requests (default 3.5, "
                          "~17 req/min - vnstock's unauthenticated limit is 20 req/min, "
                          "confirmed on a real run). Increase if still rate-limited.")
    args = ap.parse_args()

    summary = {}

    print("=== VN-Index ===")
    try:
        summary["vnindex"] = fetch_vnindex(args.days, source=args.source)
        print(f"  loaded {summary['vnindex']} rows")
    except SystemExit as e:
        print(f"  SKIPPED - rate limit hard-stop from vnstock ({e})")
        summary["vnindex"] = "ERROR: rate limited"
    except Exception as e:
        print(f"  SKIPPED - {e}")
        summary["vnindex"] = f"ERROR: {e}"

    print("=== Gold (SJC) ===")
    try:
        summary["gold"] = fetch_gold()
        print(f"  loaded {summary['gold']} rows")
    except Exception as e:
        print(f"  SKIPPED - {e}")
        summary["gold"] = f"ERROR: {e}"

    print("=== Forex (VCB) ===")
    try:
        summary["forex"] = fetch_forex()
        print(f"  loaded {summary['forex']} rows")
    except Exception as e:
        print(f"  SKIPPED - {e}")
        summary["forex"] = f"ERROR: {e}"

    print("=== Oil (WTI, FRED) ===")
    try:
        summary["oil"] = fetch_oil(args.days)
        if summary["oil"]:
            print(f"  loaded {summary['oil']} rows")
    except Exception as e:
        print(f"  SKIPPED - {e}")
        summary["oil"] = f"ERROR: {e}"

    print("=== SBV policy interest rate ===")
    summary["sbv_rate"] = fetch_sbv_rate()

    if not args.skip_prices:
        print("=== Per-company price ===")
        tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else get_tickers()
        print(f"  {len(tickers)} ticker(s)")
        try:
            summary["prices"] = fetch_ticker_prices(tickers, args.days, source=args.source, sleep_s=args.sleep)
            print(f"  loaded {summary['prices']} rows")
        except Exception as e:
            print(f"  SKIPPED - {e}")
            summary["prices"] = f"ERROR: {e}"
    else:
        summary["prices"] = "skipped (--skip-prices)"

    try:
        with get_session() as session:
            session.execute(INSERT_LOG, {
                "file_name": f"market_snapshot_{date.today().isoformat()}",
                "row_count": sum(v for v in summary.values() if isinstance(v, int)),
                "notes": str(summary),
            })
    except Exception as e:
        # Don't let a logging failure hide the real summary below - this is
        # exactly what happened the first time this ran before
        # 003_market_snapshot.sql's ingestion_log ALTER had been applied
        # yet (file_type enum didn't have 'market_snapshot', so this INSERT
        # itself threw and the summary below never printed).
        print(f"\n(ingestion_log write failed - {e})")
        print("(if this says 'Data truncated for column file_type', apply the "
              "ALTER TABLE ingestion_log ... command from the top of "
              "infra/mysql/migrations/003_market_snapshot.sql)")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
