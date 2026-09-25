"""End-to-end verification of everything built so far - run this after any
DB/schema change, or whenever you want to confirm the whole pipeline still
works before moving on to the next thesis phase (Router / RAG / models).

Does NOT hardcode specific tickers/periods - it queries the live DB for
real, currently-loaded data to test against, so it stays valid regardless
of exactly which companies you've loaded.

Usage: python scripts/verify_all.py
Exit code 0 = everything passed. Non-zero = see the FAILED lines above the summary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.causal.panel_builder import build_panel
from src.causal.treatment_effect import estimate_ate
from src.core.db import check_connection, engine, get_session
from src.structured.calculator import Period, decompose_gross_margin_change
from src.structured.dag_pipeline import answer

results = []  # (section, ok, detail)


def check(section, ok, detail=""):
    results.append((section, ok, detail))
    status = "OK" if ok else "FAILED"
    print(f"[{status}] {section}{(' - ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
print("=== 1. DB connection + schema ===")
ok = check_connection()
check("DB connection", ok)
if not ok:
    print("\nCannot continue without a DB connection. Check .env / docker compose up.")
    sys.exit(1)

REQUIRED_TABLES = {
    "companies", "financial_line_items", "financial_ratios",
    "disclosures", "ingestion_log", "macro_indicators",
    "company_price_snapshot", "disclosure_chunks",
}
with engine.connect() as conn:
    # "SHOW TABLES" was MySQL-only syntax, removed 2026-07-26 along with
    # the rest of the MySQL->Postgres migration (see scripts/init_db.py for
    # the same fix) - information_schema.tables is the portable equivalent.
    existing = {
        row[0] for row in conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        ))
    }
missing = REQUIRED_TABLES - existing
check("All 8 tables present", not missing, f"missing: {missing}" if missing else "")

# ---------------------------------------------------------------------------
print("\n=== 2. Row counts ===")
with get_session() as session:
    counts = {}
    for t in sorted(REQUIRED_TABLES & existing):
        counts[t] = session.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
    for t, c in counts.items():
        print(f"  {t}: {c}")

check("companies seeded (expect 216)", counts.get("companies", 0) >= 200,
      f"found {counts.get('companies', 0)}")
check("financial_line_items has data", counts.get("financial_line_items", 0) > 0,
      f"found {counts.get('financial_line_items', 0)}")
check("macro_indicators has data", counts.get("macro_indicators", 0) > 0,
      f"found {counts.get('macro_indicators', 0)} (run: python -m src.ingestion.macro_etl ...)")

# ---------------------------------------------------------------------------
print("\n=== 3. Sector classification (Step 6 fix) ===")
with get_session() as session:
    n_unclassified = session.execute(
        text("SELECT COUNT(*) FROM companies WHERE sector = 'Chưa phân loại'")
    ).scalar()
check("Sector classification filled in", n_unclassified == 0,
      f"{n_unclassified} companies still 'Chưa phân loại' - run: python scripts/update_sectors.py")

# ---------------------------------------------------------------------------
print("\n=== 4. DAG lookup (VI + EN) ===")
with get_session() as session:
    sample = session.execute(text(
        "SELECT ticker, fiscal_year, period FROM financial_line_items "
        "WHERE statement_type='balance_sheet' AND ms_code='280' LIMIT 1"
    )).first()

if sample:
    ticker, year, period = sample
    period_vi = {"YEAR": f"năm {year}", "Q1": f"quý 1 năm {year}", "Q2": f"quý 2 năm {year}",
                 "Q3": f"quý 3 năm {year}", "Q4": f"quý 4 năm {year}"}.get(period, f"{period}/{year}")
    q_vi = f"Tổng tài sản của {ticker} {period_vi} là bao nhiêu?"
    q_en = f"What is the total assets of {ticker} in {year}?" if period == "YEAR" else \
           f"What is the total assets of {ticker} in {period} {year}?"
    a_vi = answer(q_vi)
    a_en = answer(q_en)
    check("DAG lookup (Vietnamese)", not a_vi.startswith("Không trả lời được"), f"Q: {q_vi} -> {a_vi}")
    check("DAG lookup (English)", not a_en.startswith("Không trả lời được"), f"Q: {q_en} -> {a_en}")
else:
    check("DAG lookup (VI + EN)", False, "no financial_line_items rows to test against - load some BCTC data first")

# ---------------------------------------------------------------------------
print("\n=== 5. What-if calculator ===")
with get_session() as session:
    # SELECT only the GROUP BY column - was originally written to satisfy
    # MySQL 8's default sql_mode (only_full_group_by), which rejects
    # selecting fiscal_year here since it's not aggregated or functionally
    # dependent on ticker. Kept unchanged after the 2026-07-26 Postgres
    # migration: Postgres enforces this same rule UNCONDITIONALLY (there is
    # no sql_mode-style setting to loosen it), so this query needed no
    # changes to keep working under the new engine.
    two_periods = session.execute(text(
        "SELECT ticker FROM financial_line_items "
        "WHERE statement_type='income_statement' AND ms_code='10' AND period='YEAR' "
        "GROUP BY ticker HAVING COUNT(DISTINCT fiscal_year) >= 2 LIMIT 1"
    )).first()

if two_periods:
    ticker = two_periods[0]
    with get_session() as session:
        years = session.execute(text(
            "SELECT DISTINCT fiscal_year FROM financial_line_items "
            "WHERE ticker=:t AND statement_type='income_statement' AND ms_code='10' AND period='YEAR' "
            "ORDER BY fiscal_year"
        ), {"t": ticker}).fetchall()
    y1, y2 = years[0][0], years[-1][0]
    res = decompose_gross_margin_change(ticker, Period(y1, "YEAR"), Period(y2, "YEAR"))
    check("Bridge decomposition (calculator.py)", res.get("ok", False) and res.get("check_sums_to_delta_gp", False),
          f"{ticker} {y1}->{y2}: {res}")
else:
    check("Bridge decomposition (calculator.py)", False,
          "no ticker with >=2 years of revenue data - load more periods first")

# ---------------------------------------------------------------------------
print("\n=== 6. Cross-sectional causal panel ===")
panel = build_panel()
check("Panel builds without error", panel is not None)
if panel is not None and not panel.empty:
    print(f"  panel shape: {panel.shape[0]} rows x {panel.shape[1]} cols")
    res = estimate_ate(panel=panel)
    check("estimate_ate runs and returns a labeled result", res.get("ok", False),
          res.get("confidence", res.get("error", "")))
else:
    check("Panel has rows", False, "empty panel - need >=1 company with period='YEAR' data")

print("\n=== SUMMARY ===")
n_ok = sum(1 for _, ok, _ in results if ok)
n_total = len(results)
for section, ok, detail in results:
    if not ok:
        print(f"  NEEDS ATTENTION: {section} - {detail}")
print(f"\n{n_ok}/{n_total} checks passed.")
sys.exit(0 if n_ok == n_total else 1)
