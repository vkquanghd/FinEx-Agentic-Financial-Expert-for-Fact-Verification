"""Recurring market-factor refresh DAG: VN-Index, gold, USD/VND forex, oil
(if FRED_API_KEY is set), and per-company daily price/volume for every
ticker in `companies` - via scripts/market_snapshot.py (vnstock + FRED,
no login/crawling of any walled site - see that script's docstring).

Schedule: weekdays 10:00 UTC = ~17:00 Vietnam time (UTC+7), safely after
HOSE/HNX's 15:00 close. Airflow's own default timezone is UTC unless you've
set AIRFLOW__CORE__DEFAULT_TIMEZONE in docker-compose.yml - adjust the cron
below if you change that.

Unlike thesis_pdf_refresh (schedule=None, PDFs arrive irregularly), this
DAG runs on a real schedule because the whole point of "market factor"
data is that it changes daily - see docs/PROGRESS.md for why this was
built as a separate DAG from thesis_refresh (BCTC/macro exports still
need a manual Vietstock login+export step; this data source doesn't).

`--days 5` on every scheduled run (not just 1) deliberately overlaps
slightly - covers weekends/holidays/a missed run without gaps, and the
UPSERT in market_snapshot.py makes re-writing the same day's value safe.

`--source KBS` - confirmed on a real backfill run (2026-07-18): the
default VCI source silently hangs (30s read timeouts, no clean error)
on a real percentage of requests, most likely VCI's anti-bot layer
holding non-browser connections rather than rejecting them outright.
KBS hits a completely different backend and completed a 216-ticker,
365-day backfill (39,573 rows, 207/216 tickers - the other 9 are
invalid/delisted symbols, not a connection problem) with zero timeouts.
See docs/PROGRESS.md for the full debugging story.

`company_price_snapshot` is already created by the consolidated Postgres
schema (infra/postgres/migrations/001_init_schema.sql, applied automatically
on first `docker compose up`) - no separate manual apply step before this
DAG's first run anymore (that was only needed on the pre-2026-07-26 MySQL
setup, see docs/PROGRESS.md).

Trigger manually any time with: airflow dags trigger thesis_market_snapshot
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="thesis_market_snapshot",
    description="Daily: VN-Index, gold, forex, oil, per-company price (vnstock + FRED)",
    schedule="0 10 * * 1-5",  # weekdays 10:00 UTC (~17:00 VN time)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["thesis", "refresh", "market"],
) as dag:

    market_snapshot = BashOperator(
        task_id="market_snapshot",
        bash_command=f"cd {PROJECT_DIR} && python scripts/market_snapshot.py --days 5 --source KBS",
    )
