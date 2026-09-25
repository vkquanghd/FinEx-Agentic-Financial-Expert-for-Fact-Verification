"""Recurring refresh DAG: pick up any new BCTC / macro Excel exports that
have been dropped into data/raw/vietstock_exports/ (and its macro/
subfolder) since the last run, load them, then re-run verify_all.py as a
smoke test so a broken run shows up as a failed task in the Airflow UI
instead of silently leaving stale/partial data behind.

Default schedule is weekly (Monday 06:00, Airflow's own timezone - set
AIRFLOW__CORE__DEFAULT_TIMEZONE in docker-compose.yml if you want it to
mean Vietnam time specifically). Vietstock doesn't push data to you, so
this only helps once you've manually re-exported fresh files from Vietstock
into those folders before the scheduled run - it does not crawl Vietstock
itself. Change `schedule` below to match how often you actually plan to
re-export (daily "@daily", monthly "@monthly", or a cron string).

ingest_bctc and ingest_macro run in parallel (independent data, independent
tables) - verify only runs after both finish, successfully or not, so you
always get a fresh verify_all.py report to look at.

Trigger manually any time with: airflow dags trigger thesis_refresh
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="thesis_refresh",
    description="Recurring: ingest new BCTC + macro exports, then verify",
    schedule="0 6 * * 1",  # every Monday 06:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["thesis", "refresh"],
) as dag:

    ingest_bctc = BashOperator(
        task_id="ingest_bctc",
        bash_command=f"cd {PROJECT_DIR} && python scripts/ingest_bctc_folder.py",
    )

    ingest_macro = BashOperator(
        task_id="ingest_macro",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f'python -m src.ingestion.macro_etl "data/raw/vietstock_exports/macro/*.xlsx"'
        ),
    )

    verify = BashOperator(
        task_id="verify_all",
        bash_command=f"cd {PROJECT_DIR} && python scripts/verify_all.py",
        trigger_rule="all_done",  # run the report even if one ingest task failed
    )

    [ingest_bctc, ingest_macro] >> verify
