"""One-time bootstrap DAG: schema check -> seed the 216-company sample ->
fix sector classification. Mirrors the manual sequence in README Steps 1-3
and 6 (docker compose up -d, init_db.py, seed_companies.py,
update_sectors.py).

schedule=None on purpose - this is NOT meant to run automatically. The
company list and sector map barely ever change, so re-running this on a
timer would just be wasted work (and update_sectors.py re-reads a CSV that
you'd have to manually refresh from Vietstock anyway). Trigger it by hand
in the Airflow UI (or `airflow dags trigger thesis_setup`) the first time
you stand this project up, or again later if you deliberately add more
companies to data/reference/ticker_sector_map.csv.

For the recurring part (new BCTC filings, new macro releases), see
thesis_refresh_dag.py instead.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="thesis_setup",
    description="One-time bootstrap: schema check, seed companies, sector classification",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["thesis", "setup"],
) as dag:

    init_db = BashOperator(
        task_id="init_db",
        bash_command=f"cd {PROJECT_DIR} && python scripts/init_db.py",
    )

    seed_companies = BashOperator(
        task_id="seed_companies",
        bash_command=f"cd {PROJECT_DIR} && python scripts/seed_companies.py",
    )

    update_sectors = BashOperator(
        task_id="update_sectors",
        bash_command=f"cd {PROJECT_DIR} && python scripts/update_sectors.py",
    )

    init_db >> seed_companies >> update_sectors
