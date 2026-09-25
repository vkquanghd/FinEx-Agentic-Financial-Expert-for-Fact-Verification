"""Recurring PDF refresh DAG: pick up any new PDF disclosures dropped into
data/raw/pdf_disclosures/ since the last run, route them (text-native vs
needs-OCR), chunk the text-native pages, embed with BAAI/bge-m3, and load
into pgvector.

`schedule=None` deliberately, unlike thesis_refresh - PDF disclosures don't
arrive on a predictable weekly cadence the way BCTC/macro exports do (see
docs/PROGRESS.md), so this is trigger-only: either click "Trigger DAG" in
the Airflow UI (http://localhost:8080) after dropping new PDFs into
data/raw/pdf_disclosures/, or run
`docker exec thesis_airflow airflow dags trigger thesis_pdf_refresh`.
Change `schedule` to a cron string if your PDF exports do end up landing on
a predictable cadence later.

Airflow does NOT log into Vietstock or fetch PDFs for you - same manual
export step as everything else in this pipeline, this DAG only automates
what happens AFTER the files are already in the folder.

route_pdfs -> chunk_text -> embed_and_load, strictly sequential (each step
consumes the previous step's output file), unlike thesis_refresh's
parallel ingest_bctc/ingest_macro. No trailing verify step here - the row
count check (SELECT COUNT(*) FROM disclosure_chunks) is a one-line manual
check documented in README.md Step 10, not worth a 4th task for now.

One-time prerequisite before this DAG can succeed: the pgvector schema
must already exist (infra/pgvector/migrations/001_disclosure_chunks.sql
applied by hand - see the comment at the top of that file).
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="thesis_pdf_refresh",
    description="Trigger-only: route + chunk + embed new PDF disclosures into pgvector",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["thesis", "refresh", "rag"],
) as dag:

    route_pdfs = BashOperator(
        task_id="route_pdfs",
        bash_command=(
            f"cd {PROJECT_DIR} && python3 scripts/pdf_ingest_router.py "
            f"--input data/raw/pdf_disclosures --output data/processed"
        ),
    )

    chunk_text = BashOperator(
        task_id="chunk_text",
        bash_command=(
            f"cd {PROJECT_DIR} && python3 -m src.retrieval.text_splitter "
            f"--input data/processed/text_native --output data/processed/chunks.jsonl"
        ),
    )

    embed_and_load = BashOperator(
        task_id="embed_and_load",
        bash_command=(
            f"cd {PROJECT_DIR} && python3 -m src.retrieval.vector_store "
            f"--input data/processed/chunks.jsonl"
        ),
    )

    route_pdfs >> chunk_text >> embed_and_load
