"""Trigger-only DAG: scrape web/press content for a ticker+topic, chunk it,
and embed+load into the SEPARATE `web_chunks` pgvector table.

Built 2026-09-07 alongside scripts/web_news_ingest.py and
infra/postgres/migrations/014_web_chunks.sql - see that migration's header
comment for the full rationale (VNM's own official risk-management
disclosure is genuinely thin, checked against the real PDF, so a question
like "Rủi ro pháp lý của VNM là gì?" needs press coverage as a SEPARATE,
clearly-labeled evidence tier, not more disclosure_chunks rows).

`schedule=None`, same as thesis_pdf_refresh - news doesn't arrive on a
predictable cadence any more than PDF disclosures do, and unlike
thesis_refresh's BCTC/macro exports, there's no fixed re-export cycle to
hang a cron schedule off of. Either click "Trigger DAG w/ config" in the
Airflow UI (http://localhost:8080) and fill in the `ticker`/`query` params,
or from the command line:

    docker exec thesis_airflow airflow dags trigger thesis_web_refresh \\
        --conf '{"ticker": "VNM", "query": "rủi ro pháp lý tranh chấp xử phạt"}'

scrape -> chunk -> embed_and_load, strictly sequential (each step consumes
the previous step's output), same shape as thesis_pdf_refresh_dag.py.

One-time prerequisite before this DAG can succeed: the web_chunks table
must already exist (infra/postgres/migrations/014_web_chunks.sql applied by
hand - see the comment at the top of that file for the exact `docker exec`
command).

Requires the optional `ddgs` dependency (see scripts/web_fallback_lookup.py's
docstring) to be installed in the `qa`/`airflow` image - not in
requirements.txt's required set, same tradeoff already made for
causalml/econml there.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models.param import Param

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="thesis_web_refresh",
    description="Trigger-only: scrape + chunk + embed press/web content into web_chunks (fallback-only, NOT official disclosures)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["thesis", "refresh", "rag", "web-fallback"],
    params={
        "ticker": Param("", type="string", description="Mã cổ phiếu, ví dụ VNM (để trống nếu tìm chủ đề chung, không gắn ticker)"),
        "query": Param("rủi ro pháp lý tranh chấp xử phạt", type="string", description="Từ khoá tìm kiếm"),
        "max_results": Param(8, type="integer", description="Số kết quả tối đa mỗi lượt tìm kiếm"),
    },
) as dag:

    scrape = BashOperator(
        task_id="scrape",
        bash_command=(
            f"cd {PROJECT_DIR} && python3 scripts/web_news_ingest.py scrape "
            f'--ticker "{{{{ params.ticker }}}}" --query "{{{{ params.query }}}}" '
            f'--max-results {{{{ params.max_results }}}}'
        ),
    )

    chunk = BashOperator(
        task_id="chunk",
        bash_command=(
            f"cd {PROJECT_DIR} && python3 scripts/web_news_ingest.py chunk "
            f"--raw-dir data/raw/web_articles --output data/processed/web_chunks.jsonl"
        ),
    )

    embed_and_load = BashOperator(
        task_id="embed_and_load",
        bash_command=(
            f"cd {PROJECT_DIR} && python3 -m src.retrieval.vector_store "
            f"--web --input data/processed/web_chunks.jsonl"
        ),
    )

    scrape >> chunk >> embed_and_load
