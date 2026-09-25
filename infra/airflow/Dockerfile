# Official Airflow image + the extra Python deps the DAGs' scripts need
# (psycopg2-binary, pandas, openpyxl, scikit-learn...) so BashOperator tasks can
# just `python scripts/foo.py` directly. If `docker compose build` fails to
# pull this exact tag, check https://hub.docker.com/r/apache/airflow/tags
# for a current 2.10.x-python3.11 tag and swap it in (both here and in the
# --constraint URL below - they must match).
FROM apache/airflow:2.10.5-python3.11

# poppler-utils (pdfinfo/pdftotext) - needed by scripts/pdf_ingest_router.py
# and src/retrieval/text_splitter.py, called via BashOperator in
# thesis_pdf_refresh (infra/airflow/dags/thesis_pdf_refresh_dag.py). Not a
# pip package - apt install needs root, then switch back to the airflow
# image's normal non-root user before installing Python packages below.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

COPY infra/airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt"
