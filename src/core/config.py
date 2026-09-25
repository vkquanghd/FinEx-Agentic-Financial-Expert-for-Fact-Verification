"""App configuration: loads .env and exposes settings used across the project."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]  # project root

# --- Database (single PostgreSQL + pgvector instance, Docker service
# "postgres" in docker-compose.yml) ---
# Migrated 2026-07-26 off the old two-service setup (MySQL for structured
# tables + a separate "vectorstore" pgvector service) onto ONE Postgres
# database holding everything - companies/financial_line_items/.../
# disclosures/ingestion_log/macro_indicators/company_price_snapshot AND
# disclosure_chunks (the vector table). See docs/PROGRESS.md ("Gộp MySQL +
# pgvector thành một PostgreSQL duy nhất") for the full rationale and
# infra/postgres/migrations/001_init_schema.sql for the consolidated schema.
# Default port 5432 (was 3306 for MySQL / 5433 for the old separate
# vectorstore service, both now retired).
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "thesis_fin_qa")
DB_USER = os.getenv("DB_USER", "thesis_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Backward-compatible aliases -----------------------------------------
# src/retrieval/vector_store.py opens its OWN psycopg2 connection (separate
# from the SQLAlchemy engine in src/core/db.py) because it needs raw
# %s-parameterized cursor.execute() for the pgvector `<=>` operator and
# hand-built vector literals - it is not doing anything MySQL-specific, it
# just predates this migration. Since there is now only one database, these
# simply point at the same DB_* values above rather than a second set of
# PGVECTOR_* env vars. Kept as separate names (not a straight rename) only so
# vector_store.py's existing `config.PGVECTOR_HOST` etc. references did not
# all need touching in the same change as this file.
PGVECTOR_HOST = DB_HOST
PGVECTOR_PORT = DB_PORT
PGVECTOR_DB = DB_NAME
PGVECTOR_USER = DB_USER
PGVECTOR_PASSWORD = DB_PASSWORD

# --- LLM (src/models/, src/retrieval/rag_pipeline.py) ---
# Served locally via Ollama (https://ollama.com) - install once
# (`brew install ollama`), then `ollama pull qwen3:8b` (or the lighter
# `qwen3:4b` if 8b is too slow/heavy on your machine - see README).
# No API key needed, nothing leaves your machine. OLLAMA_HOST defaults to
# Ollama's own default port; only override if you're running it elsewhere
# (e.g. inside Docker later).
def _is_container_runtime() -> bool:
    """Whether this Python process is running inside a Linux container.

    `host.docker.internal` is meaningful from Docker Desktop containers but
    is the wrong endpoint for a macOS process. Keeping this distinction here
    prevents a shell-exported Docker value from breaking a locally started
    QA service.
    """
    return Path("/.dockerenv").exists() or os.getenv("container") is not None


_configured_ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not _is_container_runtime() and _configured_ollama_host.startswith("http://host.docker.internal"):
    # Docker-specific endpoint leaked into a host-run process. Preserve the
    # port suffix in case a non-default local Ollama port is configured.
    _suffix = _configured_ollama_host.removeprefix("http://host.docker.internal")
    OLLAMA_HOST = f"http://127.0.0.1{_suffix}"
else:
    OLLAMA_HOST = _configured_ollama_host
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# --- Market snapshot (scripts/market_snapshot.py, thesis_market_snapshot DAG) ---
# Free API key you register for yourself at
# https://fred.stlouisfed.org/docs/api/api_key.html - this is a data-access
# key (like a weather API key), not a login/trading credential, but it's
# still your own registration to make, not something this project can do
# for you. Leave unset to skip oil-price ingestion (the script prints a
# clear message and continues with everything else rather than failing).
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OCR_CACHE_DIR = BASE_DIR / "data" / "ocr_cache"
