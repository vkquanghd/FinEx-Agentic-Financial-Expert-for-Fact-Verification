"""One-off script: verify DB connection and confirm the schema (all 8
tables from infra/postgres/migrations/001_init_schema.sql - migrated
2026-07-26 off the old MySQL + separate-pgvector setup onto a single
consolidated Postgres schema, auto-applied by Docker on first startup, see
docs/PROGRESS.md) is actually present. Run this after `docker compose up -d`.

Usage: python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from src.core.db import check_connection, engine

REQUIRED_TABLES = {
    "companies", "financial_line_items", "financial_ratios",
    "disclosures", "ingestion_log", "macro_indicators",
    "company_price_snapshot", "disclosure_chunks",
}

if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)

    with engine.connect() as conn:
        # Postgres has no "SHOW TABLES" (that's MySQL-only syntax, removed
        # here 2026-07-26 along with the rest of the MySQL->Postgres
        # migration) - information_schema.tables is the portable way to
        # list tables, scoped to the "public" schema (where
        # docker-entrypoint-initdb.d's CREATE TABLE statements land by
        # default, since no other schema is created/selected anywhere in
        # this project).
        existing = {
            row[0] for row in conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ))
        }

    missing = REQUIRED_TABLES - existing
    if missing:
        print(f"Missing tables: {missing}")
        print("Migration did not run. If the postgres_data volume already existed")
        print("before, docker-entrypoint-initdb.d is skipped. Fix: ")
        print("  docker compose down -v && docker compose up -d")
        sys.exit(1)

    print(f"All tables present: {sorted(existing)}")
