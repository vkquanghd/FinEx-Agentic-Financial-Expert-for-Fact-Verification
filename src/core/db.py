"""PostgreSQL connection layer (SQLAlchemy). Single place every other module
imports from to talk to the database - keeps DB access swappable later.

Migrated 2026-07-26 from MySQL to PostgreSQL (see
infra/postgres/migrations/001_init_schema.sql and docs/PROGRESS.md) - this
is now the SAME database that also holds disclosure_chunks (the pgvector
table), not a separate service. Only src/core/config.py's DATABASE_URL
dialect prefix changed (mysql+pymysql -> postgresql+psycopg2); everything
below is unchanged, since SQLAlchemy's Session/engine API is dialect-agnostic."""
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.core.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_session():
    """Usage: with get_session() as session: session.execute(...)"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Quick sanity check that the app can reach Postgres. Returns True/False
    and prints the result - run this first after `docker compose up`."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"OK - connected to {DATABASE_URL.split('@')[-1]}")
        return True
    except Exception as e:
        print(f"FAILED to connect: {e}")
        return False
