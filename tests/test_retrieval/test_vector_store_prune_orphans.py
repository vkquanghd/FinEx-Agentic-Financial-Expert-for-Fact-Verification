"""Regression test for prune_orphaned_chunks() in src/retrieval/vector_store.py
(added 2026-07-29): a disclosure_chunks row whose (ticker, source_file, page,
chunk_index) no longer appears in the current chunks.jsonl (e.g. a PDF page
was removed or renamed) must be deleted, not left behind forever.

No real Postgres available in this sandbox (see vector_store.py's own module
docstring caveat), so get_connection() is monkeypatched with a tiny in-memory
fake that mimics just enough of the psycopg2 cursor/connection interface
(execute/executemany/fetchall/commit/close, used as a context manager) for
prune_orphaned_chunks()'s actual SQL calls.
"""
import pytest

import src.retrieval.vector_store as vector_store


class _FakeCursor:
    def __init__(self, table):
        self.table = table
        self._last_select = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, sql, params=None):
        stripped = sql.strip()
        if stripped.startswith("SELECT"):
            self._last_select = list(self.table)
        elif stripped.startswith("DELETE"):
            self.table.discard(params)

    def executemany(self, sql, batch):
        for params in batch:
            self.table.discard(tuple(params))

    def fetchall(self):
        return self._last_select


class _FakeConn:
    def __init__(self, table):
        self.table = table
        self.n_commits = 0

    def cursor(self):
        return _FakeCursor(self.table)

    def commit(self):
        self.n_commits += 1

    def close(self):
        pass


@pytest.fixture
def fake_db(monkeypatch):
    table = {
        ("AAA", "f1", 1, 0),
        ("AAA", "f1", 1, 1),
        ("BBB", "f2", 1, 0),
        ("CCC", "f3", 1, 0),  # will be "removed" -- orphan
        ("DDD", "f4", 1, 0),  # will be "removed" -- orphan
    }
    conn = _FakeConn(table)
    monkeypatch.setattr(vector_store, "get_connection", lambda: conn)
    return conn


CURRENT_KEYS = {
    ("AAA", "f1", 1, 0),
    ("AAA", "f1", 1, 1),
    ("BBB", "f2", 1, 0),
}


def test_dry_run_reports_count_without_deleting(fake_db):
    n = vector_store.prune_orphaned_chunks(CURRENT_KEYS, dry_run=True)
    assert n == 2
    assert len(fake_db.table) == 5  # untouched


def test_prune_deletes_only_orphaned_rows(fake_db):
    # This fixture's orphan ratio is 2/5 = 40%, above the default 10% safety
    # guard (see the max-delete-ratio tests below) - force it here since
    # this test is specifically about the delete-selection logic, not the
    # guard.
    n = vector_store.prune_orphaned_chunks(CURRENT_KEYS, dry_run=False, force_over_ratio=True)
    assert n == 2
    assert fake_db.table == CURRENT_KEYS


def test_prune_is_idempotent(fake_db):
    vector_store.prune_orphaned_chunks(CURRENT_KEYS, dry_run=False, force_over_ratio=True)
    n_second_pass = vector_store.prune_orphaned_chunks(CURRENT_KEYS, dry_run=False, force_over_ratio=True)
    assert n_second_pass == 0


def test_no_orphans_deletes_nothing(fake_db):
    all_keys = set(fake_db.table)
    n = vector_store.prune_orphaned_chunks(all_keys, dry_run=False)
    assert n == 0
    assert len(fake_db.table) == 5


# ---- max_delete_ratio safety guard (added 2026-07-29 after review pointed
# out that a wrong/partial --input to --prune-orphans could otherwise
# silently delete every other ticker's chunks) ----

def test_max_delete_ratio_refuses_large_deletion_by_default(fake_db):
    """Fixture's orphan ratio is 2/5 = 40%, above the 10% default guard."""
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        vector_store.prune_orphaned_chunks(CURRENT_KEYS, dry_run=False)
    assert len(fake_db.table) == 5  # refused -- must be untouched


def test_max_delete_ratio_allows_small_deletion(monkeypatch):
    table = {("A", "f", 1, 0), ("B", "f", 1, 0), ("C", "f", 1, 0)}  # 1/3 orphan is still > 10%...
    # use a fixture where the orphan ratio is genuinely small (<=10%)
    table = {("T%d" % i, "f", 1, 0) for i in range(20)}
    current = {("T%d" % i, "f", 1, 0) for i in range(19)}  # 1/20 = 5% orphaned
    conn = _FakeConn(set(table))
    monkeypatch.setattr(vector_store, "get_connection", lambda: conn)
    n = vector_store.prune_orphaned_chunks(current, dry_run=False)
    assert n == 1
    assert conn.table == current


def test_prune_force_ratio_overrides_the_guard(fake_db):
    n = vector_store.prune_orphaned_chunks(CURRENT_KEYS, dry_run=False, force_over_ratio=True)
    assert n == 2
    assert fake_db.table == CURRENT_KEYS
