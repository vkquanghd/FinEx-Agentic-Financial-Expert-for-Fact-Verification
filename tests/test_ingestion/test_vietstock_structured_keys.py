from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from src.ingestion import vietstock_structured as module


@pytest.mark.parametrize("keys", [["Beta"], [], ["Beta", "BETA"]])
def test_persist_resolves_unique_existing_company_key(monkeypatch, keys):
    session = MagicMock()
    session.execute.return_value.scalars.return_value.all.return_value = keys

    @contextmanager
    def fake_session():
        yield session

    monkeypatch.setattr(module, "get_session", fake_session)
    table = module.StructuredTable(
        "BETA", "trading", "trading", "https://finance.vietstock.vn/BETA/thong-ke-giao-dich.htm",
        0, "Prices", ["Date", "Price"], [["2026-09-18", "10"]], "a" * 64,
    )
    if len(keys) != 1:
        with pytest.raises(ValueError, match="Expected one company key"):
            module._persist_tables([table], 200)
        assert session.execute.call_count == 1
    else:
        assert module._persist_tables([table], 200) == 1
        assert session.execute.call_args.args[1]["ticker"] == "Beta"
