import pytest

from src.structured.ratio_schema import resolve_ratio
from src.router.rules import classify_intent


@pytest.mark.parametrize("ticker", ["VIC", "VNZ"])
@pytest.mark.parametrize("verb", ["happen", "happened"])
def test_general_company_question_is_not_a_ratio(ticker, verb):
    question = f"What {verb} for the {ticker} ticket in 2025?"
    assert resolve_ratio(question) is None
    assert classify_intent(question).branch == "rag"


@pytest.mark.parametrize("alias", ["P/E", "PE", "price to earnings"])
def test_explicit_pe_remains_supported(alias):
    question = f"What is {alias} for VIC in 2025?"
    resolved = resolve_ratio(question)
    assert resolved is not None
    assert "(P/E)" in resolved[1]
    assert classify_intent(question).branch == "ratio_lookup"


@pytest.mark.parametrize("question", [
    "What happened to VNZ in 2025?",
    "Describe VIC performance in 2025",
    "Chuyen gi da xay ra voi VIC nam 2025?",
])
def test_unrelated_words_do_not_imply_a_ratio(question):
    assert resolve_ratio(question) is None
