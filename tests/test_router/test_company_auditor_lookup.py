from unittest.mock import patch

import pytest

from src.router.rules import classify_intent
from src.router.query_router import route


@pytest.mark.parametrize('question,ticker,periods', [
    ('Công ty kiểm toán của VNZ năm 2025 là ai?', 'VNZ', [(2025, 'YEAR')]),
    ('Công ty kiểm toán của VNM qua từng năm ?', 'VNM', []),
    ('Who is the auditor of VNZ in 2025?', 'VNZ', [(2025, 'YEAR')]),
])
def test_company_auditor_intent(question, ticker, periods):
    intent = classify_intent(question)
    assert intent.branch == 'auditor_lookup'
    assert intent.ticker == ticker
    assert intent.periods == periods


def test_metadata_answer_does_not_call_llm():
    rows = [{'fiscal_year': 2025, 'consolidation': 'consolidated', 'auditor': 'EY'}]
    with patch('src.structured.auditor_history.lookup_auditors_by_company', return_value=rows), patch('src.structured.auditor_history.load_document_evidence', return_value=([], [])), patch('src.router.query_router.rag_answer') as rag:
        result = route('Công ty kiểm toán của VNZ năm 2025 là ai?')
    assert result['branch'] == 'auditor_lookup'
    assert 'EY' in result['answer']
    assert 'disclosures.auditor' in result['answer']
    rag.assert_not_called()


def test_missing_metadata_reports_coverage_without_inventing_firm():
    with patch('src.structured.auditor_history.lookup_auditors_by_company', return_value=[]), patch('src.structured.auditor_history.load_document_evidence', return_value=([], [])):
        result = route('Công ty kiểm toán của VNM qua từng năm?')
    assert result['branch'] == 'auditor_lookup'
    assert 'Chưa có báo cáo năm' in result['answer']
