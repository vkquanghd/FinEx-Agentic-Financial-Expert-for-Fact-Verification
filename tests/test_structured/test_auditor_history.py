from unittest.mock import patch
import pytest
from src.structured.auditor_history import answer_auditor_history


@pytest.mark.parametrize('ticker', ['VNM', 'VNZ', 'VIC', 'HPG'])
def test_each_year_separate_and_sources_numbered(ticker):
    docs = [{'fiscal_year': y, 'chunks': 1} for y in [2024, 2025]]
    chunks = [{'year': y, 'source_file': f'{ticker}_{y}', 'page': 3} for y in [2024, 2025]]
    def generate(question, evidence):
        year = evidence[0]['year']
        return {'answer': f'{year} firm [Source 1]', 'sources': [{'source_file': f'{ticker}_{year}', 'page': 3}]}
    with patch('src.structured.auditor_history.lookup_auditors_by_company', return_value=[]), patch('src.structured.auditor_history.load_document_evidence', return_value=(docs, chunks)), patch('src.structured.auditor_history.answer_with_citation', side_effect=generate) as gen:
        result = answer_auditor_history('Who is the auditor each year?', ticker, [])
    assert gen.call_count == 2
    assert '2025 firm [Source 1]' in result['answer']
    assert '2024 firm [Source 2]' in result['answer']
    assert len(result['sources']) == 2


def test_empty_unprocessed_and_unconfirmed_years_are_distinct():
    docs = [{'fiscal_year': 2024, 'chunks': 0}, {'fiscal_year': 2025, 'chunks': 3}]
    with patch('src.structured.auditor_history.lookup_auditors_by_company', return_value=[]), patch('src.structured.auditor_history.load_document_evidence', return_value=(docs, [])), patch('src.structured.auditor_history.answer_with_citation') as gen:
        result = answer_auditor_history('auditor?', 'VNM', [2023, 2024, 2025])
    assert {r['year']: r['status'] for r in result['coverage']} == {2023:'no_registered_report', 2024:'not_chunked', 2025:'not_confirmed'}
    gen.assert_not_called()
