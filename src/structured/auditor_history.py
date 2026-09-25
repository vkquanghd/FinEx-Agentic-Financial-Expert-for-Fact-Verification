"""Year-scoped auditor evidence from the active local PDF corpus."""
import re
from sqlalchemy import text

from src.core.db import get_session
from src.core.lang_detect import detect_language
from src.models.generation import answer_with_citation
from src.structured.auditor_lookup import lookup_auditors_by_company


def load_document_evidence(ticker):
    # Inventory includes unchunked reports; absence of searchable text is not
    # absence of a document. No embedding is required for this lexical lookup.
    with get_session() as session:
        inventory = session.execute(text("""
            SELECT p.id, p.fiscal_year, p.source_file, count(c.id) AS chunks
            FROM pdf_documents p LEFT JOIN disclosure_chunks c ON c.pdf_document_id=p.id
            WHERE upper(p.ticker)=:ticker AND p.is_active
              AND (p.doctype IN ('financial_statement','annual_report')
                   OR p.source_file ~* 'baocaotaichinh|baocaothuongnien|bctc|bctn|financial|annual|consol|signed')
            GROUP BY p.id, p.fiscal_year, p.source_file
        """), {'ticker': ticker.upper()}).mappings().all()
        candidates = session.execute(text("""
            WITH ranked AS (
                SELECT c.*, p.fiscal_year AS year,
                       row_number() OVER (PARTITION BY c.pdf_document_id ORDER BY
                           CASE WHEN c.chunk_text ~* 'báo cáo kiểm toán độc lập|independent auditor.s report' THEN 0 ELSE 1 END,
                           c.page,c.chunk_index) AS rank
                FROM disclosure_chunks c JOIN pdf_documents p ON p.id=c.pdf_document_id
                WHERE upper(p.ticker)=:ticker AND p.is_active
                  AND (p.doctype IN ('financial_statement','annual_report')
                       OR p.source_file ~* 'baocaotaichinh|baocaothuongnien|bctc|bctn|financial|annual|consol|signed')
                  AND c.chunk_text ~* 'kiểm toán|auditor|audited by|kpmg|ernst|deloitte|pricewaterhouse|pwc'
            )
            SELECT ticker,doctype,source_file,page,chunk_text,year,rank
            FROM ranked WHERE rank <= 4 ORDER BY year DESC NULLS LAST,rank,source_file,page
        """), {'ticker': ticker.upper()}).mappings().all()
    return [dict(r) for r in inventory], [dict(r) for r in candidates]


def answer_auditor_history(question, ticker, years):
    english = detect_language(question) == 'en'
    metadata = lookup_auditors_by_company(ticker, years)
    inventory, candidates = load_document_evidence(ticker)
    available = {r['fiscal_year'] for r in inventory if r['fiscal_year'] is not None}
    available.update(r['fiscal_year'] for r in metadata)
    targets = sorted(set(years) if years else available, reverse=True)
    lines, sources, coverage = [], [], []
    for year in targets:
        lines.append(f"### {year}")
        records = [r for r in metadata if r['fiscal_year'] == year]
        if records:
            lines.extend(f"- {r['consolidation']}: {r['auditor']} (disclosures.auditor; metadata)" for r in records)
        docs = [r for r in inventory if r['fiscal_year'] == year]
        # Round-robin by rank above keeps one document from filling the budget.
        evidence = [dict(r, similarity=0.0) for r in candidates if r['year'] == year][:8]
        status = 'metadata' if records else 'not_confirmed'
        if evidence and not records:
            instruction = (
                f"Identify the firm that actually audited {ticker}'s ANNUAL financial statements for fiscal year {year}. "
                "Verify the fiscal period in the text: a filename year is only a retrieval hint. "
                "Separate consolidated and separate statements and report conflicting firms. "
                "Do not substitute interim review, a proposed appointment, a subsidiary auditor, "
                "or a prior-year comparator. Quote and cite the supporting pages. "
                "If the actual firm and fiscal period cannot be established, say not confirmed; "
                "do not say no documents exist. "
                + ('Answer in English.' if english else 'Trả lời bằng tiếng Việt.')
            )
            generated = answer_with_citation(instruction, evidence)
            offset = len(sources)
            rendered = re.sub(r'(\[(?:Source|Nguồn)\s+)(\d+)',
                              lambda m: m[1] + str(int(m[2]) + offset), generated['answer'])
            lines.append(rendered)
            sources.extend(generated['sources'])
            status = 'generation_failed' if generated.get('generation_fallback') else 'document_evidence_reviewed'
        elif not records:
            if not docs:
                message = 'No registered report for this year.' if english else 'Chưa có báo cáo đăng ký trong kho cho năm này.'
                status = 'no_registered_report'
            elif not any(r['chunks'] for r in docs):
                message = 'Reports exist, but searchable text has not been loaded.' if english else 'Có báo cáo nhưng chưa nạp văn bản để tra cứu.'
                status = 'not_chunked'
            else:
                message = 'Reports exist; the auditor was not confirmed from the selected text.' if english else 'Có báo cáo, nhưng chưa xác nhận được hãng kiểm toán từ văn bản truy xuất.'
            lines.append(message)
        coverage.append({'year': year, 'status': status, 'registered_reports': len(docs)})
    if not targets:
        lines.append('No year-attributed annual reports or auditor metadata are available.' if english else
                     'Chưa có báo cáo năm xác định được năm hoặc metadata kiểm toán để tổng hợp.')
    unknown = sum(r['fiscal_year'] is None for r in inventory)
    if unknown:
        lines.append(f"Reports with unknown year: {unknown}." if english else f"Có {unknown} báo cáo chưa xác định được năm; chưa gán vào bảng theo năm.")
    lines.append('Coverage is limited to registered local reports; missing evidence is not proof of no audit.' if english else
                 'Phạm vi chỉ gồm báo cáo đã đăng ký trong kho; thiếu bằng chứng không có nghĩa là doanh nghiệp không được kiểm toán.')
    return {'answer': '\n\n'.join(lines), 'sources': sources, 'coverage': coverage, 'metadata': metadata}
