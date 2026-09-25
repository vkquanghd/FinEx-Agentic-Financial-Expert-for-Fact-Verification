"""End-to-end query-time RAG pipeline: question -> embed -> similarity
search over pgvector -> LLM answer with citation. This is System S1
("RAG-only") in the thesis's 3-system comparison (S1 RAG-only / S2 DAG-only
/ S3 Router) - see src/structured/dag_pipeline.py for S2's equivalent.

Prerequisite: disclosure_chunks must already be populated (Step 10 in
README.md - src/retrieval/text_splitter.py + vector_store.py) and Ollama
must be running with the configured model pulled (Step 12 - see
src/models/llm_factory.py).

NOT run end-to-end against a real Ollama instance by the environment that
wrote it (no Ollama installed there) - same "written against documented
behavior, first real run is the first real test" situation as
market_snapshot.py and vector_store.py before it. The retrieval half
(similarity_search) IS already confirmed working (verified 2026-07-16,
see docs/PROGRESS.md) - only the LLM generation half is new/unverified.

Usage:
    python -m src.retrieval.rag_pipeline "Rủi ro pháp lý của công ty là gì?"
    python -m src.retrieval.rag_pipeline --ticker VNM "Chiến lược kinh doanh năm 2025 là gì?"
"""
import re
import sys
from datetime import date
from urllib.parse import urlparse

from src.causal.market_regime import classify_market_regime
from src.core.lang_detect import detect_language
from src.models.callbacks import timed_generation
from src.models.generation import answer_with_citation
# 2026-08-06: mọi lượt truy xuất query-time đi qua hybrid_search (dense +
# sparse RRF - xem comment LEXICAL_SQL/hybrid_search trong vector_store.py).
# hybrid_search tự fail-soft về dense thuần khi nhánh lexical lỗi, nên cứ
# thay thẳng similarity_search ở các call site bên dưới, không cần cờ riêng.
from src.retrieval.vector_store import hybrid_search as similarity_search
from src.retrieval.vector_store import lexical_search

NO_CONTEXT_MSG = (
    "Không tìm thấy đoạn trích liên quan nào trong tài liệu công bố thông tin "
    "đã nạp. Có thể do: (1) dữ liệu chưa được nạp vào pgvector (xem README Step 10), "
    "(2) câu hỏi nằm ngoài phạm vi các tài liệu hiện có, hoặc (3) trang liên quan "
    "vẫn đang trong hàng đợi OCR (data/processed/ocr_queue.csv - xem docs/PROGRESS.md)."
)

# English counterpart (2026-09-08, bilingual support - see
# src/models/generation.py's comment above SYSTEM_PROMPT for the full
# rationale). answer() picks this vs NO_CONTEXT_MSG by detect_language(question).
NO_CONTEXT_MSG_EN = (
    "No relevant excerpt was found in the loaded disclosure documents. "
    "Possible reasons: (1) the data hasn't been loaded into pgvector yet "
    "(see README Step 10), (2) the question is out of scope for the documents "
    "currently available, or (3) the relevant page is still in the OCR queue "
    "(data/processed/ocr_queue.csv - see docs/PROGRESS.md)."
)

_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _extract_question_years(question: str) -> list:
    return sorted({int(y) for y in _YEAR_RE.findall(question)})


# Document-type pre-filter (2026-07-24, FinAgentBench-inspired - see
# vector_store.SIMILARITY_SQL_TICKER_DOCTYPE's comment). Same 6 label set
# `scripts/pdf_ingest_router.py`/`scripts/tinix_ingest_adapter.py` write
# into `disclosure_chunks.doctype` at ingestion time - kept as ONE shared
# source of truth would be nicer, but those scripts' `guess_doctype()` is a
# filename-matching heuristic (Vietstock export names) while this is a
# question-text heuristic (Vietnamese natural-language phrasing) - different
# enough inputs that duplicating the label set here (not the logic) is
# clearer than trying to share one function across both.
#
# Deliberately conservative: order matters (more specific phrases checked
# first, e.g. "nghị quyết hội đồng quản trị" before the bare "hội đồng quản
# trị" that also appears in governance_report questions), and returns None
# on no match rather than guessing - an unfiltered search is always a safe
# fallback (see similarity_search()'s own fallback-if-empty behavior for the
# complementary case where the guess is wrong instead of absent).
_DOCTYPE_KEYWORDS = [
    ("nghị quyết hội đồng quản trị", "board_resolution"),
    ("nghị quyết hđqt", "board_resolution"),
    ("nghị quyết đại hội đồng cổ đông", "agm_resolution"),
    ("nghị quyết đhđcđ", "agm_resolution"),
    ("nghị quyết đại hội cổ đông", "agm_resolution"),
    ("báo cáo thường niên", "annual_report"),
    ("báo cáo quản trị", "governance_report"),
    ("giải trình kết quả kinh doanh", "kqkd_explanation"),
    ("giải trình", "kqkd_explanation"),
    ("báo cáo tài chính", "financial_statement"),
    ("bctc", "financial_statement"),
    ("financial statement", "financial_statement"),
    ("financial statements", "financial_statement"),
    ("financial disclosure", "financial_statement"),
    ("disclosure page", "financial_statement"),
]


def classify_doctype(question: str) -> str:
    """Best-effort, keyword-only classification of which of the 6 Vietstock
    document types a question is most likely asking about - returns None
    (no filter applied) rather than a guess when nothing matches, since a
    wrong REQUIRED filter is worse than no filter (see similarity_search's
    doctype fallback for the complementary safety net when this guesses
    wrong instead of not guessing at all)."""
    low = question.lower()
    for phrase, doctype in _DOCTYPE_KEYWORDS:
        if phrase in low:
            return doctype
    return None


def expand_financial_terms_for_retrieval(question: str) -> str:
    """Append canonical Vietnamese labels for bilingual financial retrieval.

    The disclosure corpus is Vietnamese.  BGE-M3 is multilingual, but exact
    line-item requests benefit from placing the official Vietnamese labels in
    the retrieval query as well.  This only changes the search vector; the
    original question remains the generation prompt and audit record.
    """
    low = question.lower()
    additions = []
    if "revenue" in low or "net sales" in low:
        additions.append("Doanh thu thuần")
    if any(term in low for term in ("net profit", "net income", "profit after tax")):
        additions.append("Lợi nhuận sau thuế")
    if "gross profit" in low:
        additions.append("Lợi nhuận gộp")
    return f"{question} {' '.join(additions)}" if additions else question


def _required_financial_labels(question: str) -> list[str]:
    """Known report labels whose evidence should accompany a compound query."""
    low = question.lower()
    labels = []
    if "revenue" in low or "net sales" in low or "doanh thu" in low:
        labels.append("Doanh thu thuần")
    if any(term in low for term in (
        "net profit", "net income", "profit after tax", "lợi nhuận ròng", "lợi nhuận sau thuế",
    )):
        labels.append("Lợi nhuận sau thuế")
    return labels


def _prepend_required_statement_evidence(chunks: list, question: str, ticker: str,
                                        year: int | None, top_k: int) -> list:
    """Ensure compound statement questions expose exact line-item pages.

    Dense retrieval is still the main ranking method.  This narrow FTS pass
    only supplies pages containing explicitly requested, canonical statement
    labels and costs no additional BGE-M3 embedding call.
    """
    if not ticker or not year:
        return chunks[:top_k]
    labels = _required_financial_labels(question)
    if not labels:
        return chunks[:top_k]

    # Labels alone appear in accounting notes.  In this corpus, the actual
    # income-statement pages carry both the statement heading and form code
    # B 02, whereas the explanatory notes use B 09.  Retrieve that document
    # shape once, then keep only chunks that meet both textual guards.
    statement_query = '"Báo cáo kết quả hoạt động kinh doanh" "Mẫu B 02"'
    candidates = lexical_search(statement_query, top_k=20, ticker=ticker, year=year)
    required = [
        chunk for chunk in candidates
        if "báo cáo kết quả hoạt động kinh doanh" in chunk["chunk_text"].casefold()
        and "mẫu b 02" in chunk["chunk_text"].casefold()
    ]

    seen = set()
    merged = []
    for chunk in required + chunks:
        key = (
            chunk["ticker"], chunk["source_file"], chunk["page"],
            chunk.get("chunk_index", 0),
        )
        if key not in seen:
            seen.add(key)
            merged.append(chunk)
    return merged[:top_k]


def apply_temporal_anchor(chunks: list, question: str, top_k: int):
    """RQ4 (Temporal Anchor Filtering). Plain cosine similarity has no
    notion of "which year" - for a single-company question with no year
    named, a chunk from 2016 can rank ahead of 2024 just because its
    phrasing happens to score higher, even for facts that genuinely change
    year to year (board composition, executives, ownership). Confirmed
    live 2026-07-19: "Hội đồng quản trị PNJ gồm những ai?" (no year named)
    surfaced only 2015-2023 governance-table chunks in the top ranks, with
    the current (2024) board list absent even after fixing the retrieval
    bugs that were separately blocking it - see docs/PROGRESS.md. This
    takes an already-retrieved candidate pool (fetch more than `top_k` so
    there's something to anchor from) and:
      - question names a year -> keep only that year's chunks (falls back
        to the unfiltered pool, not an empty list, if that year isn't
        present in the candidates - a wrong year beats a hard failure)
      - no year named -> prefer the MOST RECENT year present in the pool
        (mirrors the year-1 default already used for What-if's annual-data
        lookups in src/router/rules.py), filling any remaining top_k slots
        from older years by similarity if that year alone doesn't have enough

    Returns (chunks, note) - `note` is a short Vietnamese string describing
    what anchoring happened (or None if no chunk had an extractable year,
    e.g. the real-PDF corpus's non-year-bearing filenames) - surfaced to
    the LLM prompt and the CLI output so an answer never silently implies
    "this is current" when it's actually anchored to an older year."""
    known = [c for c in chunks if c.get("year")]
    if not known:
        return chunks[:top_k], None

    q_years = _extract_question_years(question)
    if q_years:
        target = q_years[-1]
        filtered = [c for c in known if c["year"] == target]
        if filtered:
            return filtered[:top_k], f"đã ưu tiên nguồn năm {target} (nêu rõ trong câu hỏi)"
        return chunks[:top_k], f"không có nguồn năm {target} trong kết quả truy xuất, dùng similarity gốc"

    latest = max(c["year"] for c in known)
    preferred = [c for c in known if c["year"] == latest]
    if len(preferred) >= top_k:
        return preferred[:top_k], f"câu hỏi không nêu năm - đã ưu tiên năm gần nhất có dữ liệu ({latest})"
    rest = sorted(
        (c for c in known if c["year"] != latest),
        key=lambda c: (-c["year"], -c["similarity"]),
    )
    combined = (preferred + rest)[:top_k]
    other_years = sorted({c["year"] for c in combined if c["year"] != latest}, reverse=True)
    note = f"câu hỏi không nêu năm - đã ưu tiên năm gần nhất có dữ liệu ({latest})"
    if other_years:
        note += f"; một số nguồn khác năm cũng được giữ lại ({other_years}) do không đủ nguồn cùng năm {latest}"
    return combined, note


def rerank_for_regime(chunks: list, regime: str, top_k: int) -> tuple:
    """RQ3 (Trend-Aligned Dynamic Reranking) - see src/causal/market_regime.py's
    module docstring for the academic framing (History Rhymes,
    arXiv:2511.09754) and why this is a recency-weighted rerank rather than
    that paper's joint macro-text embedding.

    Only "bear"/"volatile" regimes change anything: when the market itself
    is stressed/choppy, an older disclosure chunk is a riskier proxy for
    the company's CURRENT state than the same chunk would be in a calm
    market, so recency gets an extra boost relative to pure similarity.
    "bull"/"stable"/"unknown" regimes are a no-op (pure similarity order,
    unchanged) - this identity behavior is also what "Router tĩnh" (the
    RQ3 ablation baseline with the regime signal disabled) reduces to,
    since answer(use_regime_signal=False) forces regime="stable" upstream
    rather than needing a second code path here.

    Returns (reranked_chunks, note) - note is None if nothing changed
    (regime not in bear/volatile, or no chunk has an extractable year)."""
    if regime not in ("bear", "volatile"):
        return chunks, None
    known = [c for c in chunks if c.get("year")]
    if not known:
        return chunks, None

    current_year = date.today().year
    # Each year of age costs 0.03 similarity-equivalent - a round, documented
    # judgment call (not fit to data - same "no annotated ranking data to
    # tune against in 10 weeks" situation as every other threshold in this
    # project), strong enough to reorder close calls without letting a
    # much-older chunk outrank a clearly-more-relevant recent one purely on
    # a small similarity edge.
    #
    # Neo literature (thêm 2026-09-12, vá khoảng trống "0.03 ở đâu ra? không
    # neo literature" - docs/RQ_audit_truoc_khi_training_2026-08-14.md dòng
    # 39-43/102-104): Grofsky, M. (2025). "Solving Freshness in RAG: A Simple
    # Recency Prior and the Limits of Heuristic Trend Detection."
    # arXiv:2509.19376. Bài này khảo sát ĐÚNG loại kỹ thuật đang dùng ở đây -
    # một "recency prior" đơn giản, tuyến tính theo tuổi tài liệu, không tinh
    # chỉnh trên dữ liệu gán nhãn - và tên bài cũng nói rõ giới hạn của cách
    # làm này ("Limits of Heuristic..."), khớp với chính cách project này tự
    # mô tả 0.03 là "a round, documented judgment call (not fit to data)"
    # thay vì tuyên bố đây là con số tối ưu. Trích dẫn này neo VIỆC dùng
    # recency prior là một hướng tiếp cận có trong literature 2023+, KHÔNG
    # tuyên bố giá trị 0.03 cụ thể đến từ bài báo (giá trị cụ thể vẫn là
    # judgment call của project, ghi rõ để tránh ngộ nhận).
    RECENCY_PENALTY_PER_YEAR = 0.03

    def blended_score(c):
        # .get("year", current_year) KHÔNG đủ: default chỉ áp dụng khi khóa
        # "year" hoàn toàn vắng mặt, còn ở đây khóa luôn tồn tại trong dict
        # (chunk từ disclosure_chunks) nhưng giá trị có thể là None (file
        # nguồn không có năm trong tên) - default không bao giờ được dùng,
        # và current_year - None ném TypeError. Lỗi thật, bắt được
        # 2026-08-07 khi hỏi "Xin chào" (câu chào chung chung khớp một hỗn
        # hợp chunk có/không năm) trong lúc regime là bear/volatile.
        year = c.get("year")
        if year is None:
            year = current_year
        age = max(0, current_year - year)
        return c["similarity"] - RECENCY_PENALTY_PER_YEAR * age

    before_top = [id(c) for c in chunks[:top_k]]
    reordered = sorted(chunks, key=blended_score, reverse=True)
    after_top = [id(c) for c in reordered[:top_k]]
    changed = before_top != after_top
    note = (
        f"chu kỳ thị trường '{regime}' - đã tăng trọng số ưu tiên nguồn gần đây hơn"
        + (" (thứ tự kết quả đã thay đổi)" if changed else " (thứ tự không đổi ở top_k này)")
    )
    return reordered, note


# WEB FALLBACK - added 2026-09-07, see infra/postgres/migrations/014_web_chunks.sql's
# header comment for the full rationale: found live that VNM's OWN official
# risk-management disclosure is genuinely thin (checked against the real
# PDF), so a question like "Rủi ro pháp lý của VNM là gì?" can correctly
# return NO_CONTEXT_MSG-worthy results from disclosure_chunks even after
# every relevant PDF is ingested - the answer, if it exists at all, has to
# come from press coverage instead. This is a SEPARATE, clearly-labeled
# evidence tier (see WEB_FALLBACK_SYSTEM_PROMPT below) - never silently
# blended with disclosure_chunks results, and only tried when
# disclosure_chunks has NOTHING (see answer() below), so a real disclosure
# answer never gets "diluted" by a lower-trust web source sitting next to
# it. scripts/web_news_ingest.py is what populates web_chunks in the first
# place (search + persisted snapshot, NOT a live query-time web call - see
# that script's docstring for why this must be pre-populated, not fetched
# per-question, same reproducibility argument as everywhere else in this
# project).
WEB_FALLBACK_SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi tài chính dựa trên các bài báo/nguồn web đã được lưu lại "
    "(KHÔNG phải BCTC, báo cáo thường niên hay tài liệu công bố chính thức). QUY TẮC BẮT BUỘC:\n"
    "1. LUÔN bắt đầu câu trả lời bằng đúng câu: \"Theo báo chí/nguồn web (KHÔNG phải tài liệu công bố "
    "chính thức):\" - để người đọc không nhầm đây là thông tin đã kiểm chứng từ BCTC/báo cáo thường niên.\n"
    "2. Chỉ trả lời dựa trên các đoạn trích được cung cấp bên dưới - KHÔNG dùng kiến thức nền để bổ "
    "sung số liệu/sự kiện không có trong đoạn trích.\n"
    "3. Nếu các đoạn trích không đủ thông tin, nói rõ \"Không tìm thấy thông tin trong các nguồn web "
    "đã lưu\" thay vì đoán.\n"
    "4. Với MỖI khẳng định cụ thể, ghi [Nguồn N] kèm một cụm trích dẫn ngắn (tối đa ~15 từ) lấy trực "
    "tiếp từ đúng đoạn trích N.\n"
    "5. Nếu đoạn trích có ghi ngày, nêu rõ ngày đó - tin tức cũ có thể không còn đúng."
)

# English counterpart (2026-09-08, bilingual support). web_fallback_answer()
# picks this vs WEB_FALLBACK_SYSTEM_PROMPT by detect_language(question).
WEB_FALLBACK_SYSTEM_PROMPT_EN = (
    "You are a financial QA assistant answering based on saved press/web articles "
    "(NOT financial statements, annual reports, or official disclosure documents). "
    "MANDATORY RULES:\n"
    "0. Write your ENTIRE answer in English, and ONLY English, even though the "
    "excerpts below are in Vietnamese - the only Vietnamese allowed anywhere in "
    "your answer is a proper name/title quoted verbatim as part of a [Source N] citation.\n"
    "1. ALWAYS start your answer with exactly this sentence: \"According to press/web "
    "sources (NOT an official disclosure document):\" - so the reader never mistakes "
    "this for verified financial-statement/annual-report information.\n"
    "2. Answer ONLY from the excerpts provided below - do NOT use background "
    "knowledge to add numbers/facts not present in the excerpts.\n"
    "3. If the excerpts don't have enough information, say clearly \"No information "
    "found in the saved web sources\" instead of guessing.\n"
    "4. For EVERY specific claim, cite [Source N] with a short direct quote (at most "
    "~15 words) taken directly from excerpt N.\n"
    "5. If an excerpt has a date, state it - old news may no longer be accurate."
)


def _live_web_chunks(question: str, ticker: str = None, top_k: int = 5) -> list:
    """Return eligible, in-memory web chunks for an explicitly enabled demo
    lookup. This is intentionally separate from the persisted corpus: it is
    not reproducible and never belongs in benchmark metrics."""
    from scripts.web_fallback_lookup import DEFAULT_SITES, fetch_page_text, web_search
    from scripts.web_news_ingest import is_relevant_result

    query = f"{ticker} {question}" if ticker else question
    try:
        hits = web_search(query, sites=DEFAULT_SITES, max_results=top_k)
    except (SystemExit, Exception):
        return []

    chunks = []
    for hit in hits:
        body = fetch_page_text(hit["url"]) or hit.get("snippet", "")
        if not body or not is_relevant_result(
            ticker, question, hit.get("title", ""), hit.get("snippet", ""), body
        ):
            continue
        domain = urlparse(hit["url"]).netloc.removeprefix("www.")
        if not domain:
            continue
        chunks.append({
            "domain": domain,
            "source_url": hit["url"],
            "title": hit.get("title", ""),
            "published_date": None,
            "chunk_text": body,
            "similarity": 0.0,
        })
    return chunks


def web_fallback_answer(question: str, ticker: str = None, top_k: int = 5) -> dict:
    """Tried by answer() ONLY when disclosure_chunks (official filings)
    returns nothing for this question - see the module comment above.
    Returns None (not a dict) if web_chunks also has nothing, so the caller
    falls through to the original NO_CONTEXT_MSG rather than fabricating a
    web answer out of zero evidence."""
    from src.models import llm_factory
    from src.retrieval.vector_store import web_similarity_search

    try:
        chunks = web_similarity_search(question, top_k=top_k, ticker=ticker)
    except Exception:
        chunks = []

    # The persistent web corpus is optional. An empty table should have the
    # same explicit demo fallback behavior as a missing table, but only when
    # the operator opts in because live search cannot be reproduced later.
    if not chunks:
        import os
        if os.getenv("FINEX_LIVE_WEB_FALLBACK", "0") == "1":
            chunks = _live_web_chunks(question, ticker=ticker, top_k=top_k)
    if not chunks:
        return None

    # Bilingual (2026-09-08) - same detect_language() pick as
    # src.models.generation.answer_with_citation, applied here too since
    # this is a second, separate prompt-building path (see module comment).
    lang = detect_language(question)
    if lang == "en":
        parts = []
        for i, c in enumerate(chunks, start=1):
            date_part = f", date {c['published_date']}" if c.get("published_date") else ""
            parts.append(f"[Source {i}: {c['domain']}{date_part}]\n{c['chunk_text']}")
        context = "\n\n".join(parts)
        prompt = f"Press/web excerpts:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer:"
        system_prompt = WEB_FALLBACK_SYSTEM_PROMPT_EN
    else:
        parts = []
        for i, c in enumerate(chunks, start=1):
            date_part = f", ngày {c['published_date']}" if c.get("published_date") else ""
            parts.append(f"[Nguồn {i}: {c['domain']}{date_part}]\n{c['chunk_text']}")
        context = "\n\n".join(parts)
        prompt = f"Các đoạn trích báo chí/web:\n\n{context}\n\n---\n\nCâu hỏi: {question}\n\nTrả lời:"
        system_prompt = WEB_FALLBACK_SYSTEM_PROMPT

    answer_text, elapsed = timed_generation(
        llm_factory.generate, prompt, system=system_prompt
    )

    return {
        "question": question,
        "answer": answer_text,
        "sources": [
            {
                "domain": c["domain"], "source_url": c["source_url"], "title": c.get("title"),
                "published_date": c.get("published_date"), "similarity": round(c["similarity"], 4),
            }
            for c in chunks
        ],
        "elapsed_s": round(elapsed, 2),
        "source_tier": "web_fallback",  # never present on a normal disclosure_chunks answer - callers (Streamlit/API) use this to render the warning banner
        "answer_lang": lang,
        # P1 (2026-09-12) - fabrication risk applies just as much to a web-
        # sourced answer as a disclosure-sourced one, so verify here too. See
        # verify_numeric_claims()'s module comment (defined below this
        # function) for the design/assumptions.
        "numeric_verification": verify_numeric_claims(answer_text, chunks),
    }


# ---------------------------------------------------------------------------
# P1 (2026-09-12, thiết kế trong docs/FinEx_rebrand_and_Q2_roadmap_2026-09-07.md
# §6c) - bước verify khớp số liệu: so con số LLM nêu trong answer với con số
# thật sự xuất hiện trong text của các chunk đã trích dẫn, để bắt trường hợp
# LLM suy diễn/làm tròn/bịa một con số không có trong nguồn. Có citation
# (đã có, xem answer_with_citation()) chỉ chứng minh "có trích một chunk nào
# đó" - KHÔNG chứng minh "số trong câu trả lời khớp với đúng chunk đó". Đây
# là khoảng trống thật §4 mục 1 của roadmap chỉ ra, không phải rủi ro giả
# định - xem thêm cuộc trao đổi 2026-09-12 về cách ngăn LLM bịa số.
#
# QUYẾT ĐỊNH CHUẨN HOÁ ĐƠN VỊ (roadmap §6c mục 3 nói rõ đây là việc CẦN Quang
# xác nhận trước khi code - dưới đây là GIẢ ĐỊNH mặc định tôi chọn để có thể
# code ngay, KHÔNG phải quyết định đã được Quang duyệt. Coi nhãn
# "verified"/"unverified_claim_mismatch" là DIAGNOSTIC/nội bộ cho tới khi
# Quang xác nhận hoặc chỉnh lại quy tắc dưới đây, đừng đưa thẳng vào số liệu
# chính thức của luận văn):
#   - Số có NHIỀU dấu "." hoặc "," -> mọi dấu đều là phân cách nghìn (vd
#     "12.500.000" = 12500000). Đây là văn phong phổ biến nhất trong BCTC
#     tiếng Việt cho số nguyên lớn.
#   - Số có ĐÚNG 1 dấu, theo sau bởi 1-2 chữ số ở CUỐI chuỗi -> dấu đó là
#     thập phân (vd "12,5" hay "12.5" đều = 12.5) - áp dụng cho cả kiểu viết
#     thập phân VN (dùng ",") lẫn EN (dùng ".") vì bài toán này cần so khớp
#     answer/nguồn có thể lệch ngôn ngữ (demo song ngữ VI/EN).
#   - Số có ĐÚNG 1 dấu, theo sau bởi ĐÚNG 3 chữ số -> GIẢ ĐỊNH đây là phân
#     cách nghìn (vd "12.500" = 12500), KHÔNG phải thập phân 3 chữ số - vì
#     "phân cách nghìn 3 chữ số" phổ biến hơn nhiều trong BCTC so với "thập
#     phân đúng 3 chữ số". RỦI RO ĐÃ BIẾT: một số cực hiếm kiểu "1,234" nghĩa
#     là 1.234 (thập phân) sẽ bị đọc sai thành 1234 - chấp nhận rủi ro này vì
#     tần suất thấp hơn nhiều so với cách đọc ngược lại.
#   - Ngưỡng "khớp": sai số tương đối <=2% (cho phép làm tròn/khác đơn vị in
#     ấn) HOẶC (nếu là %) sai số tuyệt đối <=0.2 điểm phần trăm - lấy điều
#     kiện dễ đạt hơn trong 2 điều kiện. Đây là judgment call, giống mọi
#     ngưỡng khác trong project (không có dữ liệu gán nhãn để tune trong 10
#     tuần) - CẦN Quang xác nhận trước khi coi là số liệu chính thức.
#
# GIẢI THÍCH ĐƯỢC (justification, thêm 2026-09-12): literature fact-checking
# coi "explanation/justification generation" là MỘT GIAI ĐOẠN BẮT BUỘC của
# pipeline, không phải phần thêm cho đẹp - Eldifrawi et al. (2024), "Automated
# Justification Production for Claim Veracity in Fact Checking: A Survey on
# Architectures and Approaches", ACL 2024, mô tả pipeline chuẩn: claim
# detection -> evidence retrieval -> veracity prediction -> justification
# generation. Bản đầu tiên của verify_numeric_claims() chỉ trả
# matched=True/False (phán quyết KHÔNG giải thích - không nói "sai so với số
# nào, lệch bao nhiêu"), chưa đạt giai đoạn 4. Vá ngay dưới đây: mỗi claim
# giờ có thêm "closest_source_value" (con số trong nguồn gần nhất cùng loại
# %/tuyệt đối) và "diff_pct"/"diff_points" (mức lệch cụ thể) - "matched" vẫn
# giữ nguyên Ý NGHĨA và NGƯỠNG như trước (xét TẤT CẢ candidate nguồn, không
# chỉ candidate gần nhất - closest chỉ dùng để BÁO CÁO, không dùng để quyết
# định matched, nên hành vi cũ không đổi).
# ---------------------------------------------------------------------------
_UNIT_MULTIPLIER = {
    "tỷ đồng": 1e9, "tỷ": 1e9, "billion": 1e9,
    "triệu đồng": 1e6, "triệu": 1e6, "million": 1e6,
    "nghìn đồng": 1e3, "nghìn": 1e3, "thousand": 1e3,
}
_PERCENT_UNITS = {"%", "percent", "phần trăm"}

# Chỉ bắt số có ĐƠN VỊ tài chính rõ ràng HOẶC có dấu phân cách/thập phân -
# loại trừ năm ("2024"), số trang, thứ tự [Nguồn N] (đều là số trần, không
# đơn vị, không dấu phân cách) khỏi việc bị coi nhầm là "numeric claim" cần
# verify - đúng tinh thần thiết kế gốc trong roadmap ("regex tiếng Việt: 'tỷ
# đồng', 'triệu đồng', '%', dấu phẩy/chấm thập phân").
_NUMERIC_CLAIM_RE = re.compile(
    r"(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+[.,]\d+)"
    r"\s*(tỷ đồng|triệu đồng|nghìn đồng|tỷ|triệu|nghìn|billion|million|thousand|%|percent|phần trăm)?",
    re.IGNORECASE,
)
# Số trần (không dấu phân cách) NHƯNG có đơn vị tài chính ngay sau (vd "50%",
# "3 tỷ") cũng phải bắt được - _NUMERIC_CLAIM_RE ở trên yêu cầu dấu phân cách
# HOẶC match nhóm 2; \d+[.,]\d+ đã đủ cho số có dấu, còn số trần+đơn vị cần
# nhánh riêng vì \d{1,3}(?:[.,]\d{3})+ bắt buộc có dấu phân cách.
_BARE_WITH_UNIT_RE = re.compile(
    r"(?<![\d.,])(\d+)\s*(tỷ đồng|triệu đồng|nghìn đồng|tỷ|triệu|nghìn|billion|million|thousand|%|percent|phần trăm)",
    re.IGNORECASE,
)


def _normalize_number(raw: str) -> float:
    """Xem comment ở khối trên _NUMERIC_CLAIM_RE cho quy tắc chuẩn hoá và rủi
    ro đã biết của quy tắc này."""
    seps = re.findall(r"[.,]", raw)
    if not seps:
        return float(raw)
    last_sep_pos = max(raw.rfind("."), raw.rfind(","))
    digits_after_last = len(raw) - last_sep_pos - 1
    if len(seps) == 1 and digits_after_last in (1, 2):
        return float(raw[:last_sep_pos] + "." + raw[last_sep_pos + 1:])
    return float(re.sub(r"[.,]", "", raw))


def _extract_numeric_claims(text: str) -> list:
    """Trả về [(value, is_percent), ...] - value đã nhân theo đơn vị (tỷ/
    triệu/nghìn -> giá trị VND tuyệt đối); % giữ nguyên số phần trăm gốc (vd
    "15%" -> (15.0, True), KHÔNG đổi thành 0.15) để so percent-với-percent
    đúng thang đo với text nguồn."""
    claims = []
    seen_spans = set()
    for pattern in (_NUMERIC_CLAIM_RE, _BARE_WITH_UNIT_RE):
        for match in pattern.finditer(text):
            span = match.span()
            if span in seen_spans:
                continue
            seen_spans.add(span)
            raw, unit = match.group(1), (match.group(2) or "").strip().lower()
            try:
                value = _normalize_number(raw)
            except ValueError:
                continue
            is_percent = unit in _PERCENT_UNITS
            if unit in _UNIT_MULTIPLIER:
                value *= _UNIT_MULTIPLIER[unit]
            claims.append((value, is_percent))
    return claims


def _closest_source_claim(value: float, is_percent: bool, source_claims: list):
    """Chọn candidate nguồn GẦN NHẤT với `value` (cùng loại %/tuyệt đối), để
    BÁO CÁO giải thích ("so với số nào") - KHÔNG dùng để quyết định
    "matched" (xem verify_numeric_claims() bên dưới - matched xét TẤT CẢ
    candidate, không chỉ candidate gần nhất, để không đổi hành vi cũ). Trả
    về None nếu không có candidate nguồn nào cùng loại %/tuyệt đối - khi đó
    không thể tính "lệch bao nhiêu" một cách có ý nghĩa.

    Điểm % dùng khoảng cách tuyệt đối (điểm phần trăm là đơn vị người đọc
    hiểu trực tiếp); giá trị tuyệt đối (VND) dùng sai số TƯƠNG ĐỐI (đúng
    thang đo đang dùng để xét ngưỡng khớp - xem module comment phía trên)."""
    candidates = [s_value for s_value, s_is_percent in source_claims if s_is_percent == is_percent]
    if not candidates:
        return None
    if is_percent:
        return min(candidates, key=lambda s: abs(value - s))
    return min(candidates, key=lambda s: abs(value - s) if s == 0 else abs(value - s) / abs(s))


def verify_numeric_claims(answer_text: str, chunks: list) -> dict:
    """P1 roadmap item (docs/FinEx_rebrand_and_Q2_roadmap_2026-09-07.md §6c).
    So mỗi con số tìm thấy trong `answer_text` với các con số THẬT SỰ có
    trong text của `chunks` đã trích dẫn. CẦN gọi với `chunks` gốc (có
    "chunk_text") - KHÔNG phải "sources" đã rút gọn mà answer_with_citation()
    trả về (generation.py cố tình bỏ chunk_text khỏi "sources" để log eval
    nhẹ hơn - xem answer_with_citation()'s docstring).

    Trả về {"label": "verified" | "unverified_claim_mismatch" |
    "no_numeric_claim", "claims": [{"value", "is_percent", "matched",
    "closest_source_value", "diff_pct", "diff_points"}]}. "verified" chỉ khi
    TẤT CẢ claim đều khớp - một con số sai đủ để gắn nhãn cả câu trả lời
    "unverified_claim_mismatch" (một số bịa vẫn là bịa, dù các số khác đúng -
    cùng logic RQ6's negative-control không để 1 kết quả tốt che giấu 1 kết
    quả xấu).

    "closest_source_value"/"diff_pct"/"diff_points" (thêm 2026-09-12, xem
    module comment phía trên _UNIT_MULTIPLIER, mục "GIẢI THÍCH ĐƯỢC") là
    phần GIẢI THÍCH cho "matched" - trả lời "so với số nào trong nguồn, lệch
    bao nhiêu" thay vì chỉ True/False trơn. `closest_source_value` là None
    nếu KHÔNG có con số nào cùng loại (%/tuyệt đối) trong nguồn để so - khi
    đó `matched` chắc chắn là False (không có gì để khớp). `diff_pct` (sai
    số tương đối, đơn vị %) chỉ có giá trị khi `is_percent=False`;
    `diff_points` (chênh lệch điểm phần trăm) chỉ có giá trị khi
    `is_percent=True` - trường còn lại luôn là None. `diff_pct` cũng là None
    khi `closest_source_value=0` (chia cho 0 không có nghĩa tương đối).

    LƯU Ý (xem khối comment lớn phía trên _UNIT_MULTIPLIER): ngưỡng khớp và
    quy tắc chuẩn hoá đơn vị ở đây là GIẢ ĐỊNH mặc định, CHƯA được Quang xác
    nhận theo đúng yêu cầu của roadmap §6c mục 3 - coi nhãn trả về là
    diagnostic/nội bộ, chưa phải số liệu chính thức của luận văn."""
    answer_claims = _extract_numeric_claims(answer_text)
    if not answer_claims:
        return {"label": "no_numeric_claim", "claims": []}

    source_text = " ".join(c.get("chunk_text", "") for c in chunks)
    source_claims = _extract_numeric_claims(source_text)

    def _matches(value, is_percent):
        for s_value, s_is_percent in source_claims:
            if is_percent != s_is_percent:
                continue
            if is_percent and abs(value - s_value) <= 0.2:
                return True
            if s_value != 0 and abs(value - s_value) / abs(s_value) <= 0.02:
                return True
            if s_value == 0 and value == 0:
                return True
        return False

    claim_results = []
    all_matched = True
    for value, is_percent in answer_claims:
        matched = _matches(value, is_percent)
        all_matched = all_matched and matched

        closest = _closest_source_claim(value, is_percent, source_claims)
        diff_pct = None
        diff_points = None
        if closest is not None:
            if is_percent:
                diff_points = round(abs(value - closest), 4)
            elif closest != 0:
                diff_pct = round(abs(value - closest) / abs(closest) * 100, 4)

        claim_results.append({
            "value": value, "is_percent": is_percent, "matched": matched,
            "closest_source_value": closest, "diff_pct": diff_pct, "diff_points": diff_points,
        })

    label = "verified" if all_matched else "unverified_claim_mismatch"
    return {"label": label, "claims": claim_results}


def answer(question: str, top_k: int = 5, ticker: str = None, use_regime_signal: bool = True) -> dict:
    """Returns the same dict shape as src.models.generation.answer_with_citation:
    {"question", "answer", "sources", "elapsed_s"}, plus "temporal_note"
    when temporal anchoring (RQ4) did something worth surfacing. `ticker`,
    if given, is pushed straight into similarity_search's SQL WHERE clause
    (not filtered client-side after the fact) - see
    vector_store.SIMILARITY_SQL_TICKER's comment for why: at 18k+ chunks, a
    global top-K search for a single-company question can return zero
    chunks for that company even though it has thousands of rows in the
    table, because none of them happen to rank in the *global* top-K.
    Confirmed as a real bug 2026-07-19.

    Two different strategies depending on whether the question names a year,
    both existing to fix real (not hypothetical) failures found 2026-07-19:

    1. Question names a year (e.g. "...năm 2024...") -> filter to that
       year's files IN SQL first (similarity_search(..., year=target)), so
       top_k is a true top-K within that year. A first attempt at this
       (fetch a wide top-K across ALL years, then filter to the named year
       client-side, widening fetch_k to try to compensate) still failed:
       "Công ty kiểm toán của PNJ năm 2024 là ai?"'s correct chunk ranked
       #33 GLOBALLY (not deeply buried) but PNJ has enough OTHER 2024
       chunks that outrank it globally without answering the question, so
       no reasonable global fetch_k reliably left it in the final top_k
       after filtering. SQL-level year filtering avoids the problem
       entirely (see vector_store.SIMILARITY_SQL_TICKER_YEAR). Falls back
       to strategy 2 if the year-scoped search returns nothing (e.g. no
       file matches that year's naming pattern).
    2. Question does NOT name a year -> can't SQL-filter by year (don't
       know which one yet), so fetch a wide candidate pool across all years
       and let apply_temporal_anchor() pick the most recent year present
       client-side, same as before.

    Also applies a document-type pre-filter (2026-07-24, FinAgentBench-
    inspired - see classify_doctype() and vector_store.similarity_search's
    `doctype` param) whenever the question's phrasing confidently matches
    one of the 6 known document types. Passed into every similarity_search()
    call below alongside ticker/year - self-heals via similarity_search's
    own empty-result fallback if the guess is wrong, so it never needs its
    own retry logic here the way the year-scoped branch does above.

    `use_regime_signal` (RQ3, Trend-Aligned Dynamic Reranking, default True):
    when True, classify_market_regime() reads the current VN-Index
    trend/volatility and rerank_for_regime() uses it to boost recency in
    the ranking of any candidate pool that isn't already hard-filtered to
    one exact year (see that function's docstring). Pass False to force
    the "Router tĩnh" ablation baseline for RQ3 - identical behavior to the
    regime signal always reading "stable" (pure similarity ranking,
    unchanged from before this feature existed)."""
    retrieval_question = expand_financial_terms_for_retrieval(question)
    q_years = _extract_question_years(question)
    doctype = classify_doctype(question)
    if use_regime_signal:
        regime_info = classify_market_regime()
    else:
        regime_info = {
            "ok": True, "regime": "stable",
            "reason": "Tín hiệu chu kỳ thị trường tắt (ablation RQ3 - Router tĩnh).",
        }
    regime = regime_info["regime"]
    temporal_note = None
    regime_note = None
    chunks = []

    if ticker and q_years:
        target = q_years[-1]
        # Even scoped to a single ticker+year, top_k=5 is still too narrow
        # for some facts - confirmed live 2026-07-19: within PNJ's 463
        # chunks from 2024-dated files alone, the chunk actually naming the
        # auditor (PwC) ranked #9-10 by similarity, behind several
        # boilerplate financial-statement chunks that just happen to share
        # more surface vocabulary with the question ("kiểm toán", "PNJ",
        # "2024") without answering it. Since the pool is already
        # year-scoped (SQL-filtered, not a global fetch), widening this is
        # cheap and doesn't reintroduce cross-year noise - see
        # scripts/rank_within_year.py (diagnostic, safe to delete).
        year_top_k = max(top_k * 3, 12)
        chunks = similarity_search(retrieval_question, top_k=year_top_k, ticker=ticker, year=target, doctype=doctype)
        # The wider query improves recall, but the generator must receive the
        # requested context budget, not every candidate.  Passing all 12-15
        # chunks here made a year-scoped question substantially slower than
        # its `top_k` setting promised and could exhaust the local Qwen
        # timeout before it produced an answer.
        chunks = chunks[:top_k]
        chunks = _prepend_required_statement_evidence(
            chunks, question, ticker, target, top_k,
        )
        if chunks:
            temporal_note = f"đã ưu tiên nguồn năm {target} (nêu rõ trong câu hỏi, lọc theo năm trực tiếp trong SQL)"
        # No regime rerank here: the pool is already hard-filtered to one
        # exact year, so a recency-vs-similarity blend has nothing to work
        # with - every candidate has the same age.

    if ticker and not chunks:
        # No year named, or the year-scoped SQL search came up empty
        # (e.g. non-TiniX source with no year in its filename) - fetch wide
        # across all years and filter/rank client-side instead.
        fetch_k = max(top_k * 10, 40)
        wide = similarity_search(retrieval_question, top_k=fetch_k, ticker=ticker, doctype=doctype)
        chunks, temporal_note = apply_temporal_anchor(wide, question, top_k)
        chunks, regime_note = rerank_for_regime(chunks, regime, top_k)
    elif not ticker:
        # Fetch a wider pool than top_k so rerank_for_regime() actually has
        # something to reorder (a pool truncated to exactly top_k up front
        # leaves no room for a recency-boosted chunk to rise into view).
        fetch_k = max(top_k * 3, 15)
        wide_pool = similarity_search(retrieval_question, top_k=fetch_k, doctype=doctype)
        chunks, regime_note = rerank_for_regime(wide_pool, regime, top_k)
        chunks = chunks[:top_k]

    if not chunks:
        # disclosure_chunks (official filings) has nothing for this
        # question - try the separate web_chunks fallback before giving up
        # (see web_fallback_answer()'s docstring / the module comment above
        # it). Returns None if web_chunks ALSO has nothing, so this still
        # falls through to NO_CONTEXT_MSG rather than ever fabricating an
        # answer from zero evidence.
        web_result = web_fallback_answer(question, ticker=ticker)
        if web_result:
            return web_result
        lang = detect_language(question)
        msg = NO_CONTEXT_MSG_EN if lang == "en" else NO_CONTEXT_MSG
        return {"question": question, "answer": msg, "sources": [], "elapsed_s": 0.0, "answer_lang": lang}

    result = answer_with_citation(question, chunks)
    if temporal_note:
        result["temporal_note"] = temporal_note
    if doctype:
        result["doctype_filter"] = doctype
    result["market_regime"] = regime
    result["market_regime_method"] = regime_info.get("method")
    result["market_regime_reason"] = regime_info.get("reason")
    if regime_note:
        result["regime_rerank_note"] = regime_note
    # P1 (2026-09-12) - verify khớp số liệu (xem module comment lớn phía
    # trên verify_numeric_claims()). Dùng `chunks` gốc (có chunk_text), KHÔNG
    # phải result["sources"] đã rút gọn.
    result["numeric_verification"] = verify_numeric_claims(result["answer"], chunks)
    return result


def format_answer(result: dict) -> str:
    """CLI/demo-friendly rendering - matches dag_pipeline.py's plain-string
    style so both systems' outputs read consistently side by side.

    2026-09-08: source-list/tag labels now follow result.get("answer_lang")
    ("en" -> English labels, anything else -> the original Vietnamese labels
    unchanged) - see answer_with_citation()/web_fallback_answer()/answer()
    for where "answer_lang" is set via detect_language(). A result without
    "answer_lang" (e.g. hand-built dicts in older tests) falls back to
    Vietnamese, the pre-existing behavior."""
    lines = []
    is_en = result.get("answer_lang") == "en"
    if result.get("source_tier") == "web_fallback":
        if is_en:
            lines.append(
                "=" * 78 + "\n"
                "WARNING: ANSWER FROM PRESS/WEB SOURCES - NOT AN OFFICIAL DISCLOSURE DOCUMENT.\n"
                "disclosure_chunks (financial statements/annual reports...) had no information "
                "for this question.\n" + "=" * 78
            )
        else:
            lines.append(
                "=" * 78 + "\n"
                "CẢNH BÁO: TRẢ LỜI TỪ NGUỒN BÁO CHÍ/WEB - KHÔNG PHẢI TÀI LIỆU CÔNG BỐ CHÍNH THỨC.\n"
                "disclosure_chunks (BCTC/báo cáo thường niên...) không có thông tin cho câu hỏi này.\n"
                + "=" * 78
            )
    lines.append(result["answer"])
    if result["sources"]:
        lines.append("\nSources:" if is_en else "\nNguồn:")
        for i, s in enumerate(result["sources"], start=1):
            if result.get("source_tier") == "web_fallback":
                if is_en:
                    date_part = f" - date {s['published_date']}" if s.get("published_date") else ""
                else:
                    date_part = f" - ngày {s['published_date']}" if s.get("published_date") else ""
                lines.append(
                    f"  [{i}] {s['domain']}{date_part} - {s['source_url']} "
                    f"(similarity={s['similarity']})"
                )
                continue
            if is_en:
                year_part = f" - year {s['year']}" if s.get("year") else ""
                lines.append(
                    f"  [{i}] {s['ticker']} - {s['doctype']} - {s['source_file']} "
                    f"page {s['page']}{year_part} (similarity={s['similarity']})"
                )
            else:
                year_part = f" - năm {s['year']}" if s.get("year") else ""
                lines.append(
                    f"  [{i}] {s['ticker']} - {s['doctype']} - {s['source_file']} "
                    f"trang {s['page']}{year_part} (similarity={s['similarity']})"
                )
    if result.get("temporal_note"):
        lines.append(f"\n[RQ4 - lọc theo thời gian: {result['temporal_note']}]")
    if result.get("doctype_filter"):
        lines.append(f"\n[FinAgentBench - lọc theo loại tài liệu: {result['doctype_filter']}]")
    if result.get("market_regime"):
        tag = f"\n[RQ3 - chu kỳ thị trường: {result['market_regime']}]"
        if result.get("regime_rerank_note"):
            tag += f" ({result['regime_rerank_note']})"
        lines.append(tag)
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    ticker_filter = None
    if args and args[0] == "--ticker":
        ticker_filter = args[1]
        args = args[2:]

    q = " ".join(args) or "Rủi ro pháp lý của công ty là gì?"
    print(f"Q: {q}" + (f" (ticker={ticker_filter})" if ticker_filter else ""))
    res = answer(q, ticker=ticker_filter)
    print(f"A:\n{format_answer(res)}")
    print(f"\n(elapsed: {res['elapsed_s']}s)")
