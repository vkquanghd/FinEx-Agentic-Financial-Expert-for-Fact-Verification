"""Standalone, human-invoked web lookup - DELIBERATELY OUT OF SCOPE for
RQ1-RQ7 (2026-07-29). Use this when a ticker/period isn't loaded into
`disclosure_chunks` yet and you don't want to wait for Step 9-10 (route/
chunk/embed real PDFs) just to answer one question.

Why this is a separate evidence tier from the reproducible RAG corpus:

- It is not a Router intent. The normal benchmark path does not invoke live
  web search. `src/retrieval/rag_pipeline.py` may use these helpers only
  when an operator explicitly sets `FINEX_LIVE_WEB_FALLBACK=1`; that path
  remains outside `data/eval/benchmark.jsonl` and every RQ1-RQ7 metric.
- NOT scored anywhere - there is no "web_fallback" benchmark category and
  there should never be one: a live web page can change or disappear
  between two runs of the same script, which breaks the reproducibility
  every other branch in this project is held to (see
  `docs/PROGRESS.md`'s "no invented ground truth" principle - a citation
  to a page that might not say the same thing tomorrow is a different kind
  of problem, but the same underlying issue: an answer whose grounding
  cannot be checked later).
- Every answer this script prints carries an explicit warning banner + the
  exact URL(s) it used, specifically so it's never mistaken for a verified
  `rag_faithfulness` answer if copy-pasted into notes/PROGRESS.md by hand.

Optional dependency, NOT in requirements.txt's required set (same
tradeoff already made for causalml/econml above it in that file):
    pip install ddgs
(`ddgs` - the maintained successor to the now-archived `duckduckgo-search`
package. Free, no API key, no account needed.)

Usage:
    python3 scripts/web_fallback_lookup.py --ticker VNM --question "Công ty kiểm toán của VNM năm 2026 là ai?"
    python3 scripts/web_fallback_lookup.py --ticker VNM --question "..." --sites cafef.vn vietstock.vn hnx.vn
    python3 scripts/web_fallback_lookup.py --question "Giá vàng SJC hôm nay bao nhiêu?" --no-fetch
"""
import argparse
from html import unescape
import re
import sys
from pathlib import Path

# Support invocation as `python3 scripts/web_fallback_lookup.py ...` from the
# project root. Running a file by path otherwise omits the project root from
# sys.path, so imports such as `src.models.llm_factory` fail.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WARNING_BANNER = (
    "=" * 78 + "\n"
    "CẢNH BÁO: KẾT QUẢ TỪ WEB SEARCH TRỰC TIẾP - KHÔNG PHẢI RAG ĐÃ VERIFY.\n"
    "Không tái lập được (nguồn có thể đổi/mất theo thời gian). KHÔNG nằm trong\n"
    "data/eval/benchmark.jsonl, KHÔNG được chấm điểm bởi bất kỳ RQ nào (RQ1-RQ7).\n"
    "Chỉ dùng để tham khảo/tiện ích cá nhân - không dán vào PROGRESS.md hay kết\n"
    "quả nghiên cứu chính thức mà không ghi rõ đây là nguồn web, không phải RAG.\n"
    + "=" * 78
)

DEFAULT_SITES = ["cafef.vn", "vietstock.vn", "hnx.vn", "hsx.vn", "ssc.gov.vn"]


def build_search_query(query: str, sites: list = None) -> str:
    """Pure-logic query-string builder, split out from web_search() so it's
    unit-testable without mocking the ddgs network client (see
    tests/test_dialogue/test_web_fallback_lookup.py). `sites` narrows the
    search with `(site:a OR site:b OR ...)` - defaults to the Vietnamese
    financial-disclosure sources discussed 2026-07-29 (CafeF, Vietstock,
    HNX/HSX, the State Securities Commission), since an unscoped web search
    over financial questions is much more likely to surface unreliable
    blogspam than these known sources."""
    if not sites:
        return query
    site_filter = " OR ".join(f"site:{s}" for s in sites)
    return f"{query} ({site_filter})"


def web_search(query: str, sites: list = None, max_results: int = 5) -> list[dict]:
    """Returns a list of {title, url, snippet} dicts via DuckDuckGo (no API
    key)."""
    try:
        from ddgs import DDGS
    except ImportError:
        raise SystemExit(
            "Thiếu thư viện `ddgs`. Cài dependencies của project trong `.venv` bằng "
            "`.venv/bin/python -m pip install -r requirements.txt` trước."
        )

    full_query = build_search_query(query, sites)

    with DDGS() as ddgs:
        results = list(ddgs.text(full_query, region="vn-vi", max_results=max_results))
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results]


def strip_html(html: str, max_chars: int = 4000) -> str:
    """Extract article-like text from HTML, with a regex fallback.

    News pages commonly place their navigation before the article. Keeping
    the entire body made the first retrieval chunks mostly menus, so use
    lxml (already required by ``ddgs``) to prefer article/main/content
    containers and remove boilerplate before flattening the text. The
    fallback keeps this helper usable if lxml is unavailable.
    """
    try:
        from lxml import html as lxml_html

        document = lxml_html.fromstring(html)
        for node in document.xpath("//script|//style|//nav|//header|//footer|//aside|//form|//svg|//noscript"):
            node.drop_tree()

        candidates = []
        for node in document.xpath("//*[self::article or self::main or self::section or self::div]"):
            marker = f"{node.tag} {node.get('id', '')} {node.get('class', '')}".lower()
            if node.tag in {"article", "main"} or any(
                token in marker for token in ("article", "content", "detail", "post", "story", "entry", "body")
            ):
                text = " ".join(node.itertext())
                paragraph_count = len(node.xpath(".//p"))
                link_text = " ".join(node.xpath(".//a//text()"))
                score = len(text) + (paragraph_count * 300) - (len(link_text) * 0.35)
                candidates.append((score, text))

        if candidates:
            text = max(candidates, key=lambda item: item[0])[1]
        else:
            body = document.find(".//body")
            text = " ".join((body if body is not None else document).itertext())
        text = " ".join(unescape(text).split())
        if text:
            return text[:max_chars]
    except (ImportError, ValueError, TypeError):
        pass

    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(unescape(text).split())[:max_chars]


def fetch_page_text(url: str, max_chars: int = 4000, timeout: int = 15) -> str:
    """Best-effort plain-text extraction of a page - returns "" (not an
    exception) on any failure so the caller can fall back to the search
    snippet instead."""
    try:
        import requests
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return strip_html(resp.text, max_chars=max_chars)
    except Exception:
        return ""


def answer_with_web_fallback(question: str, ticker: str = None, sites: list = None,
                              max_results: int = 5, fetch_top_page: bool = True) -> dict:
    """Orchestrates search -> (optional) page fetch -> LLM synthesis with
    forced citation. Reuses the SAME local Qwen3/Ollama model every other
    branch in this project uses (src.models.llm_factory) - no new model,
    no cloud LLM API - only the SEARCH step is new/external here."""
    from src.models.llm_factory import generate

    query = f"{ticker} {question}" if ticker else question
    results = web_search(query, sites=sites or DEFAULT_SITES, max_results=max_results)
    if not results:
        return {"answer": None, "sources": [], "note": "Không tìm thấy kết quả web nào cho câu hỏi này."}

    context_parts = []
    for r in results:
        page_text = fetch_page_text(r["url"]) if fetch_top_page else ""
        body = page_text or r["snippet"]
        context_parts.append(f"Nguồn: {r['url']}\nTiêu đề: {r['title']}\nNội dung: {body}")
        fetch_top_page = False  # only fetch the full page for the FIRST (top) result, rest use snippets only

    context = "\n\n---\n\n".join(context_parts)
    system = (
        "Bạn là trợ lý tra cứu tài chính. CHỈ trả lời dựa trên nội dung được cung cấp bên dưới, "
        "KHÔNG suy đoán/bịa thêm. Nếu nội dung không đủ để trả lời, nói rõ 'không tìm thấy đủ thông tin'. "
        "LUÔN kết thúc câu trả lời bằng dòng 'Nguồn: <url>' ghi đúng URL bạn đã dùng."
    )
    prompt = f"Câu hỏi: {question}\n\nCác kết quả tìm kiếm web:\n\n{context}\n\nTrả lời:"
    answer = generate(prompt, system=system, temperature=0.0)
    return {"answer": answer, "sources": [r["url"] for r in results], "note": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", required=True)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--sites", nargs="*", default=None,
                         help=f"Mặc định: {DEFAULT_SITES}")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--no-fetch", action="store_true",
                         help="Chỉ dùng snippet từ kết quả tìm kiếm, không tải nội dung trang top-1.")
    args = parser.parse_args()

    result = answer_with_web_fallback(
        args.question, ticker=args.ticker, sites=args.sites,
        max_results=args.max_results, fetch_top_page=not args.no_fetch,
    )

    print(WARNING_BANNER)
    if result["note"]:
        print(result["note"])
        sys.exit(1)
    print(f"\nCâu hỏi: {args.question}")
    if args.ticker:
        print(f"Mã: {args.ticker}")
    print(f"\nTrả lời:\n{result['answer']}")
    print(f"\nCác nguồn đã tìm được (không phải chỉ nguồn LLM trích dẫn ở trên):")
    for url in result["sources"]:
        print(f"  - {url}")
    print("\n" + WARNING_BANNER)


if __name__ == "__main__":
    main()
