#!/usr/bin/env python3
"""Scrape + chunk web/press content into data/raw/web_articles/ and
data/processed/web_chunks.jsonl, feeding the SEPARATE `web_chunks` pgvector
table (infra/postgres/migrations/014_web_chunks.sql) - NOT disclosure_chunks.

Built 2026-09-07 while diagnosing why "Rủi ro pháp lý của VNM là gì?" stayed
thin even after the missing Annual Report PDFs were ingested: VNM's OWN
official risk-management disclosure genuinely IS that thin (checked against
the real PDF - see docs/PROGRESS.md same date), so no amount of re-ingesting
OFFICIAL disclosures fixes this class of question. Press coverage is a
different evidence tier (can be wrong/retracted/opinion, unlike an audited
BCTC) - see 014_web_chunks.sql's header comment for why that's a SEPARATE
table, not more disclosure_chunks.doctype values.

Reuses web_search()/fetch_page_text() from scripts/web_fallback_lookup.py
(same `ddgs` dependency, same optional-install story) rather than
duplicating them - this script's job is different (PERSIST a snapshot into
the corpus) but the underlying search/fetch mechanics are identical.

Two-stage pipeline (mirrors pdf_ingest_router.py -> text_splitter.py's
separation, collapsed to one script since web content has no OCR-routing
step to separate out):

  1. `scrape`: search -> fetch full page text -> save ONE raw JSON snapshot
     per URL to data/raw/web_articles/<sha256(url)[:16]>.json. Idempotent -
     re-running with overlapping queries skips URLs already snapshotted
     (use --force-refetch to re-fetch anyway, e.g. if a page was updated).
  2. `chunk`: read every snapshot in data/raw/web_articles/, chunk_text()
     each (reusing src/retrieval/text_splitter.py's chunker/chunk_hash -
     same CHUNK_SIZE/CHUNK_OVERLAP, same content-hash convention), write
     data/processed/web_chunks.jsonl. Safe to re-run any time.

Embedding into pgvector is a SEPARATE step (src/retrieval/vector_store.py's
upsert_web_records(), same file the disclosure pipeline uses) - kept
consistent with the disclosure pipeline's "chunking and embedding are
independently re-runnable steps" principle.

Usage:
    python3 scripts/web_news_ingest.py scrape --ticker VNM --query "rủi ro pháp lý"
    python3 scripts/web_news_ingest.py scrape --ticker VNM --query "rủi ro pháp lý" --sites cafef.vn dantri.com.vn
    python3 scripts/web_news_ingest.py chunk
    python3 -m src.retrieval.vector_store --web --input data/processed/web_chunks.jsonl
"""
import argparse
import hashlib
import json
import sys
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# Support the documented `python3 scripts/...` commands from the project
# root. Python otherwise places only `scripts/` on sys.path, which prevents
# imports of both `src` and sibling modules under `scripts`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Financial/regulatory sources (same as web_fallback_lookup.py's DEFAULT_SITES)
# PLUS general press (2026-09-07 decision - see the AskUserQuestion answer in
# docs/PROGRESS.md same date: "Thêm cả báo chí phổ thông" was chosen so a
# story like the VNM legal-risk question, which official filings don't cover
# in detail, actually has a chance of being found). Mainstream, editorially-
# reviewed outlets only - deliberately NOT an unrestricted web search (see
# that same decision) to keep the false-information rate manageable without
# per-article human review.
DEFAULT_SITES = [
    "cafef.vn", "vietstock.vn", "hnx.vn", "hsx.vn", "ssc.gov.vn",
    "dantri.com.vn", "tuoitre.vn", "vnexpress.net",
]

RAW_DIR_DEFAULT = "data/raw/web_articles"
CHUNKS_OUT_DEFAULT = "data/processed/web_chunks.jsonl"

# Full-article fetch, not the 4000-char snippet-sized default
# web_fallback_lookup.fetch_page_text() uses for its single top-1-result
# LLM-context case - here every result is meant to be READ, not just
# summarized, so a real news article's full body should fit.
FETCH_MAX_CHARS = 12000

# Question words do not establish that a result is relevant. The remaining
# terms are used as a conservative lexical guard after search/fetch so pages
# that only mention a ticker in their navigation are not embedded as evidence.
QUERY_STOP_WORDS = {
    "about", "after", "are", "and", "cho", "cua", "does", "for", "from", "gi", "is",
    "la", "latest", "most", "near", "nhung", "recent", "the", "to", "ve", "what", "with",
}


def _url_key(url: str) -> str:
    """Stable filename for a URL's raw snapshot - sha256 (not the URL text
    itself) so path length/characters are never an issue across OSes."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _domain_of(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


def _search_normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_marks.replace("đ", "d").replace("Đ", "D").lower()


def is_relevant_result(ticker: str, query: str, title: str, snippet: str, text: str) -> bool:
    """Require both the requested ticker and at least one material query
    term. This deliberately favors an honest empty corpus over injecting a
    generic article into a company-specific financial answer."""
    corpus = _search_normalize(" ".join((title, snippet, text)))
    normalized_ticker = _search_normalize(ticker or "").strip()
    if normalized_ticker and not re.search(rf"\b{re.escape(normalized_ticker)}\b", corpus):
        return False

    terms = [
        term for term in re.findall(r"[a-z0-9]+", _search_normalize(query))
        if len(term) >= 3 and term not in QUERY_STOP_WORDS and term != normalized_ticker
    ]
    return not terms or any(re.search(rf"\b{re.escape(term)}", corpus) for term in terms)


def has_material_query_term(text: str, query: str, ticker: str = None) -> bool:
    """Check the fetched body itself, not its result title. Search-result
    pages sometimes contain a correct title but return only navigation HTML;
    those bodies are replaced with the result snippet before persistence."""
    corpus = _search_normalize(text)
    normalized_ticker = _search_normalize(ticker or "").strip()
    terms = [
        term for term in re.findall(r"[a-z0-9]+", _search_normalize(query))
        if len(term) >= 3 and term not in QUERY_STOP_WORDS and term != normalized_ticker
    ]
    return bool(terms) and any(re.search(rf"\b{re.escape(term)}", corpus) for term in terms)


def scrape(ticker: str, query: str, sites: list, max_results: int,
           raw_dir: Path, force_refetch: bool = False) -> dict:
    """Search + fetch, writing one raw JSON snapshot per NEW url to
    raw_dir. Returns a summary dict. Skips urls that already have a
    snapshot file unless force_refetch=True (same "don't hammer a source
    that's already saved" courtesy scripts/pdf_ingest_router.py's
    incremental design already follows for PDFs)."""
    from scripts.web_fallback_lookup import fetch_page_text, web_search

    full_query = f"{ticker} {query}" if ticker else query
    results = web_search(full_query, sites=sites, max_results=max_results)
    raw_dir.mkdir(parents=True, exist_ok=True)

    n_new, n_skipped, n_empty, n_irrelevant = 0, 0, 0, 0
    for r in results:
        key = _url_key(r["url"])
        snapshot_path = raw_dir / f"{key}.json"
        if snapshot_path.exists() and not force_refetch:
            n_skipped += 1
            continue

        text = fetch_page_text(r["url"], max_chars=FETCH_MAX_CHARS)
        snippet = r.get("snippet", "")
        if not text or not has_material_query_term(text, query, ticker):
            # Fall back to the search snippet rather than storing nothing -
            # a search result can have a relevant title while its fetched
            # HTML is only navigation/login chrome. The snippet is weaker
            # evidence but preferable to embedding that boilerplate.
            text = snippet
        if not text:
            n_empty += 1
            continue

        relevant = is_relevant_result(ticker, query, r.get("title", ""), snippet, text)
        snapshot = {
            "url": r["url"],
            "domain": _domain_of(r["url"]),
            "title": r.get("title", ""),
            "ticker": ticker,
            "query": query,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "eligible_for_retrieval": relevant,
        }
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        if relevant:
            n_new += 1
        else:
            n_irrelevant += 1

    return {
        "n_results": len(results), "n_new": n_new, "n_skipped": n_skipped,
        "n_empty": n_empty, "n_irrelevant": n_irrelevant,
    }


def chunk_snapshots(raw_dir: Path, out_path: Path) -> dict:
    """Reads every *.json snapshot in raw_dir, chunks its text (reusing
    src/retrieval/text_splitter.py's chunk_text()/chunk_hash() - same
    CHUNK_SIZE/CHUNK_OVERLAP/hash convention as the disclosure pipeline, so
    the two corpora are at least comparably-sized chunks even though they
    live in separate tables), writes web_chunks.jsonl. Rewrites the whole
    output file each run (not incremental like text_splitter.py's manifest
    system) - the web corpus is expected to be small enough (dozens to
    low-hundreds of articles, not 650k+ PDF pages) that re-chunking
    everything each run is cheap; vector_store.upsert_web_records()'s own
    skip_existing logic (matching chunk_hash+embedding_version) is what
    actually avoids repeat embedding work, same as the disclosure side."""
    from src.retrieval.text_splitter import chunk_hash, chunk_text

    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_paths = sorted(raw_dir.glob("*.json"))
    n_chunks = 0
    n_articles = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for snap_path in snapshot_paths:
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
            # Legacy snapshots created before the relevance guard are not
            # trusted automatically. Re-scrape them so they are assessed
            # under the current query-specific evidence policy.
            if not snap.get("eligible_for_retrieval", False):
                continue
            text = " ".join(snap["text"].split())  # collapse whitespace, same normalization spirit as text_splitter's normalize_page_text()
            title = " ".join((snap.get("title") or "").split())
            if title:
                text = f"{title}\n{text}"
            pieces = chunk_text(text)
            if not pieces:
                continue
            n_articles += 1
            for idx, piece in enumerate(pieces):
                record = {
                    "ticker": snap.get("ticker"),
                    "source_type": "news_web",
                    "domain": snap["domain"],
                    "source_url": snap["url"],
                    "title": snap.get("title"),
                    "published_date": None,  # best-effort extraction not implemented yet - honest None, not guessed (see NOTE below)
                    "scraped_at": snap["scraped_at"],
                    "chunk_index": idx,
                    "chunk_text": piece,
                    "char_count": len(piece),
                    "chunk_hash": chunk_hash(piece),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1
    return {"n_snapshots": len(snapshot_paths), "n_articles_with_chunks": n_articles, "n_chunks": n_chunks}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_scrape = sub.add_parser("scrape", help="Search + fetch + save raw snapshots")
    p_scrape.add_argument("--ticker", default=None)
    p_scrape.add_argument("--query", required=True)
    p_scrape.add_argument("--sites", nargs="*", default=None, help=f"Mặc định: {DEFAULT_SITES}")
    p_scrape.add_argument("--max-results", type=int, default=8)
    p_scrape.add_argument("--raw-dir", default=RAW_DIR_DEFAULT)
    p_scrape.add_argument("--force-refetch", action="store_true",
                           help="Re-fetch urls even if a snapshot already exists")

    p_chunk = sub.add_parser("chunk", help="Chunk all raw snapshots into web_chunks.jsonl")
    p_chunk.add_argument("--raw-dir", default=RAW_DIR_DEFAULT)
    p_chunk.add_argument("--output", default=CHUNKS_OUT_DEFAULT)

    args = ap.parse_args()

    if args.mode == "scrape":
        summary = scrape(
            ticker=args.ticker, query=args.query, sites=args.sites or DEFAULT_SITES,
            max_results=args.max_results, raw_dir=Path(args.raw_dir),
            force_refetch=args.force_refetch,
        )
        print(f"Search trả về {summary['n_results']} kết quả: {summary['n_new']} snapshot mới, "
              f"{summary['n_skipped']} đã có sẵn (bỏ qua), {summary['n_empty']} không lấy được nội dung, "
              f"{summary['n_irrelevant']} không đủ liên quan nên không index.")
    elif args.mode == "chunk":
        summary = chunk_snapshots(Path(args.raw_dir), Path(args.output))
        print(f"Đọc {summary['n_snapshots']} snapshot, chunk được {summary['n_articles_with_chunks']} bài "
              f"thành {summary['n_chunks']} chunk -> {args.output}")


if __name__ == "__main__":
    main()
