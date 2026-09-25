"""Lightweight interactive CLI for hand-labeling REAL disclosure chunks into
`data/eval/absa_annotations.jsonl` (see src/absa/annotation_schema.py for the
schema/validation this writes against). Task #6 from the 2026-07-29
architecture review - infrastructure for building the ABSA labeled corpus
src/absa/absa_pipeline.py's RQ5 deliberately does NOT have (zero-shot LLM by
design, see that module's docstring).

This script does NOT invent labels - it pulls real chunk text from
disclosure_chunks (via src/retrieval/vector_store.similarity_search() or a
plain ticker-scoped fetch) and asks a HUMAN, interactively, to assign
aspect/sentiment/evidence for each chunk. Every write goes through
src/absa/annotation_schema.py's append_annotation(), which validates against
the SAME 5-aspect/5-sentiment taxonomy absa_pipeline.py's zero-shot LLM call
uses - so a labeled example and a live RQ5 prediction are always directly
comparable.

**Always-5-aspects prompting (revised 2026-07-29 evening, Task #12 - see
annotation_schema.py's module docstring for the full rationale): this CLI
now asks the annotator about EVERY one of the 5 aspects for EVERY chunk,
one at a time (1=positive, 2=negative, 3=neutral, 4=mixed, 5=not_mentioned)
- it no longer offers a "pick which aspects apply" menu that let the rest
be implicitly skipped. This is more clicks per chunk, but is exactly the
point: "governance not discussed here" is now written down as a real,
explicit not_mentioned label instead of silently vanishing from the
corpus.**

Resume-safe: already-annotated (ticker, source_file, page, chunk_index) keys
are skipped (see annotation_schema.already_annotated_keys()) - interrupting
a labeling session with Ctrl+C loses nothing already written (append-only
writes, same crash-safety convention as every other incremental JSONL
writer in this project) and picks up where it left off on the next run.

Usage (run on your own machine - needs a live Postgres connection, this
cannot be exercised in the sandbox that wrote it):
    python3 -m scripts.absa_annotate_cli --ticker VNM --limit 20
    python3 -m scripts.absa_annotate_cli --ticker VNM --doctype governance_report --annotator-id quang
"""
import argparse

from src.absa.absa_pipeline import ASPECTS, VALID_SENTIMENTS
from src.absa.annotation_schema import (
    ANNOTATIONS_PATH,
    already_annotated_keys,
    append_annotation,
    compute_chunk_hash,
    load_annotations,
    now_iso,
)

_ASPECT_MENU = list(ASPECTS.items())  # [(key, label), ...] - stable order, all 5 always prompted

# 1-5, in this fixed order for every aspect prompt - "not_mentioned" IS a
# normal menu option now (unlike the pre-2026-07-29-evening version, where
# leaving an aspect unselected was how not_mentioned got expressed - see
# annotation_schema.py's docstring for why that design was reversed).
_SENTIMENT_MENU = {
    "1": "positive",
    "2": "negative",
    "3": "neutral",
    "4": "mixed",
    "5": "not_mentioned",
}
_SENTIMENT_MENU_HELP = "1=positive, 2=negative, 3=neutral, 4=mixed, 5=not_mentioned"

DEFAULT_ANNOTATION_GUIDELINE_VERSION = "v1"

FETCH_CANDIDATE_CHUNKS_SQL = """
    SELECT ticker, doctype, source_file, page, chunk_index, chunk_text
    FROM disclosure_chunks
    WHERE embedding IS NOT NULL
      AND (%(ticker)s::text IS NULL OR ticker = %(ticker)s)
      AND (%(doctype)s::text IS NULL OR doctype = %(doctype)s)
    ORDER BY ticker, source_file, page, chunk_index
    LIMIT %(limit)s
"""


def fetch_candidate_chunks(ticker: str = None, doctype: str = None, limit: int = 50) -> list:
    """Plain ticker/doctype-scoped fetch from disclosure_chunks - NOT
    similarity_search(), since there's no query to rank against here; we
    want a representative sample of real chunk text to label, not a
    top-k-by-relevance result. Deliberately a thin, standalone query
    (mirrors scripts/pdf_ingest_router.py's own stated preference for
    dependency-light standalone scripts over importing vector_store.py's
    heavier machinery for something this simple)."""
    from src.retrieval.vector_store import get_connection
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(FETCH_CANDIDATE_CHUNKS_SQL, {"ticker": ticker, "doctype": doctype, "limit": limit})
            rows = cur.fetchall()
    finally:
        conn.close()
    cols = ["ticker", "doctype", "source_file", "page", "chunk_index", "chunk_text"]
    return [dict(zip(cols, row)) for row in rows]


def load_candidate_chunks_from_file(path, double_annotate_only: bool = False) -> list:
    """Loads a FROZEN candidate list written by
    scripts/absa/sample_pilot_paragraphs.py (Task #22) instead of a live
    ticker/doctype query - used for the pilot's inter-annotator agreement
    step, where TWO people must label the exact SAME set of chunks (a live
    query re-run later, even with identical arguments, is not guaranteed to
    return the same rows if disclosure_chunks has changed in between).
    Each line is a JSON object with at least ticker/doctype/source_file/
    page/chunk_index/chunk_text, matching fetch_candidate_chunks()'s
    per-chunk dict shape, plus a "double_annotate" boolean flag."""
    import json
    from pathlib import Path as _Path

    p = _Path(path)
    chunks = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if double_annotate_only and not row.get("double_annotate"):
                continue
            chunks.append(row)
    return chunks


def parse_sentiment_choice(raw: str) -> str:
    """1-5 -> positive/negative/neutral/mixed/not_mentioned. Unlike the
    pre-2026-07-29-evening version, "not_mentioned" IS a valid menu answer
    here now - every aspect is prompted for every chunk (see this module's
    docstring), so the annotator must be able to say "no" explicitly."""
    raw = raw.strip()
    if raw not in _SENTIMENT_MENU:
        raise ValueError(f"'{raw}' không hợp lệ - chọn {_SENTIMENT_MENU_HELP}.")
    return _SENTIMENT_MENU[raw]


def collect_aspects_for_chunk(chunk_text: str, input_fn=input) -> list:
    """Prompts for all 5 ASPECTS, one at a time, and returns a fully-
    populated aspects list (always length 5, one entry per ASPECTS key) -
    or None if the annotator asked to skip this chunk entirely ('s'/'q' at
    any prompt). Factored out from annotate_session() so it's unit-testable
    by injecting a fake input_fn instead of needing a real terminal."""
    aspects = []
    for aspect_key, label in _ASPECT_MENU:
        while True:
            raw = input_fn(f"  [{label}] Sentiment ({_SENTIMENT_MENU_HELP}, 's'=bỏ qua đoạn trích này): ")
            raw = raw.strip()
            if raw.lower() in ("s", "q"):
                return None
            try:
                sentiment = parse_sentiment_choice(raw)
            except ValueError as e:
                print(f"    {e}")
                continue
            break

        entry = {"aspect": aspect_key, "sentiment": sentiment}
        if sentiment == "not_mentioned":
            entry["evidence_span"] = None
        else:
            while True:
                evidence = input_fn(f"    Trích dẫn ngắn làm bằng chứng cho '{label}': ").strip()
                if evidence:
                    break
                print("    evidence_span là bắt buộc trừ khi sentiment='not_mentioned'.")
            entry["evidence_span"] = evidence
        aspects.append(entry)
    return aspects


def build_record(chunk: dict, aspects: list, annotator_id: str, label_source: str, split: str,
                  annotation_guideline_version: str, annotation_round: int, next_id: int) -> dict:
    """Assembles one annotation record dict from a chunk + the (always
    length-5) aspects list collect_aspects_for_chunk() returned - factored
    out for unit testing without needing to simulate a full terminal
    session."""
    return {
        "id": f"absa_{next_id:04d}",
        "ticker": chunk["ticker"],
        "source_file": chunk["source_file"],
        "page": chunk["page"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk["chunk_text"],
        "chunk_hash": compute_chunk_hash(chunk["chunk_text"]),
        "aspects": aspects,
        "annotator_id": annotator_id,
        "label_source": label_source,
        "split": split,
        "annotation_guideline_version": annotation_guideline_version,
        "annotation_round": annotation_round,
        "created_at": now_iso(),
        "notes": "",
    }


def annotate_session(ticker: str, doctype: str, limit: int, annotator_id: str, label_source: str, split: str,
                      annotation_guideline_version: str, annotation_round: int, from_file: str = None,
                      double_annotate_only: bool = False) -> int:
    """The interactive loop. Returns the number of NEW annotations written
    this session. Pulled into its own function (rather than living in
    main()) so a future test could monkeypatch fetch_candidate_chunks()
    and builtins.input to drive it end-to-end without a real terminal/DB -
    not done here since this file's whole purpose is a REAL human's
    judgment, but the seam exists for that.

    from_file (Task #22): if given, chunks come from a FROZEN candidate
    file (scripts/absa/sample_pilot_paragraphs.py's output) instead of a
    live ticker/doctype query - see load_candidate_chunks_from_file().

    done_keys is now scoped to THIS annotator_id (see annotation_schema.
    already_annotated_keys()'s 2026-07-30 update) - a chunk another
    annotator already labeled is NOT skipped here, which is exactly what
    makes double-annotation possible: two different --annotator-id runs
    against the same --from-file both get prompted for every chunk."""
    if from_file:
        chunks = load_candidate_chunks_from_file(from_file, double_annotate_only=double_annotate_only)
    else:
        chunks = fetch_candidate_chunks(ticker=ticker, doctype=doctype, limit=limit)
    done_keys = already_annotated_keys(annotator_id=annotator_id)
    existing = load_annotations()
    next_id = len(existing) + 1

    n_new = 0
    for chunk in chunks:
        key = (chunk["ticker"], chunk["source_file"], chunk["page"], chunk["chunk_index"])
        if key in done_keys:
            continue

        print("\n" + "=" * 70)
        print(f"[{chunk['ticker']}] {chunk['source_file']} - trang {chunk['page']} (doctype={chunk['doctype']})")
        print("-" * 70)
        print(chunk["chunk_text"][:1500])
        print("-" * 70)
        print("Đánh giá LẦN LƯỢT cả 5 khía cạnh (nhập 's' bất kỳ lúc nào để bỏ qua đoạn trích này):")

        try:
            aspects = collect_aspects_for_chunk(chunk["chunk_text"])
        except EOFError:
            break
        if aspects is None:
            print("  Bỏ qua đoạn trích này.")
            continue

        record = build_record(chunk, aspects, annotator_id, label_source, split,
                               annotation_guideline_version, annotation_round, next_id)
        append_annotation(record)
        print(f"  Đã ghi {record['id']} vào {ANNOTATIONS_PATH}")
        next_id += 1
        n_new += 1

    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None, help="Giới hạn theo 1 mã (mặc định: tất cả).")
    ap.add_argument("--doctype", default=None, help="Giới hạn theo doctype (mặc định: tất cả).")
    ap.add_argument("--limit", type=int, default=50, help="Số đoạn trích tối đa để duyệt qua trong phiên này.")
    ap.add_argument("--annotator-id", default="quang", help="Tên/id người gán nhãn (ghi vào mỗi record).")
    ap.add_argument("--label-source", default="human", choices=["human", "llm_assisted_reviewed"],
                     help="'human' (mặc định) nếu bạn tự đọc và gán nhãn; 'llm_assisted_reviewed' nếu bạn đang "
                          "xem lại/sửa gợi ý từ absa_pipeline.analyze_text() thay vì gán từ đầu. "
                          "'llm_unreviewed' KHÔNG có ở đây - đó là nhãn thô LLM chưa qua review của con người, "
                          "phải được tạo qua đường khác (không phải phiên gán nhãn tương tác này) và chỉ được "
                          "split='train' - xem annotation_schema.py.")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"],
                     help="Tập dữ liệu (train/dev/test - mặc định 'dev'). Gán nhãn thật của con người thường "
                          "nên vào dev/test; dùng 'train' nếu bạn đang tạo dữ liệu cho một thử nghiệm fine-tune sau này.")
    ap.add_argument("--annotation-guideline-version", default=DEFAULT_ANNOTATION_GUIDELINE_VERSION,
                     help=f"Phiên bản hướng dẫn gán nhãn đang dùng (mặc định '{DEFAULT_ANNOTATION_GUIDELINE_VERSION}') - "
                          "tăng lên mỗi khi hướng dẫn/taxonomy thay đổi.")
    ap.add_argument("--annotation-round", type=int, default=1,
                     help="Số nguyên >= 1 (mặc định 1) - tăng lên khi đây là một lượt GÁN LẠI có chủ đích "
                          "(vd: adjudication sau khi 2 người gán khác nhau), không phải lượt gán đầu tiên.")
    ap.add_argument("--from-file", default=None,
                     help="Đường dẫn tới file candidate ĐÓNG BĂNG từ "
                          "scripts/absa/sample_pilot_paragraphs.py, thay vì query trực tiếp disclosure_chunks. "
                          "BẮT BUỘC dùng cờ này (cùng 1 file) cho bước double-annotation (Task #22) - 2 người "
                          "PHẢI thấy đúng cùng 1 tập đoạn trích, không phải 2 lần query --ticker/--doctype riêng "
                          "(không đảm bảo trả về cùng tập dòng nếu disclosure_chunks đã thay đổi ở giữa).")
    ap.add_argument("--double-annotate-only", action="store_true",
                     help="Cùng với --from-file: chỉ duyệt các đoạn trích được đánh dấu double_annotate=true "
                          "trong candidate file (tập con ~30-50 đoạn cần 2 người gán độc lập để tính kappa).")
    args = ap.parse_args()

    n_new = annotate_session(args.ticker, args.doctype, args.limit, args.annotator_id, args.label_source,
                              args.split, args.annotation_guideline_version, args.annotation_round,
                              from_file=args.from_file, double_annotate_only=args.double_annotate_only)
    print(f"\nPhiên gán nhãn kết thúc: {n_new} annotation mới đã ghi vào {ANNOTATIONS_PATH}.")


if __name__ == "__main__":
    main()
