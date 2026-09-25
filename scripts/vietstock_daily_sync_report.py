#!/usr/bin/env python3
"""In ra bản tóm tắt tiếng Việt, dễ đọc, cho lần chạy vietstock-premium-sync
GẦN NHẤT (bảng vietstock_ingestion_runs / vietstock_ticker_sync /
vietstock_document_manifest - xem infra/postgres/migrations/016_vietstock_
premium_ingestion.sql).

Vì sao cần file này: `scripts/vietstock_premium_sync.py sync` đã in JSON thô
(counters) + hàng trăm dòng tiến trình theo mã vào
data/private/vietstock_daily_sync.log - đủ dữ liệu nhưng không ai muốn đọc
log thô mỗi ngày. Script này đọc THẲNG từ DB (nguồn sự thật, không parse
log) và in ra một bản tóm tắt ngắn.

Cách dùng thủ công:
    python -m scripts.vietstock_daily_sync_report
    python -m scripts.vietstock_daily_sync_report --run-id 42
    python -m scripts.vietstock_daily_sync_report --json   # cho máy đọc

Tự động sau mỗi lần launchd chạy 18:20: xem cuối
scripts/install_vietstock_daily_sync.py - lệnh này được nối thêm vào SAU
lệnh sync trong cùng plist, cùng ghi vào data/private/vietstock_daily_sync.log,
nên log mỗi ngày giờ có sẵn bản tóm tắt ở cuối thay vì chỉ có JSON thô.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from src.core.db import get_session  # noqa: E402

_RUN_SQL = text("""
    SELECT id, scope, status, started_at, finished_at, requested_by, config, counters, error_summary
    FROM vietstock_ingestion_runs
    WHERE (:run_id IS NULL OR id = :run_id)
    ORDER BY started_at DESC
    LIMIT 1
""")

_TICKER_SQL = text("""
    SELECT ticker, status, http_status, document_count, news_count, error_message
    FROM vietstock_ticker_sync
    WHERE run_id = :run_id
    ORDER BY (status <> 'completed') DESC, ticker
""")

# vietstock_document_manifest không có run_id (xem migration 016) - chỉ có
# first_seen_at/last_seen_at, nên "tài liệu mới trong lần chạy này" là suy
# ra theo cửa sổ thời gian của run, KHÔNG phải khóa ngoại chính xác. Nêu rõ
# đây là ước lượng trong output để không trình bày như con số chắc chắn.
_NEW_DOCS_SQL = text("""
    SELECT ticker, source_url, title, document_type, download_status, first_seen_at
    FROM vietstock_document_manifest
    WHERE first_seen_at BETWEEN :started_at AND COALESCE(:finished_at, now())
    ORDER BY first_seen_at DESC
    LIMIT 50
""")

_FAILED_DOCS_SQL = text("""
    SELECT ticker, source_url, download_status, last_error
    FROM vietstock_document_manifest
    WHERE last_seen_at BETWEEN :started_at AND COALESCE(:finished_at, now())
      AND download_status IN ('failed', 'access_denied')
    ORDER BY ticker
    LIMIT 50
""")


def _fmt_dt(dt) -> str:
    return dt.strftime("%H:%M %d/%m/%Y") if dt else "—"


def build_report(run_id: int | None) -> dict:
    with get_session() as session:
        run = session.execute(_RUN_SQL, {"run_id": run_id}).mappings().first()
        if run is None:
            return {"ok": False, "message": "Chưa có lần chạy vietstock-premium-sync nào trong DB."}

        tickers = session.execute(_TICKER_SQL, {"run_id": run["id"]}).mappings().all()
        new_docs = session.execute(_NEW_DOCS_SQL, {
            "started_at": run["started_at"], "finished_at": run["finished_at"],
        }).mappings().all()
        failed_docs = session.execute(_FAILED_DOCS_SQL, {
            "started_at": run["started_at"], "finished_at": run["finished_at"],
        }).mappings().all()

    failed_tickers = [t for t in tickers if t["status"] in ("failed", "partial")]
    counters = dict(run["counters"] or {})

    return {
        "ok": True,
        "run": {
            "id": run["id"], "scope": run["scope"], "status": run["status"],
            "started_at": run["started_at"], "finished_at": run["finished_at"],
            "requested_by": run["requested_by"], "error_summary": run["error_summary"],
        },
        "counters": counters,
        "n_tickers_synced": len(tickers),
        "n_tickers_failed_or_partial": len(failed_tickers),
        "failed_tickers": [
            {"ticker": t["ticker"], "status": t["status"], "http_status": t["http_status"],
             "error": t["error_message"]}
            for t in failed_tickers
        ],
        "new_documents": [dict(d) for d in new_docs],
        "failed_documents": [dict(d) for d in failed_docs],
    }


def render_vietnamese(report: dict) -> str:
    if not report["ok"]:
        return report["message"]

    run = report["run"]
    c = report["counters"]
    lines = []
    lines.append(f"=== Lần chạy vietstock-premium-sync #{run['id']} ({run['scope']}) ===")
    lines.append(f"Bắt đầu: {_fmt_dt(run['started_at'])}  ·  Kết thúc: {_fmt_dt(run['finished_at'])}  ·  Trạng thái: {run['status']}")
    if run["error_summary"]:
        lines.append(f"Lỗi tổng quát: {run['error_summary']}")
    lines.append("")

    if c:
        lines.append(
            f"Mã yêu cầu: {c.get('requested', '—')} → "
            f"xong: {c.get('completed', '—')}, "
            f"một phần: {c.get('partial', '—')}, "
            f"thất bại: {c.get('failed', '—')}"
        )
        lines.append(
            f"Tài liệu: phát hiện {c.get('documents_discovered', '—')}, "
            f"tải về {c.get('documents_downloaded', '—')}  ·  "
            f"Tin tức/sự kiện: {c.get('news_events', '—')}"
        )
    else:
        lines.append("(Lần chạy này chưa có counters - có thể đang chạy dở hoặc bị ngắt giữa chừng.)")
    lines.append("")

    if report["failed_tickers"]:
        lines.append(f"{len(report['failed_tickers'])} mã lỗi/một phần:")
        for t in report["failed_tickers"][:20]:
            lines.append(f"  - {t['ticker']}: {t['status']}"
                          f"{' (HTTP ' + str(t['http_status']) + ')' if t['http_status'] else ''}"
                          f"{' — ' + t['error'] if t['error'] else ''}")
        if len(report["failed_tickers"]) > 20:
            lines.append(f"  ... và {len(report['failed_tickers']) - 20} mã khác.")
    else:
        lines.append("Không có mã nào lỗi hoặc chạy một phần.")
    lines.append("")

    if report["new_documents"]:
        lines.append(f"Tài liệu MỚI phát hiện trong khoảng thời gian lần chạy này "
                      f"(ước lượng theo first_seen_at, không phải khóa ngoại chính xác - "
                      f"xem docstring): {len(report['new_documents'])}")
        for d in report["new_documents"][:15]:
            lines.append(f"  - [{d['ticker']}] {d['document_type'] or '?'}: "
                          f"{(d['title'] or d['source_url'])[:80]} ({d['download_status']})")
        if len(report["new_documents"]) > 15:
            lines.append(f"  ... và {len(report['new_documents']) - 15} tài liệu khác.")
    else:
        lines.append("Không có tài liệu mới nào được phát hiện trong khoảng thời gian này.")
    lines.append("")

    if report["failed_documents"]:
        lines.append(f"{len(report['failed_documents'])} tài liệu tải lỗi/bị từ chối quyền:")
        for d in report["failed_documents"][:15]:
            lines.append(f"  - [{d['ticker']}] {d['download_status']}: {(d['last_error'] or '')[:100]}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=None, help="Mặc định: lần chạy mới nhất")
    parser.add_argument("--json", action="store_true", help="In JSON thay vì bản tóm tắt tiếng Việt")
    args = parser.parse_args()

    try:
        report = build_report(args.run_id)
    except Exception as exc:  # DB có thể chưa sẵn sàng - đừng làm launchd job coi là crash cả pipeline
        print(f"Không tạo được báo cáo (DB lỗi hoặc chưa sẵn sàng): {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, default=str, indent=2))
    else:
        print(render_vietnamese(report))


if __name__ == "__main__":
    main()
