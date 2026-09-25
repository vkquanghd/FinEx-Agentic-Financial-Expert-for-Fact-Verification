"""Quick manual VI/EN smoke test WITH a live-ticking timer on screen - added
2026-09-08 in direct response to "test với câu hỏi trực diện, phải có bộ đếm
thời gian" (test with straightforward questions, need a visible time
counter). Distinct from scripts/bilingual_latency_benchmark.py (that one
runs a slim retrieval+generation replica and reports AGGREGATE stats across
many pairs, no live display) - this script:
  - calls the REAL production entry point, src.router.query_router.route()
    (same call the Streamlit "Router" tab makes), not a slim replica, so
    what you see here is exactly what the live demo will show.
  - prints a LIVE, ticking "⏱ X.Xs" counter on screen WHILE Ollama is still
    generating (a background thread runs route(); the main thread redraws
    the elapsed time every 0.2s), not just a final number after the fact -
    useful for a live demo/presentation where the audience should visibly
    see time passing, and for a first sanity check before running the full
    aggregate benchmark.
  - uses a SMALL, fixed set of "câu hỏi trực diện" (straightforward, direct
    questions) on the 4 "mã chính" (main tickers) - AAV/KTS/AAH/ACV, chosen
    2026-09-07 for the deep-dive and confirmed 2026-09-08 (user request) to
    be the focus of every bilingual test/demo asset in this project, ahead
    of the wider "đủ dữ liệu"/TiniX groups covered by the full aggregate
    benchmark (scripts/bilingual_latency_benchmark.py) - so a full run here
    finishes in well under a couple of minutes of wall-clock waiting.

Requires a live Postgres + Ollama stack (same prerequisite as
rag_pipeline.py itself) - NOT runnable in a sandbox with no such stack.

Usage:
    python3 -m scripts.quick_bilingual_test
    python3 -m scripts.quick_bilingual_test --tick 0.5   # slower on-screen refresh
"""
import argparse
import statistics
import threading
import time

# Câu hỏi trực diện (straightforward, direct) trên 4 MÃ CHÍNH - mỗi cặp là
# CÙNG nội dung ở 2 ngôn ngữ, chỉ đổi ngôn ngữ để so sánh công bằng (giống
# bilingual_latency_pairs.jsonl, cùng group "4_ma_chinh"). Công ty/lĩnh vực
# đã tra cứu thật 2026-09-07: AAV (Việt Tiên Sơn Địa Ốc - BĐS/xây dựng/thực
# phẩm đông lạnh/du lịch/đông nam dược, HNX), KTS (Đường Kon Tum - sản xuất
# đường mía, HNX), AAH (Hợp Nhất - khai thác than hầm lò, UPCOM), ACV (Cảng
# Hàng không Việt Nam - quản lý 22 cảng hàng không, UPCOM).
QUESTIONS = [
    {"id": "aav_legal_risk", "ticker": "AAV",
     "vi": "Rủi ro pháp lý của AAV là gì?",
     "en": "What are AAV's legal risks?"},
    {"id": "kts_business_overview", "ticker": "KTS",
     "vi": "Hoạt động kinh doanh chính của KTS là gì?",
     "en": "What is KTS's main business activity?"},
    {"id": "aah_legal_risk", "ticker": "AAH",
     "vi": "Rủi ro pháp lý của AAH là gì?",
     "en": "What are AAH's legal risks?"},
    {"id": "acv_business_overview", "ticker": "ACV",
     "vi": "Hoạt động kinh doanh chính của ACV là gì?",
     "en": "What is ACV's main business activity?"},
]


def run_with_live_timer(question: str, tick: float) -> tuple:
    """Runs route(question) in a background thread while the main thread
    redraws an elapsed-time counter on the same terminal line every `tick`
    seconds. Returns (result_dict, wall_clock_seconds).

    2026-09-08: worker() now catches its own exceptions instead of letting
    them die silently in the thread - a real run hit an Ollama 500 mid-batch
    and the ORIGINAL version left `holder` empty, so the main thread's
    `holder["result"]` raised an unrelated KeyError that killed the whole
    script instead of reporting the real error and moving on to the next
    question. `result_dict` is now always a dict - either the real route()
    result, or {"error": "..."} - so callers can check `.get("error")`
    instead of the caller crashing outright."""
    from src.router.query_router import route

    holder = {}

    def worker():
        try:
            holder["result"] = route(question)
        except Exception as e:
            holder["result"] = {"error": f"{type(e).__name__}: {e}", "answer": None,
                                 "branch": None, "intent_reason": None, "fallback_from": None}

    t0 = time.monotonic()
    th = threading.Thread(target=worker)
    th.start()
    while th.is_alive():
        elapsed = time.monotonic() - t0
        print(f"\r  ⏱  {elapsed:5.1f}s - đang xử lý / processing...", end="", flush=True)
        time.sleep(tick)
    th.join()
    elapsed_total = time.monotonic() - t0
    print(f"\r  ⏱  {elapsed_total:5.1f}s - xong / done.                    ")
    return holder.get("result", {"error": "worker thread produced no result (unexpected)"}), elapsed_total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tick", type=float, default=0.2, help="on-screen refresh interval, seconds")
    args = ap.parse_args()

    from src.core.lang_detect import detect_language

    rows = []
    for item in QUESTIONS:
        for lang_key in ("vi", "en"):
            q = item[lang_key]
            print(f"\n{'=' * 78}")
            print(f"[{item['id']}/{lang_key}] {q}")
            print("=" * 78)
            result, wall_s = run_with_live_timer(q, args.tick)
            detected = detect_language(q)
            if result.get("error"):
                # 2026-09-08 - route() itself failed (e.g. Ollama 500 mid-batch,
                # see run_with_live_timer()'s docstring) - report it and move on
                # to the next question instead of crashing the whole script.
                print(f"LỖI / ERROR: {result['error']}")
                print(f">>> THỜI GIAN / TIME (đến khi lỗi / until failure): {wall_s:.2f}s wall-clock "
                      f"(ngôn ngữ phát hiện / detected language: {detected.upper()})")
                rows.append({"id": item["id"], "lang": lang_key, "wall_s": wall_s, "branch": None, "error": result["error"]})
                continue
            print(f"Nhánh / branch: {result['branch']}"
                  + (f" (fallback từ {result['fallback_from']})" if result.get("fallback_from") else ""))
            print(f"Lý do / reason: {result['intent_reason']}")
            print(f"Trả lời / answer:\n{result['answer']}")
            print(f"\n>>> THỜI GIAN / TIME: {wall_s:.2f}s wall-clock "
                  f"(ngôn ngữ phát hiện / detected language: {detected.upper()})")
            rows.append({"id": item["id"], "lang": lang_key, "wall_s": wall_s, "branch": result["branch"]})

    print("\n" + "=" * 78)
    print("TÓM TẮT / SUMMARY")
    print("=" * 78)
    for lang in ("vi", "en"):
        times = [r["wall_s"] for r in rows if r["lang"] == lang]
        if times:
            print(f"  {lang.upper():4s}  n={len(times)}  mean={statistics.mean(times):.2f}s  "
                  f"median={statistics.median(times):.2f}s  min={min(times):.2f}s  max={max(times):.2f}s")
    vi_mean = statistics.mean([r["wall_s"] for r in rows if r["lang"] == "vi"]) if any(r["lang"] == "vi" for r in rows) else None
    en_mean = statistics.mean([r["wall_s"] for r in rows if r["lang"] == "en"]) if any(r["lang"] == "en" for r in rows) else None
    if vi_mean and en_mean:
        delta_pct = (en_mean - vi_mean) / vi_mean * 100
        print(f"\nEnglish trung bình {delta_pct:+.1f}% so với Vietnamese trên cùng bộ câu hỏi trực diện.")
    print("\n(Muốn số liệu đầy đủ hơn - token/latency chi tiết trên nhiều câu hỏi hơn - "
          "xem scripts/bilingual_latency_benchmark.py.)")


if __name__ == "__main__":
    main()
