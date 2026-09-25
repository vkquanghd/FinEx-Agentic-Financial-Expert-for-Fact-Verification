"""Verify every ticker in the 3 groups identified 2026-09-07 (4 mã chính /
đủ dữ liệu / TiniX-xác nhận) is actually a KNOWN ticker in the live
`companies` table - added 2026-09-10 after a real run_eval.py run showed
VCB (listed in the "đủ dữ liệu" group) fails routing to "dag" because
is_known_ticker("VCB") is False: VCB was deliberately dropped from CORE_16
in favor of MWG/PNJ/DGW/FRT (scripts/seed_companies.py's docstring - banks
use a different financial-statement template, out of this project's DAG
schema scope), so it was never seeded, despite showing up with SOME data in
whatever coverage query originally produced these ticker groups (that query
apparently didn't require a `companies` row to exist).

Run this ONCE before trusting any of these 3 groups for a demo/benchmark run
- catches every VCB-shaped gap in one query instead of discovering them one
at a time through expensive full run_eval.py passes (each RAG/whatif item
costs a real LLM call, tens of seconds; a routing-only miss is cheap to spot
but still wastes a slide/demo moment if found live).

Requires a live Postgres connection (same prerequisite as any other script
in scripts/) - NOT runnable in a sandbox with no such stack.

Usage:
    python3 -m scripts.verify_ticker_groups
"""
from src.structured.company_lookup import is_known_ticker

# Kept in sync BY HAND with data/eval/bilingual_latency_pairs.jsonl's
# "group" field and docs/eval_methodology_reference.md's 3-group table -
# not read from those files directly, since this script's whole point is to
# be a fast, standalone sanity check runnable before trusting any of them.
GROUPS = {
    "4_ma_chinh": ["AAV", "KTS", "AAH", "ACV"],
    "du_lieu_du": ["AAA", "VCB", "CTD", "KTS", "AAV", "AAH", "AGM", "AAS", "AAT", "CMC"],
    "tinix_xac_nhan": ["AAV", "KTS", "SLS", "AAH", "ACV"],
}


def main():
    all_tickers = sorted({t for tickers in GROUPS.values() for t in tickers})
    print(f"Checking {len(all_tickers)} distinct ticker(s) across 3 groups against "
          f"the live `companies` table (is_known_ticker())...\n")

    results = {}
    for t in all_tickers:
        try:
            results[t] = is_known_ticker(t)
        except Exception as e:
            results[t] = f"ERROR: {type(e).__name__}: {e}"

    bad = {t: r for t, r in results.items() if r is not True}
    good = {t: r for t, r in results.items() if r is True}

    print(f"OK ({len(good)}): {', '.join(sorted(good))}\n")
    if bad:
        print(f"KHÔNG PHẢI ticker hợp lệ / lỗi khi kiểm tra ({len(bad)}):")
        for t, r in sorted(bad.items()):
            print(f"  {t}: {r}")
        print("\nCác mã này sẽ không route vào DAG/whatif (thiếu 'ticker' trong "
              "intent) - loại khỏi câu hỏi test cần tra cứu theo ticker, hoặc "
              "kiểm tra lại scripts/seed_companies.py xem có nên seed thêm không "
              "(chỉ nếu KHÔNG phải ngân hàng/tổ chức tín dụng dùng mẫu BCTC khác).")
    else:
        print("Tất cả tickers đều hợp lệ - an toàn dùng cho benchmark/demo.")

    print("\n--- Theo từng nhóm ---")
    for group_name, tickers in GROUPS.items():
        bad_in_group = [t for t in tickers if results.get(t) is not True]
        status = "OK" if not bad_in_group else f"CÓ VẤN ĐỀ: {bad_in_group}"
        print(f"  {group_name} ({len(tickers)} mã): {status}")


if __name__ == "__main__":
    main()
