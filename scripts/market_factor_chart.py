"""Static PNG chart export for direct insertion into the thesis document
(Word/PDF) - the same rebased multi-ticker-vs-gold/VN-Index comparison as
app_streamlit.py's "So sánh mã & Vàng" tab, but as a file instead of an
interactive widget. Built for the 2025 gold-price case study
(docs/market_factor_case_study_gold_2025.md) but works for any ticker list.

Usage:
    python -m scripts.market_factor_chart --tickers PNJ,CTD,VNM --with-gold --with-vnindex \
        --events sbv_aml_inspection_selloff,decree_232_gold_monopoly_end \
        --output docs/figures/pnj_ctd_vnm_vs_gold_2025.png

    # Oil overlay (2026-07-19 addition, needs FRED_API_KEY-backed oil data):
    python -m scripts.market_factor_chart --tickers GAS,CTD --with-oil \
        --output docs/figures/gas_ctd_vs_oil_2025.png

NOT run against real data by the environment that wrote it (no MySQL
reachable here, same situation as every other DB-backed script in this
project) - run it for real and report back if the chart looks wrong.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # no display needed - just writes a file
import matplotlib.pyplot as plt

from src.causal.market_factors import (
    KNOWN_2025_GOLD_EVENTS, compare_tickers, gold_price_series, oil_price_series, vnindex_series,
)


def build_chart(tickers: list[str], with_gold: bool, with_vnindex: bool, with_oil: bool,
                 event_keys: list[str], output_path: str):
    cmp_result = compare_tickers(tickers)
    if not cmp_result["ok"]:
        print(f"ERROR: {cmp_result['error']}")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    wide = cmp_result["wide"]
    for t in tickers:
        if t in wide.columns:
            ax.plot(wide["trade_date"], wide[t], label=t, linewidth=1.8)

    if with_gold:
        gold = gold_price_series()
        if not gold.empty:
            rebased = gold["gold_price"] / gold["gold_price"].iloc[0] * 100
            ax.plot(gold["trade_date"], rebased, label="Vàng SJC (rebase)", linewidth=1.8,
                     linestyle="--", color="goldenrod")
        else:
            print("  (with-gold requested but no gold data loaded yet - skipped)")

    if with_vnindex:
        vni = vnindex_series()
        if not vni.empty:
            rebased = vni["vnindex_close"] / vni["vnindex_close"].iloc[0] * 100
            ax.plot(vni["trade_date"], rebased, label="VN-Index (rebase)", linewidth=1.8,
                     linestyle=":", color="dimgray")
        else:
            print("  (with-vnindex requested but no VN-Index data loaded yet - skipped)")

    if with_oil:
        oil = oil_price_series()
        if not oil.empty:
            rebased = oil["oil_price"] / oil["oil_price"].iloc[0] * 100
            ax.plot(oil["trade_date"], rebased, label="Dầu WTI (rebase)", linewidth=1.8,
                     linestyle="--", color="black")
        else:
            print("  (with-oil requested but no oil data loaded yet - needs FRED_API_KEY - skipped)")

    for key in event_keys:
        ev = KNOWN_2025_GOLD_EVENTS.get(key)
        if ev is None:
            print(f"  (unknown event key '{key}' - skipped, see KNOWN_2025_GOLD_EVENTS in market_factors.py)")
            continue
        ax.axvline(ev["date"], color="red", linestyle="-.", linewidth=1, alpha=0.7)
        ax.text(ev["date"], ax.get_ylim()[1], f" {key}", rotation=90, va="top", fontsize=8, color="red")

    ax.set_title("So sánh hiệu suất giá (rebase = 100 tại ngày đầu)")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Chỉ số (rebase = 100)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved: {out} ({len(wide)} trading days, tickers={tickers})")
    if cmp_result["missing"]:
        print(f"  (no data for: {cmp_result['missing']})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, required=True, help="Comma-separated, e.g. PNJ,CTD,VNM")
    ap.add_argument("--with-gold", action="store_true")
    ap.add_argument("--with-vnindex", action="store_true")
    ap.add_argument("--with-oil", action="store_true", help="Overlay rebased WTI oil price (needs FRED_API_KEY-backed data)")
    ap.add_argument("--events", type=str, default="",
                     help="Comma-separated keys from KNOWN_2025_GOLD_EVENTS to mark with vertical lines")
    ap.add_argument("--output", type=str, default="docs/figures/market_factor_comparison.png")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    event_keys = [k.strip() for k in args.events.split(",") if k.strip()]
    build_chart(tickers, args.with_gold, args.with_vnindex, args.with_oil, event_keys, args.output)


if __name__ == "__main__":
    main()
