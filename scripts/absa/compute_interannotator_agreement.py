"""Task #22, roadmap item 5 (2026-07-30): computes Cohen's kappa + raw
agreement for the ABSA pilot's double-annotated subset - "compute Cohen's
kappa + agreement by aspect + agreement by sentiment. If agreement is low,
fix the annotation guideline before fine-tuning."

Reads data/eval/absa_annotations.jsonl (via src.absa.annotation_schema.
load_annotations()) and data/eval/absa_pilot_candidates.jsonl (via
scripts.absa.sample_pilot_paragraphs's candidate format) to find the
double_annotate=True subset, then pairs up the two annotators' labels for
every (chunk, aspect) in that subset.

Cohen's kappa is implemented directly with numpy (no external stats
package needed) - see cohens_kappa() below: po = observed agreement, pe =
chance agreement from each annotator's own marginal label distribution,
kappa = (po - pe) / (1 - pe). Standard formula (Cohen 1960), not a novel
metric.

Reported, per the roadmap's explicit list:
- Overall kappa (every (chunk, aspect) judgment pooled into one
  classification task across all 5 aspects).
- Kappa PER ASPECT (5 separate numbers) - a taxonomy can have good overall
  agreement while one specific aspect (e.g. "risk_disclosure", often
  genuinely ambiguous) is much worse; pooling would hide that.
- Raw percent agreement (both overall and per-aspect) alongside kappa -
  kappa already corrects for chance, but a reader should be able to see
  both numbers, not just the corrected one.
- Agreement BY SENTIMENT CLASS - for each of the 5 sentiment values, what
  fraction of (chunk, aspect) instances where EITHER annotator picked that
  class did BOTH annotators pick it (a class-level agreement rate, catches
  e.g. "mixed" being the specific value annotators disagree on most, even
  if overall kappa looks fine).

Interpretation guidance printed alongside the numbers: Landis & Koch
(1977)'s standard kappa bands (<0.00 poor, 0.00-0.20 slight, 0.21-0.40
fair, 0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.00 almost perfect) -
a well-known, external convention, not one invented for this thesis. Per
the roadmap's explicit instruction, kappa below the "substantial" (0.61)
threshold prints a loud recommendation to fix the annotation guideline
BEFORE doing any further large-scale labeling or fine-tuning, rather than
silently proceeding.

Usage (needs data/eval/absa_annotations.jsonl with rows from AT LEAST 2
distinct annotator_id values covering the double_annotate subset - i.e.
run scripts/absa_annotate_cli.py --from-file ... twice, once per person,
first):
    python3 -m scripts.absa.compute_interannotator_agreement
    python3 -m scripts.absa.compute_interannotator_agreement --annotator-a quang --annotator-b linh
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.absa.absa_pipeline import ASPECTS
from src.absa.annotation_schema import ANNOTATIONS_PATH, load_annotations

CANDIDATES_PATH = Path("data/eval/absa_pilot_candidates.jsonl")

KAPPA_SUBSTANTIAL_THRESHOLD = 0.61  # Landis & Koch 1977 - "substantial agreement" band starts here


def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Standard Cohen's kappa (Cohen 1960) over two equal-length lists of
    paired categorical labels. Returns 1.0 for perfect agreement even in
    the degenerate all-one-category case (po=pe=1), 0.0 if po<1 in that
    same degenerate case (pe=1 but not perfect agreement -> kappa formula's
    0/0 would otherwise be undefined)."""
    assert len(labels_a) == len(labels_b), "labels_a/labels_b must be the same length (paired judgments)"
    n = len(labels_a)
    if n == 0:
        return float("nan")

    categories = sorted(set(labels_a) | set(labels_b))
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    confusion = np.zeros((k, k))
    for a, b in zip(labels_a, labels_b):
        confusion[idx[a], idx[b]] += 1

    po = float(np.trace(confusion) / n)
    row_marg = confusion.sum(axis=1) / n
    col_marg = confusion.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))

    if pe >= 1.0 - 1e-12:
        return 1.0 if po >= 1.0 - 1e-12 else 0.0
    return (po - pe) / (1 - pe)


def load_double_annotate_keys(candidates_path: Path = CANDIDATES_PATH) -> set:
    keys = set()
    with candidates_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("double_annotate"):
                keys.add((row["ticker"], row["source_file"], row["page"], row["chunk_index"]))
    return keys


def _chunk_key(record: dict):
    return (record.get("ticker"), record.get("source_file"), record.get("page"), record.get("chunk_index"))


def pair_annotations(records: list, double_annotate_keys: set, annotator_a: str = None,
                      annotator_b: str = None) -> tuple:
    """Groups `records` by (annotator_id, chunk_key), restricted to
    double_annotate_keys, and returns (annotator_a, annotator_b, paired)
    where paired is {chunk_key: {"a": aspects_list_a, "b": aspects_list_b}}
    for every key BOTH annotators have a row for. Auto-detects the two
    annotator_ids present in the double-annotate subset if not given
    explicitly - raises ValueError if there aren't exactly 2 (ambiguous
    which pair to compare, or not enough people have labeled the subset
    yet)."""
    by_annotator_key = defaultdict(dict)
    for r in records:
        key = _chunk_key(r)
        if key in double_annotate_keys:
            by_annotator_key[r["annotator_id"]][key] = r["aspects"]

    if annotator_a is None or annotator_b is None:
        present = sorted(by_annotator_key.keys())
        if len(present) != 2:
            raise ValueError(
                f"Cần đúng 2 annotator_id đã gán nhãn tập double_annotate để tính kappa, "
                f"hiện có {len(present)}: {present}. Dùng --annotator-a/--annotator-b để chỉ rõ nếu có nhiều hơn 2."
            )
        annotator_a, annotator_b = present

    keys_a = set(by_annotator_key.get(annotator_a, {}).keys())
    keys_b = set(by_annotator_key.get(annotator_b, {}).keys())
    common = keys_a & keys_b
    if not common:
        raise ValueError(
            f"'{annotator_a}' và '{annotator_b}' không có chunk nào chung trong tập double_annotate "
            f"({len(keys_a)} vs {len(keys_b)} chunk đã gán riêng lẻ) - cả 2 người cần gán nhãn cùng 1 file "
            f"--from-file với --double-annotate-only."
        )

    paired = {
        key: {"a": by_annotator_key[annotator_a][key], "b": by_annotator_key[annotator_b][key]}
        for key in common
    }
    return annotator_a, annotator_b, paired


def compute_agreement_report(paired: dict) -> dict:
    """paired: {chunk_key: {"a": aspects_list, "b": aspects_list}} (both
    aspects_list are length-5, one entry per ASPECTS key, per the
    always-5-aspects schema). Returns overall + per-aspect kappa/agreement
    + per-sentiment-class agreement."""
    overall_a, overall_b = [], []
    per_aspect_a = defaultdict(list)
    per_aspect_b = defaultdict(list)

    for key, sides in paired.items():
        a_by_aspect = {e["aspect"]: e["sentiment"] for e in sides["a"]}
        b_by_aspect = {e["aspect"]: e["sentiment"] for e in sides["b"]}
        for aspect in ASPECTS:
            if aspect not in a_by_aspect or aspect not in b_by_aspect:
                continue  # shouldn't happen under the always-5-aspects schema, but skip defensively rather than crash
            sa, sb = a_by_aspect[aspect], b_by_aspect[aspect]
            overall_a.append(sa)
            overall_b.append(sb)
            per_aspect_a[aspect].append(sa)
            per_aspect_b[aspect].append(sb)

    n_judgments = len(overall_a)
    overall_agree = float(np.mean([a == b for a, b in zip(overall_a, overall_b)])) if n_judgments else float("nan")
    overall_kappa = cohens_kappa(overall_a, overall_b) if n_judgments else float("nan")

    by_aspect = {}
    for aspect in ASPECTS:
        a_list, b_list = per_aspect_a[aspect], per_aspect_b[aspect]
        if not a_list:
            continue
        by_aspect[aspect] = {
            "n": len(a_list),
            "percent_agreement": float(np.mean([a == b for a, b in zip(a_list, b_list)])),
            "kappa": cohens_kappa(a_list, b_list),
        }

    all_sentiments = sorted(set(overall_a) | set(overall_b))
    by_sentiment = {}
    for sentiment in all_sentiments:
        both = sum(1 for a, b in zip(overall_a, overall_b) if a == sentiment and b == sentiment)
        either = sum(1 for a, b in zip(overall_a, overall_b) if a == sentiment or b == sentiment)
        by_sentiment[sentiment] = {
            "n_either_annotator_used_this_label": either,
            "n_both_agreed": both,
            "agreement_rate": (both / either) if either else float("nan"),
        }

    return {
        "n_chunks": len(paired), "n_judgments": n_judgments,
        "overall_percent_agreement": overall_agree, "overall_kappa": overall_kappa,
        "by_aspect": by_aspect, "by_sentiment": by_sentiment,
    }


def _kappa_band(kappa: float) -> str:
    if np.isnan(kappa):
        return "n/a"
    if kappa < 0:
        return "poor (< 0.00)"
    if kappa < 0.20:
        return "slight (0.00-0.20)"
    if kappa < 0.40:
        return "fair (0.21-0.40)"
    if kappa < 0.60:
        return "moderate (0.41-0.60)"
    if kappa < KAPPA_SUBSTANTIAL_THRESHOLD:
        return "moderate-to-substantial (0.60-0.61)"
    if kappa < 0.81:
        return "substantial (0.61-0.80)"
    return "almost perfect (0.81-1.00)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-file", type=Path, default=ANNOTATIONS_PATH)
    ap.add_argument("--candidates-file", type=Path, default=CANDIDATES_PATH)
    ap.add_argument("--annotator-a", default=None)
    ap.add_argument("--annotator-b", default=None)
    ap.add_argument("--report", type=Path, default=Path("reports/absa_interannotator_agreement.json"))
    args = ap.parse_args()

    records = load_annotations(args.annotations_file)
    double_keys = load_double_annotate_keys(args.candidates_file)
    annotator_a, annotator_b, paired = pair_annotations(records, double_keys, args.annotator_a, args.annotator_b)
    report = compute_agreement_report(paired)
    report["annotator_a"] = annotator_a
    report["annotator_b"] = annotator_b

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n=== Inter-annotator agreement: {annotator_a} vs {annotator_b} ===")
    print(f"n_chunks={report['n_chunks']}  n_judgments (chunks x 5 aspects)={report['n_judgments']}")
    print(f"Overall: percent_agreement={report['overall_percent_agreement']:.1%}  "
          f"kappa={report['overall_kappa']:.3f} ({_kappa_band(report['overall_kappa'])})")
    print("\nBy aspect:")
    for aspect, info in report["by_aspect"].items():
        print(f"  {aspect:20s} n={info['n']:3d}  agreement={info['percent_agreement']:.1%}  "
              f"kappa={info['kappa']:.3f} ({_kappa_band(info['kappa'])})")
    print("\nBy sentiment class (agreement rate = both-agreed / either-used-this-label):")
    for sentiment, info in report["by_sentiment"].items():
        print(f"  {sentiment:15s} n_either_used={info['n_either_annotator_used_this_label']:3d}  "
              f"agreement_rate={info['agreement_rate']:.1%}")

    if report["overall_kappa"] < KAPPA_SUBSTANTIAL_THRESHOLD:
        print(
            f"\n*** Overall kappa ({report['overall_kappa']:.3f}) is BELOW the 'substantial agreement' "
            f"threshold ({KAPPA_SUBSTANTIAL_THRESHOLD}) - per the roadmap's explicit instruction, fix the "
            f"annotation guideline BEFORE labeling more data or fine-tuning anything on this corpus. "
            f"Check the per-aspect/per-sentiment breakdown above for WHERE the disagreement concentrates. ***"
        )
    else:
        print(f"\nOverall kappa ({report['overall_kappa']:.3f}) meets the 'substantial agreement' threshold "
              f"({KAPPA_SUBSTANTIAL_THRESHOLD}) - safe to proceed to the full ~1,000-2,000-sample labeling round.")

    print(f"\nReport written: {args.report}")


if __name__ == "__main__":
    main()
