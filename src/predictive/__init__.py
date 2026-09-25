"""Data layer for the (not-yet-built) PREDICTIVE query-routing branch -
forecasting a future financial metric (e.g. next fiscal year's net_margin)
from this year's fundamentals + macro context, via a classical ML model
(Ridge/XGBoost - see docs/PROGRESS.md's 2026-07-29 architecture review for
why classical ML rather than a fine-tuned LLM: financial_line_items/
financial_ratios are large enough for it, unlike the ~140-row benchmark.jsonl
which is only big enough for evaluation, not training).

Added 2026-07-29 (Task #5 from that review): src/predictive/feature_builder.py
builds `predictive_features` (see infra/postgres/migrations/005_predictive_features.sql),
a wide-format, leakage-controlled feature table - explicit target/horizon
columns and an information_cutoff (the disclosure_date the current period's
figures actually became public) so a future train/test split can enforce
"only train on rows whose information_cutoff is before the test period",
not just a random shuffle that could let a company's own future disclosure
leak into its own training row.

Deliberately data-layer ONLY at this stage - no model training code here yet.
Per the review's own conclusion: "Bước tiếp theo không phải thêm model ngay,
mà là tạo ba artefact rõ ràng" (the next step is not to add a model yet, but
to build three clear data artefacts) - this is one of the three (the other
two: src/causal/panel_builder.py's causal_panel_snapshot, and the
ABSA annotation schema - Task #6). No router branch, no XGBoost import,
and no training script exist yet - see docs/PROGRESS.md for the up-to-date
status before assuming otherwise.
"""
