"""Single shared loader joining `predictive_features` against the FROZEN
`predictive_split_manifest` (Task #18) - every model script from here on
(M0/M1 baselines, M2/M3, SHAP) MUST import load_features_with_split()
rather than re-deriving train/validation/test membership itself. That's the
whole point of freezing the split into its own table (see
scripts/predictive/freeze_predictive_split.py's module docstring): a model
script that computed its own year-based split inline could silently drift
from another model script's split (e.g. after someone tweaks a boundary
year in one place and not the other), making cross-model metric comparisons
meaningless. Import this instead of writing "year < 2023" anywhere else.
"""
import pandas as pd
from sqlalchemy import text

from src.core.db import get_session
from src.predictive.feature_builder import FEATURE_COLUMNS
from scripts.predictive.freeze_predictive_split import SPLIT_MANIFEST_VERSION

_FEATURE_SELECT_COLUMNS = (
    ["ticker", "fiscal_year", "period", "consolidation", "sector", "exchange"]
    + FEATURE_COLUMNS
    + [f"{c}_lag1" for c in FEATURE_COLUMNS]
    + ["revenue_growth_yoy", "net_profit_growth_yoy", "total_assets_growth_yoy"]
    + ["cpi_annual_avg", "gdp_growth_annual_avg", "gold_price_annual_avg", "usdvnd_annual_avg", "oil_price_annual_avg"]
    + ["information_cutoff", "target_col", "horizon", "target_value", "target_observed_at",
       "eligible_for_training", "feature_hash"]
)


def load_features_with_split(feature_version: str = "predfeat_v1", target_col: str = "net_margin",
                              horizon: int = 1, split_manifest_version: str = None) -> pd.DataFrame:
    """Returns a DataFrame of predictive_features rows INNER JOINed against
    predictive_split_manifest on the shared natural key, with a "split"
    column added ('train' / 'validation' / 'test').

    INNER join, deliberately: a predictive_features row that hasn't been
    frozen yet (scripts/predictive/freeze_predictive_split.py hasn't been
    (re-)run since it was written) is silently EXCLUDED rather than given
    an ad-hoc split here - this forces the freeze step to actually have
    happened before any model can train, rather than each model script
    quietly falling back to computing its own split for rows the manifest
    doesn't yet cover.

    split_manifest_version defaults to the CURRENT split rule's
    auto-derived version string (see freeze_predictive_split.py) - pass an
    explicit older string only if you deliberately want to reproduce a
    result against a since-changed split rule."""
    if split_manifest_version is None:
        split_manifest_version = SPLIT_MANIFEST_VERSION

    cols_sql = ", ".join(f"pf.{c}" for c in _FEATURE_SELECT_COLUMNS)
    query = text(f"""
        SELECT {cols_sql}, m.split
        FROM predictive_features pf
        JOIN predictive_split_manifest m
          ON m.split_manifest_version = :split_manifest_version
         AND m.ticker = pf.ticker AND m.fiscal_year = pf.fiscal_year AND m.period = pf.period
         AND m.consolidation IS NOT DISTINCT FROM pf.consolidation
         AND m.target_col = pf.target_col AND m.horizon = pf.horizon
        WHERE pf.feature_version = :feature_version
          AND pf.target_col = :target_col
          AND pf.horizon = :horizon
    """)
    with get_session() as session:
        result = session.execute(query, {
            "split_manifest_version": split_manifest_version,
            "feature_version": feature_version, "target_col": target_col, "horizon": horizon,
        })
        rows = result.fetchall()
        columns = list(result.keys())

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        print(f"load_features_with_split(): 0 row(s) - has scripts/predictive/freeze_predictive_split.py "
              f"been run for feature_version={feature_version!r}, target_col={target_col!r}, horizon={horizon}?")
        return df

    counts = df["split"].value_counts().to_dict()
    print(f"load_features_with_split(): {len(df)} row(s) - "
          f"train={counts.get('train', 0)}, validation={counts.get('validation', 0)}, test={counts.get('test', 0)}")
    return df
