"""Cross-sectional causal/what-if analysis layer, built ON TOP of the DAG's
financial_line_items table.

Scope note (why this module looks the way it does): a SINGLE company's
disclosed history is only 15-20 quarters - far too few observations for
time-series causal discovery (PCMCI, Granger, VAR-LiNGAM) to be reliable.
But the COMPANY PANEL (216 tickers x up to ~19 periods = hundreds of
ticker-period rows) is exactly the data shape cross-sectional causal
methods (CausalML/EconML-style meta-learners) are designed for: many units,
few time points each - not one unit with a long time series. That is the
version of "causal ML" implemented here: cross-sectional treatment-effect
estimation across companies (e.g. "does higher leverage cause lower
margins, controlling for sector/year?"), not single-company forecasting.

As more Vietstock exports get loaded, panel size grows and the confidence
gate in treatment_effect.py automatically upgrades results from
"exploratory" to "standard" - no code change needed.
"""
