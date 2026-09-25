# Vietstock Premium host schedule

The project deliberately does not mount a logged-in Premium browser profile
into Docker or Airflow. On macOS, after interactive login and company
discovery, install the opt-in host job:

```bash
python -m scripts.install_vietstock_daily_sync --install --hour 18 --minute 10
```

It runs the incremental sync daily and writes status to the database visible
in the Admin `Vietstock sync` tab. If the Premium browser session expires,
the run is recorded as failed; refresh it with the normal `login` command.
Remove the job with:

```bash
python -m scripts.install_vietstock_daily_sync --uninstall
```

## Structured trading snapshot

The structured trading crawler is a separate job from the PDF/document
crawler. It records rendered Vietstock trading tables in
`vietstock_structured_tables` and is intended to run once per day after the
market closes. It is a daily browser snapshot, not a live tick feed; the
normalized daily OHLCV path remains `scripts/market_snapshot.py`.

First complete one interactive login with the project's persistent profile,
then verify one ticker. The normal login command opens a browser profile that
the scheduled job reuses:

```bash
cd /Users/mac/Desktop/Project/Thesis_Project
source .venv/bin/activate
PYTHONPATH=. python -m scripts.vietstock_premium_sync login
PYTHONPATH=. python -m scripts.vietstock_trading_daily_sync \
  --tickers VNM
```

After the smoke test succeeds, install the daily host job:

```bash
PYTHONPATH=. python -m scripts.install_vietstock_trading_daily_sync \
  --install --hour 18 --minute 20
```

The job launches its own headless browser process from the saved project
profile; it does not expect Chrome on port `9222` to remain available. Its log
is `data/private/vietstock_trading_daily_sync.log`. Remove it with
`--uninstall` when the schedule is no longer needed.
