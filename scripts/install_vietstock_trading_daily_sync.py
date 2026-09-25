#!/usr/bin/env python3
"""Install or remove the opt-in macOS daily structured trading sync job.

The job uses the project's persistent Playwright profile, so it does not
depend on a Chrome window or CDP port being open at the scheduled time. It
never stores a password or browser cookie in launchd. Complete one interactive
Vietstock login with the existing login command before installing the job.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = "com.finex.vietstock-trading-daily-sync"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG_PATH = ROOT / "data" / "private" / "vietstock_trading_daily_sync.log"


def _plist(hour: int, minute: int) -> dict:
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        raise RuntimeError(f"Project Python was not found at {python}")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(python), "-m", "scripts.vietstock_trading_daily_sync",
            "--all-known",
        ],
        "WorkingDirectory": str(ROOT),
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "StandardOutPath": str(LOG_PATH),
        "StandardErrorPath": str(LOG_PATH),
        "ProcessType": "Background",
    }


def _bootout(uid: int) -> None:
    subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(PLIST_PATH)], capture_output=True, text=True)


def install(hour: int, minute: int) -> None:
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("hour must be 0-23 and minute must be 0-59")
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLIST_PATH.open("wb") as f:
        plistlib.dump(_plist(hour, minute), f, sort_keys=False)
    uid = os.getuid()
    _bootout(uid)
    subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(PLIST_PATH)], check=True)
    print(f"Installed {LABEL}: daily at {hour:02d}:{minute:02d}; logs: {LOG_PATH}")


def uninstall() -> None:
    _bootout(os.getuid())
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
    print(f"Removed {LABEL}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--install", action="store_true")
    action.add_argument("--uninstall", action="store_true")
    parser.add_argument("--hour", type=int, default=18)
    parser.add_argument("--minute", type=int, default=20)
    args = parser.parse_args()
    if args.install:
        install(args.hour, args.minute)
    else:
        uninstall()


if __name__ == "__main__":
    main()
