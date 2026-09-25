#!/usr/bin/env python3
"""Install or remove the opt-in macOS daily Vietstock Premium sync job.

The job uses the existing persistent Playwright profile created by
``python -m scripts.vietstock_premium_sync login``. It never carries a
password in the plist. If the Premium session expires, the scheduled run
records an honest failure in ``vietstock_ingestion_runs``; refresh the local
session with the login command rather than adding credentials to launchd.

Examples:
  python -m scripts.install_vietstock_daily_sync --install --hour 18 --minute 10
  python -m scripts.install_vietstock_daily_sync --uninstall
"""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = "com.finex.vietstock-premium-sync"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG_PATH = ROOT / "data" / "private" / "vietstock_daily_sync.log"


def _plist(hour: int, minute: int) -> dict:
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        raise RuntimeError(f"Project Python was not found at {python}")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 2026-09-21: chạy sync xong thì in luôn bản tóm tắt tiếng Việt
    # (scripts/vietstock_daily_sync_report.py) nối vào CÙNG log - launchd
    # ProgramArguments chỉ chạy được 1 lệnh, nên bọc cả 2 bước trong
    # `sh -c "A; B"` thay vì đổi ProgramArguments thành một lệnh đơn.
    # `;` (không phải `&&`) có chủ ý: report vẫn chạy và cho biết lần chạy
    # trước dừng ở đâu ngay cả khi bước sync thoát khác 0 (ví dụ phiên
    # Premium hết hạn) - im lặng bỏ qua report chỉ vì sync lỗi sẽ che mất
    # đúng thông tin cần xem nhất lúc đó.
    sync_cmd = f'"{python}" -m scripts.vietstock_premium_sync sync --all-known --documents'
    report_cmd = f'"{python}" -m scripts.vietstock_daily_sync_report'
    return {
        "Label": LABEL,
        "ProgramArguments": ["/bin/sh", "-c", f"{sync_cmd}; {report_cmd}"],
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
    parser.add_argument("--minute", type=int, default=10)
    args = parser.parse_args()
    if args.install:
        install(args.hour, args.minute)
    else:
        uninstall()


if __name__ == "__main__":
    main()
