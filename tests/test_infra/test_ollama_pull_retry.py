"""Test cho infra/docker/ollama-pull.sh (2026-08-20, sau khi gặp lỗi thật).

Bối cảnh: `ollama pull` bị dừng giữa chừng bởi `docker compose down` nhiều
lần (khi debug tốc độ mạng) làm hỏng sổ sách file tạm bên trong Ollama. Mọi
lần tải sau đó đều dừng ở ĐÚNG một lỗi:
    Error: remove .../blobs/sha256-...-partial-0: no such file or directory
Script CŨ có `set -e` nên lỗi này làm cả script thoát ngay, container chết
theo (đây là tiến trình PID 1), `restart: unless-stopped` khởi động lại,
script chạy lại từ đầu, gặp ĐÚNG lỗi đó lần nữa - vòng lặp crash âm thầm,
người dùng chỉ thấy container cứ "Restarting" mà không rõ vì sao.

KHÔNG chạy `ollama` thật (không cần mạng, không cần Docker) - dựng một
`ollama` GIẢ trong PATH mô phỏng đúng lỗi trên, rồi chạy CHÍNH SÁCH THẬT
qua `sh`, kiểm tra hành vi thật chứ không đoán.

Chạy:
    python3 -m unittest tests.test_infra.test_ollama_pull_retry -v
"""
import fcntl
import os
import shutil
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "infra" / "docker" / "ollama-pull.sh"

FAKE_OLLAMA = """#!/bin/sh
# ollama giả cho test - mô phỏng: serve chạy nền, list rỗng tới khi có
# marker "pulled", pull thất bại N lần đầu với ĐÚNG lỗi thật đã gặp rồi
# mới thành công (hoặc luôn thất bại nếu FAKE_FAIL_COUNT rất lớn).
case "$1" in
  serve)
    trap 'exit 0' TERM INT
    while true; do sleep 1; done
    ;;
  list)
    echo "NAME             ID       SIZE    MODIFIED"
    if [ -f "$FAKE_STATE_DIR/pulled" ]; then
      echo "${OLLAMA_MODEL}    abc123    1B    now"
    fi
    exit 0
    ;;
  pull)
    n=$(cat "$FAKE_STATE_DIR/attempts" 2>/dev/null || echo 0)
    n=$((n + 1))
    echo "$n" > "$FAKE_STATE_DIR/attempts"
    if [ "$n" -le "${FAKE_FAIL_COUNT:-0}" ]; then
      echo "Error: remove /fake/blobs/sha256-x-partial-0: no such file or directory" >&2
      exit 1
    fi
    touch "$FAKE_STATE_DIR/pulled"
    echo "pulling manifest"
    exit 0
    ;;
  *) exit 1 ;;
esac
"""


class _OllamaPullHarness(unittest.TestCase):
    """Dựng PATH có `ollama` giả + thư mục trạng thái tạm cho mỗi test."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ollama_pull_test_"))
        self.bin_dir = self.tmp / "bin"
        self.bin_dir.mkdir()
        self.state_dir = self.tmp / "state"
        self.state_dir.mkdir()
        self.blobs_dir = self.tmp / "blobs"
        self.blobs_dir.mkdir()

        fake = self.bin_dir / "ollama"
        fake.write_text(FAKE_OLLAMA, encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, fail_count: int, max_attempts: int = 5, timeout: int = 20):
        env = {
            **os.environ,
            "PATH": f"{self.bin_dir}:{os.environ.get('PATH', '')}",
            "FAKE_STATE_DIR": str(self.state_dir),
            "FAKE_FAIL_COUNT": str(fail_count),
            "OLLAMA_MODEL": "fakemodel",
            "OLLAMA_BLOBS_DIR": str(self.blobs_dir),
            "OLLAMA_PULL_MAX_ATTEMPTS": str(max_attempts),
        }
        proc = subprocess.Popen(
            ["sh", str(SCRIPT)], env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        # QUAN TRỌNG: `readline()` ở chế độ blocking mặc định CHẶN VÔ HẠN
        # nếu chưa có dòng mới - deadline kiểm tra GIỮA hai lần đọc không
        # cứu được vì bản thân lệnh đọc không có timeout. Bản đầu của test
        # này bị đúng lỗi đó và treo cả sandbox. Chuyển fd sang non-blocking
        # để vòng lặp thật sự tôn trọng deadline.
        fd = proc.stdout.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        buf = ""
        deadline = time.time() + timeout
        success_marker = "[ollama-pull] Sẵn sàng. Model: fakemodel"
        try:
            while time.time() < deadline:
                try:
                    chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
                    if chunk:
                        buf += chunk
                except BlockingIOError:
                    chunk = ""
                if success_marker in buf:
                    proc.terminate()
                    proc.wait(timeout=5)
                    return buf, None
                if chunk == "" and proc.poll() is not None:
                    # Tiến trình đã thoát - vét nốt phần còn lại trong pipe.
                    try:
                        while True:
                            rest = os.read(fd, 4096)
                            if not rest:
                                break
                            buf += rest.decode("utf-8", errors="replace")
                    except (BlockingIOError, OSError):
                        pass
                    return buf, proc.returncode
                if chunk == "":
                    time.sleep(0.05)
            proc.kill()
            self.fail(f"Script không kết thúc/thành công trong {timeout}s.\nLog:\n{buf}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            proc.stdout.close()


class TestSelfHealsFromKnownBug(_OllamaPullHarness):
    def test_recovers_after_the_exact_error_seen_in_production(self):
        """Lỗi thật gặp 2026-08-20: 2 lần đầu thất bại đúng thông điệp đã
        thấy trên terminal người dùng, lần 3 thành công. Script CŨ (set -e,
        không thử lại) sẽ dừng ngay ở lần 1 - test này chỉ xanh với bản sửa."""
        out, code = self._run(fail_count=2, max_attempts=5)
        self.assertIsNone(code, "script phải chạy tới lúc serve, không tự thoát lỗi")
        self.assertIn("Lần thử 1/5", out)
        self.assertIn("Lần thử 2/5", out)
        self.assertIn("Lần thử 3/5", out)
        self.assertIn("Sẵn sàng. Model: fakemodel", out)

    def test_cleans_up_stale_partial_files_before_retrying(self):
        """Đúng cơ chế tự hồi phục: sau lần thất bại, file '*-partial-*'
        còn sót phải bị dọn trước khi thử lại - đây là phần thay đổi hành
        vi thật so với bản cũ, không chỉ là thử lại suông."""
        stale = self.blobs_dir / "sha256-deadbeef-partial-0"
        stale.write_text("rác từ lần tải bị ngắt giữa chừng")
        self.assertTrue(stale.exists())

        out, code = self._run(fail_count=1, max_attempts=5)
        self.assertIsNone(code)
        self.assertIn("Sẵn sàng. Model: fakemodel", out)
        self.assertFalse(stale.exists(), "file tạm hỏng phải bị xoá trong lúc tự hồi phục")

    def test_already_pulled_model_skips_straight_through(self):
        """Không được thử tải lại nếu model đã có - hành vi cũ vẫn phải
        giữ nguyên, bản sửa chỉ thêm khả năng thử lại, không đổi đường
        'đã có sẵn'."""
        (self.state_dir / "pulled").write_text("1")
        out, code = self._run(fail_count=0, max_attempts=5)
        self.assertIsNone(code)
        self.assertIn("Đã có model fakemodel, bỏ qua bước tải.", out)
        self.assertNotIn("Lần thử", out)


class TestGivesUpLoudlyInsteadOfLoopingSilently(_OllamaPullHarness):
    def test_exits_with_actionable_message_after_max_attempts(self):
        """Đây chính là lỗi đã gặp: nếu KHÔNG có giới hạn, script cũ lặp
        crash vô hạn trong im lặng. Bản sửa phải DỪNG sau MAX_ATTEMPTS và
        in ra đúng câu lệnh khắc phục, không phải chỉ 'thất bại' mù mờ."""
        out, code = self._run(fail_count=999, max_attempts=3)
        self.assertEqual(code, 1)
        self.assertIn("Lần thử 1/3", out)
        self.assertIn("Lần thử 2/3", out)
        self.assertIn("Lần thử 3/3", out)
        self.assertNotIn("Lần thử 4", out, "phải dừng đúng ở MAX_ATTEMPTS, không thử thêm")
        self.assertIn("docker volume rm thesis_project_ollama_models", out)
        self.assertNotIn("[ollama-pull] Sẵn sàng. Model: fakemodel", out)


if __name__ == "__main__":
    unittest.main()
