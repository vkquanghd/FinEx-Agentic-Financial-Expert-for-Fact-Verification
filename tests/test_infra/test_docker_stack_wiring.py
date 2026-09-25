"""Test dây nối Docker (2026-08-20) - KHÔNG cần Docker, KHÔNG cần mạng.

Vì sao cần: lỗi cấu hình Docker thường chỉ lộ ra sau khi build xong vài GB
rồi container khởi động sai - vòng phản hồi rất chậm. Các test dưới đây bắt
đúng những lỗi ĐÃ THẬT SỰ MẮC trong lúc dựng stack này, mỗi test khoá lại
một lỗi cụ thể chứ không kiểm tra chung chung:

  1. anchor `x-api-common` từng bị đặt NHẦM trong khối `services:` -> Compose
     hiểu nó là một service rỗng. Test: `services` chỉ chứa đúng tên thật.
  2. `ADMIN_API_KEY: ${ADMIN_API_KEY}` từng được khai trong compose. Hai
     biến đó KHÔNG có trong .env gốc nên Compose đặt chúng thành chuỗi
     rỗng, mà biến môi trường ưu tiên cao hơn env_file trong
     pydantic-settings -> ghi đè giá trị thật trong api/.env -> login trả
     503 dù cấu hình đúng. Test: hai khoá đó KHÔNG được xuất hiện.
  3. "sửa code thì Docker tự cập nhật" chỉ đúng nếu CẢ HAI điều kiện có
     mặt: bind-mount code VÀ cờ --reload. Thiếu một trong hai là hỏng âm
     thầm (vẫn chạy, nhưng phải build lại mới thấy thay đổi).
  4. SSE (/api/qa/ask/stream) hỏng âm thầm nếu nginx bật đệm - người dùng
     thấy spinner câm rồi nhận tất cả một lượt.

Chạy:
    python3 -m unittest tests.test_infra.test_docker_stack_wiring -v
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

COMPOSE = ROOT / "docker-compose.yml"
NGINX_CONF = ROOT / "infra" / "docker" / "nginx" / "default.conf"
API_DOCKERFILE = ROOT / "infra" / "docker" / "api.Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"

API_SERVICES = ["qa", "feedback", "admin", "gateway"]


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


class TestComposeStructure(unittest.TestCase):
    def test_all_expected_services_present(self):
        s = _compose()["services"]
        for name in ["postgres", "airflow", "ollama", "frontend", *API_SERVICES]:
            with self.subTest(service=name):
                self.assertIn(name, s)

    def test_anchor_is_not_mistaken_for_a_service(self):
        """Lỗi #1: anchor đặt trong `services:` thì Compose cố dựng một
        service tên 'x-api-common'. Khoá `x-*` phải ở cấp cao nhất."""
        d = _compose()
        self.assertIn("x-api-common", d, "anchor phải ở CẤP CAO NHẤT")
        for key in d["services"]:
            self.assertFalse(key.startswith("x-"),
                             f"'{key}' là anchor bị lọt vào services:")

    def test_anchor_merged_into_all_four_api_services(self):
        s = _compose()["services"]
        for name in API_SERVICES:
            with self.subTest(service=name):
                self.assertEqual(s[name]["image"], "thesis-api:latest",
                                 "anchor chưa được merge vào service này")


class TestSecretsAreNotClobbered(unittest.TestCase):
    """Lỗi #2 - lỗi tinh vi nhất trong cả stack, vì nó hỏng ÂM THẦM."""

    def test_admin_key_and_session_secret_absent_from_compose_env(self):
        s = _compose()["services"]
        for name in API_SERVICES:
            env = s[name].get("environment", {})
            for key in ("ADMIN_API_KEY", "SESSION_SECRET"):
                with self.subTest(service=name, key=key):
                    self.assertNotIn(
                        key, env,
                        f"{key} khai trong compose sẽ thành chuỗi rỗng và ghi đè "
                        f"api/.env (biến môi trường ưu tiên hơn env_file) -> 503.",
                    )

    def test_api_env_reaches_container_via_bind_mount(self):
        """Hệ quả của test trên: nếu KHÔNG khai trong compose thì api/.env
        BẮT BUỘC phải vào được container, nếu không hai biến đó biến mất
        hoàn toàn. Bind mount ./api chính là đường đi đó."""
        s = _compose()["services"]
        for name in API_SERVICES:
            with self.subTest(service=name):
                self.assertIn("./api:/app/api:ro", s[name]["volumes"])


class TestHotReloadActuallyWorks(unittest.TestCase):
    """Lỗi #3: cần ĐỦ CẢ HAI (bind mount + --reload) mới có 'sửa code là
    Docker tự cập nhật'. Thiếu một cái vẫn chạy được nên rất dễ lọt."""

    def test_every_api_service_bind_mounts_source(self):
        s = _compose()["services"]
        for name in API_SERVICES:
            vols = s[name]["volumes"]
            with self.subTest(service=name):
                self.assertIn("./src:/app/src:ro", vols)
                self.assertIn("./api:/app/api:ro", vols)

    def test_every_api_service_runs_uvicorn_with_reload(self):
        s = _compose()["services"]
        for name in API_SERVICES:
            cmd = s[name]["command"]
            with self.subTest(service=name):
                self.assertIn("--reload", cmd)
                self.assertIn("--reload-dir src", cmd)
                self.assertIn("--reload-dir api/services", cmd)

    def test_dockerfile_does_not_copy_application_code(self):
        """Nếu Dockerfile COPY src/ hay api/ vào image thì code trong image
        sẽ CHE bind mount ở một số cấu hình, và mọi thay đổi lại đòi build
        lại - đúng thứ thiết lập này tồn tại để tránh."""
        text = API_DOCKERFILE.read_text(encoding="utf-8")
        copies = re.findall(r"^COPY\s+(\S+)", text, re.MULTILINE)
        for src in copies:
            with self.subTest(copy=src):
                self.assertNotIn("src", src.split("/")[0].replace("api", ""))
                self.assertFalse(src in ("src", "api", "./src", "./api"),
                                 f"Dockerfile COPY {src} phá cơ chế bind-mount")

    def test_frontend_is_bind_mounted_not_baked(self):
        """Frontend là HTML/CSS/JS thuần (Alpine.js qua CDN), không có bước
        build - nên nginx phải phục vụ THẲNG thư mục trên máy để sửa file
        rồi F5 là thấy."""
        fe = _compose()["services"]["frontend"]
        self.assertIn("./api/frontend:/usr/share/nginx/html:ro", fe["volumes"])


class TestFrontendBackendSeparation(unittest.TestCase):
    """Ranh giới người dùng chốt 2026-08-20: backend REST thuần, nginx giữ UI."""

    def test_backend_does_not_serve_frontend(self):
        s = _compose()["services"]
        for name in API_SERVICES:
            with self.subTest(service=name):
                self.assertEqual(str(s[name]["environment"]["SERVE_FRONTEND"]), "0")

    def test_gateway_is_not_exposed_to_host(self):
        """Chỉ nginx chạm được từ ngoài - đúng nguyên tắc 'một cửa vào duy
        nhất' trong docstring của gateway."""
        gw = _compose()["services"]["gateway"]
        self.assertNotIn("ports", gw, "gateway không được map cổng ra host")

    def test_only_frontend_publishes_web_port(self):
        fe = _compose()["services"]["frontend"]
        self.assertTrue(any("443" in str(p) for p in fe["ports"]))


class TestOllamaWiring(unittest.TestCase):
    def test_api_services_point_at_ollama_service(self):
        s = _compose()["services"]
        for name in API_SERVICES:
            with self.subTest(service=name):
                self.assertIn("ollama:11434", s[name]["environment"]["OLLAMA_HOST"])

    def test_does_not_interpolate_the_host_ollama_host_variable(self):
        """Bẫy thật: .env ở gốc đã có OLLAMA_HOST=http://localhost:11434
        (đúng cho việc chạy script thẳng trên máy). Compose đọc CHÍNH file
        đó để nội suy ${...}, nên `${OLLAMA_HOST:-http://ollama:11434}` sẽ
        không bao giờ dùng tới giá trị mặc định - nó lấy 'localhost', mà
        trong container localhost là chính container đó.

        Phải dùng tên biến RIÊNG (OLLAMA_HOST_DOCKER) để hai ngữ cảnh
        không giẫm lên nhau."""
        # Bỏ dòng comment trước khi kiểm: comment giải thích ngay trong
        # docker-compose.yml có TRÍCH DẪN đúng cái mẫu sai này để cảnh báo
        # người đọc, nên so khớp trên văn bản thô sẽ báo động giả.
        code = "\n".join(ln for ln in COMPOSE.read_text(encoding="utf-8").splitlines()
                         if not ln.strip().startswith("#"))
        self.assertNotIn("${OLLAMA_HOST:", code,
                         "nội suy ${OLLAMA_HOST} sẽ nhặt phải localhost từ .env gốc")
        self.assertIn("${OLLAMA_HOST_DOCKER:-http://ollama:11434}", code)

    def test_root_env_example_still_documents_host_side_ollama(self):
        """OLLAMA_HOST trong .env gốc vẫn phải là localhost - nó phục vụ
        việc chạy scripts/*.py thẳng trên máy, KHÔNG phải container."""
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("OLLAMA_HOST=http://localhost:11434", text)

    def test_model_volume_persists_so_pull_happens_once(self):
        oll = _compose()["services"]["ollama"]
        self.assertIn("ollama_models:/root/.ollama", oll["volumes"])
        self.assertIn("ollama_models", _compose()["volumes"])

    def test_no_service_depends_on_ollama(self):
        """Có `depends_on: ollama` thì KHÔNG tắt được container ollama để
        chuyển sang Ollama.app native - Compose luôn kéo nó dậy, làm hỏng
        `docker compose up -d --scale ollama=0`.

        Bỏ được vì API không cần LLM lúc khởi động: 4/5 nhánh câu hỏi
        (DAG/whatif/causal/compare) không gọi LLM, còn nhánh RAG gọi qua
        HTTP nên Ollama lên muộn vẫn được."""
        s = _compose()["services"]
        for name in API_SERVICES:
            with self.subTest(service=name):
                self.assertNotIn("ollama", s[name].get("depends_on", {}))

    def test_switching_to_native_ollama_is_documented_and_real(self):
        """Từng ghi nhầm `docker compose --profile native-ollama up` trong
        comment nhưng KHÔNG hề tạo profile đó - hướng dẫn chạy sẽ lỗi.
        Khoá lại: cách chuyển sang native phải là cách THẬT SỰ chạy được."""
        raw = COMPOSE.read_text(encoding="utf-8")
        self.assertNotIn("--profile native-ollama", raw,
                         "profile này không tồn tại, đừng hướng dẫn dùng nó")
        self.assertIn("--scale ollama=0", raw)


class TestNginxConf(unittest.TestCase):
    def setUp(self):
        self.conf = NGINX_CONF.read_text(encoding="utf-8")

    def test_sse_buffering_disabled(self):
        """Lỗi #4: bật đệm thì SSE gom hết rồi trả một lượt - mất sạch ý
        nghĩa của streaming mà _proxy_sse() trong gateway đã cẩn thận giữ."""
        self.assertRegex(self.conf, r"proxy_buffering\s+off")

    def test_proxies_to_gateway_internal_port(self):
        self.assertIn("server gateway:8100", self.conf)

    def test_root_level_endpoints_have_explicit_locations(self):
        """/health và /__devreload không có tiền tố /api nên nếu không khai
        riêng, nginx sẽ đi tìm FILE tên 'health' rồi trả 404."""
        self.assertIn("location = /health", self.conf)
        self.assertIn("location = /__devreload", self.conf)

    def test_spa_fallback_present(self):
        self.assertIn("try_files", self.conf)
        self.assertIn("/index.html", self.conf)

    def test_nginx_timeout_not_shorter_than_gateway(self):
        """Gateway có request_timeout_s = 180. Nếu nginx timeout NGẮN hơn
        thì lỗi hiện ra là 504 của nginx, che mất lỗi thật của backend."""
        m = re.search(r"proxy_read_timeout\s+(\d+)s", self.conf)
        self.assertIsNotNone(m, "thiếu proxy_read_timeout")
        self.assertGreaterEqual(int(m.group(1)), 180)


class TestDockerignore(unittest.TestCase):
    def test_heavy_dirs_excluded_from_build_context(self):
        """Không có các dòng này thì MỖI lệnh build gửi ~6GB
        (.venv 2.0G + data/raw 2.8G + api/.venv 1.1G) lên Docker daemon."""
        text = DOCKERIGNORE.read_text(encoding="utf-8")
        for pattern in [".venv/", "api/.venv/", "data/raw/"]:
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, text)

    def test_secrets_excluded_from_image_layers(self):
        """Layer image đọc lại được kể cả khi file bị xoá ở layer sau."""
        text = DOCKERIGNORE.read_text(encoding="utf-8")
        for pattern in ["api/.env", "*.pem"]:
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, text)


if __name__ == "__main__":
    unittest.main()
