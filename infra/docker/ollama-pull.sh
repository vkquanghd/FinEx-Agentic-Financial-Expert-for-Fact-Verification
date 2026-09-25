#!/bin/sh
# Tự tải model về lần đầu, rồi giữ server chạy (2026-08-20, thêm tự thử lại
# + tự dọn file tạm hỏng ngày 2026-08-20 buổi chiều sau khi gặp lỗi thật).
#
# Vì sao cần script thay vì để `ollama serve` không: image ollama/ollama
# khởi động với KHÔNG có model nào. Không có bước này thì mọi câu hỏi đi
# vào nhánh LLM sẽ lỗi "model not found", và người dùng phải nhớ chạy tay
# `docker compose exec ollama ollama pull qwen3:8b` sau MỖI lần dựng lại
# volume - đúng thứ mà yêu cầu "tự động hết trên docker" muốn tránh.
#
# Tải về nằm trong named volume `ollama_models`, nên chỉ tốn mạng LẦN ĐẦU.
# `docker compose down` (không có -v) giữ nguyên volume; chỉ `down -v` mới
# xoá và bắt tải lại.
#
# --- BUG THẬT ĐÃ GẶP (2026-08-20) VÀ VÌ SAO CÓ BẢN SỬA NÀY ---
# `ollama pull` bị dừng giữa chừng nhiều lần (`docker compose down` trong
# lúc đang tải) làm lệch sổ sách file tạm mà nó dùng để nối lại chỗ dở
# dang. Lần tải sau đó luôn dừng ở đúng một lỗi:
#     Error: remove .../blobs/sha256-...-partial-0: no such file or directory
# Script CŨ không có gì bắt lỗi này: `set -e` làm nó thoát ngay, container
# chết theo (đây là tiến trình PID 1), `restart: unless-stopped` khởi động
# lại, script chạy lại từ đầu, gặp ĐÚNG lỗi đó lần nữa - một vòng lặp crash
# âm thầm, nhìn ngoài chỉ thấy container cứ "Restarting" liên tục mà không
# rõ vì sao.
#
# Bản sửa: thử lại có giới hạn, và trước mỗi lần thử lại, XOÁ CÁC FILE
# TẠM CÒN SÓT ("*-partial-*") - an toàn vì chúng chỉ là phần TẢI DỞ của
# một blob, không phải blob đã hoàn chỉnh; xoá chúng chỉ khiến Ollama tải
# lại đúng phần đó, không mất phần đã xong. Sau MAX_ATTEMPTS lần vẫn lỗi,
# in hướng dẫn xử lý rõ ràng rồi thoát - không lặp vô hạn trong im lặng.
set -u

MODEL="${OLLAMA_MODEL:-qwen3:8b}"
BLOBS_DIR="${OLLAMA_BLOBS_DIR:-/root/.ollama/models/blobs}"
MAX_ATTEMPTS="${OLLAMA_PULL_MAX_ATTEMPTS:-5}"

# serve chạy nền vì `ollama pull` cần nói chuyện với server đang chạy.
ollama serve &
SERVE_PID=$!

# Chờ server sẵn sàng. Không dùng `sleep 5` cố định: máy chậm thì 5 giây
# chưa đủ, máy nhanh thì phí 5 giây mỗi lần khởi động.
echo "[ollama-pull] Đợi ollama serve sẵn sàng..."
i=0
until ollama list >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 60 ]; then
        echo "[ollama-pull] LỖI: ollama serve không sẵn sàng sau 60 giây." >&2
        exit 1
    fi
    sleep 1
done

# Chỉ pull nếu CHƯA có - `ollama pull` trên model đã có vẫn gọi mạng để
# kiểm tra bản mới, làm chậm khởi động và hỏng khi offline.
if ollama list | grep -q "^${MODEL%%:*}"; then
    echo "[ollama-pull] Đã có model $MODEL, bỏ qua bước tải."
else
    echo "[ollama-pull] Đang tải $MODEL (lần đầu ~5GB, các lần sau dùng lại volume)..."

    attempt=1
    ok=0
    while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
        echo "[ollama-pull] Lần thử $attempt/$MAX_ATTEMPTS..."
        if ollama pull "$MODEL"; then
            ok=1
            break
        fi

        echo "[ollama-pull] Lần $attempt lỗi. Dọn file tạm còn sót (an toàn - chỉ" \
             "xoá phần TẢI DỞ, không đụng phần đã tải xong) rồi thử lại..." >&2
        # Không cần lọc theo thời gian: script này chạy TUẦN TỰ (một lần
        # `ollama pull` chạy xong hẳn - thành công hoặc thất bại - rồi mới
        # tới lượt dọn dẹp), không có tiến trình pull nào khác chạy song
        # song trong CÙNG container tại thời điểm này để lỡ xoá nhầm.
        find "$BLOBS_DIR" -name '*-partial-*' -delete 2>/dev/null || true

        attempt=$((attempt + 1))
        # Nghỉ tăng dần (2s, 4s, 6s...) - đủ để một lần "stalled" thoáng
        # qua của mạng tự hết mà không đợi quá lâu giữa các lần thử.
        sleep $((attempt * 2))
    done

    if [ "$ok" -ne 1 ]; then
        echo "[ollama-pull] LỖI: tải $MODEL thất bại sau $MAX_ATTEMPTS lần thử." >&2
        echo "[ollama-pull] Đây thường là do mạng không ổn định, không phải lỗi cấu hình." >&2
        echo "[ollama-pull] Xoá cache và tải lại một lần liền mạch (đừng 'docker compose down'" >&2
        echo "[ollama-pull]   giữa chừng khi đang tải - đó là nguyên nhân gây lỗi này):" >&2
        echo "[ollama-pull]     docker compose down" >&2
        echo "[ollama-pull]     docker volume rm thesis_project_ollama_models" >&2
        echo "[ollama-pull]     docker compose up -d" >&2
        exit 1
    fi
fi

echo "[ollama-pull] Sẵn sàng. Model: $MODEL"

# Chuyển quyền điều khiển lại cho tiến trình serve - `wait` giữ container
# sống và chuyển tín hiệu dừng đúng cách khi `docker compose down`.
wait "$SERVE_PID"
