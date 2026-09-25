/* Live-reload cho môi trường phát triển.
 *
 * Poll /__devreload (gateway) mỗi giây; khi mốc mtime đổi (nghĩa là
 * index.html/app.js/styles.css vừa được sửa) thì tự location.reload() -
 * khỏi phải bấm F5 tay mỗi lần code đổi.
 *
 * Không phải build tool, không WebSocket, chỉ một con số so sánh mỗi
 * giây - an toàn để luôn bật, kể cả lúc không ai đang sửa gì. Nếu gateway
 * đang tự khởi động lại (uvicorn --reload bắt được thay đổi ở
 * api/services/gateway/main.py), request sẽ lỗi tạm thời - bỏ qua, thử
 * lại giây sau, tự phục hồi khi gateway lên lại.
 */
(function () {
  let last = null;

  async function poll() {
    try {
      const r = await fetch("/__devreload", { cache: "no-store" });
      const { mtime } = await r.json();
      if (last !== null && mtime !== last) {
        location.reload();
        return; // không cần setTimeout nữa, trang sắp tải lại
      }
      last = mtime;
    } catch {
      // gateway đang restart hoặc mạng chập chờn - im lặng thử lại
    }
    setTimeout(poll, 1000);
  }

  poll();
})();
