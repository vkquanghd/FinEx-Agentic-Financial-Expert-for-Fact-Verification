-- 014 (2026-08-07): mở rộng CHECK nhánh + thêm persona/username vào answer_runs.
--
-- HAI vấn đề, cùng một bảng, gộp một migration vì cùng chạm
-- feedback.answer_runs và cùng phải xong TRƯỚC khi bắt đầu thu HITL thật.
--
-- === Vấn đề 1: CHECK nhánh chặn log im lặng (BUG THẬT) ===
-- 013 khai CHECK (branch IN ('dag','rag','whatif','causal','compare','absa',
-- 'predictive','unsupported')). Nhưng route() (src/router/query_router.py)
-- thực tế trả về:
--   - 'auditor_lookup' (thêm 2026-08-07, tra cứu ngược hãng kiểm toán)
--   - 'dag_trend'      (thêm 2026-08-07, chuỗi nhiều năm)
--   - 'whatif_unsupported' / 'causal_unsupported' / 'compare_unsupported' /
--     'absa_unsupported' - CHECK cũ chỉ có 'unsupported' TRƠN, không khớp
--     nhãn thật nào trong 4 nhãn này (lỗi có từ 013, không phải mới)
-- Hậu quả: log_answer_run() INSERT vi phạm CHECK -> ném exception -> bị
-- _log_run() trong api/services/qa/main.py NUỐT (try/except trả None, cố ý
-- để lỗi ghi log không bao giờ phá câu trả lời đã tính xong) -> answer_run_id
-- = None -> frontend không hiện ô đánh giá -> 0 dữ liệu HITL cho các nhánh
-- đó, và không có tiếng động nào báo hiệu. Nếu bắt đầu thu HITL trước khi
-- sửa, bộ dữ liệu lệch ngay từ ngày đầu: các nhánh im lặng bị loại hoàn
-- toàn khỏi mọi thống kê, trong khi thống kê lại trông "bình thường".
--
-- VARCHAR(20) cũ cũng chật: 'compare_unsupported' đúng 19 ký tự (vừa đủ),
-- nhưng không còn chỗ cho nhãn dài hơn trong tương lai -> nới lên 32.
--
-- === Vấn đề 2: không có cách nào biết AI đánh giá (vai trò nào) ===
-- Dự án vừa thêm đăng nhập phân vai (2026-08-07) nhưng answer_runs không
-- ghi lại người/vai trò nào đã hỏi, nên không thể trả lời câu hỏi nghiên
-- cứu "vai trò nào thấy nhánh nào hữu ích" - trong khi đây chính là luận cứ
-- cho Adaptive Routing (router chỉ đáng giá khi MỘT người dùng hỏi cả câu
-- định lượng lẫn định tính trong cùng phiên).
--
-- Phân biệt hai khái niệm KHÁC NHAU, cố ý tách 2 cột:
--   - persona: vai trò NGHIỆP VỤ (analyst/auditor/executive/investor) - thứ
--     dùng để phân tích nghiên cứu.
--   - username: danh tính đăng nhập - giữ riêng vì một persona có thể có
--     nhiều tài khoản sau này, và để truy vết khi một chuỗi feedback trông
--     bất thường.
-- KHÔNG dùng lại 'role' của gateway (user/admin) - đó là vai trò KIỂM SOÁT
-- TRUY CẬP, không phải vai trò nghiệp vụ; gộp hai thứ này là nhầm lẫn kinh
-- điển và sẽ làm hỏng phân tích (mọi persona đều có role='user').
--
-- Cả hai cột NULLABLE có chủ đích: các lượt đã ghi trước migration này
-- không có thông tin đó và KHÔNG được phép bịa ra - NULL = "không biết",
-- trung thực hơn một giá trị mặc định giả.
--
-- Idempotent (chạy lại nhiều lần an toàn) - đúng quy ước mọi migration
-- trong dự án này, xem 010's DROP CONSTRAINT IF EXISTS.

-- --- Vấn đề 1: CHECK nhánh ---
ALTER TABLE feedback.answer_runs
    DROP CONSTRAINT IF EXISTS answer_runs_branch_check;

ALTER TABLE feedback.answer_runs
    ALTER COLUMN branch TYPE VARCHAR(32);

ALTER TABLE feedback.answer_runs
    ADD CONSTRAINT answer_runs_branch_check CHECK (branch IN (
        -- nhánh trả lời được
        'dag', 'dag_trend', 'rag', 'whatif', 'causal', 'compare', 'absa',
        'auditor_lookup',
        -- 2026-09-14: 2 nhánh mới thêm vào src/router/rules.py CÙNG NGÀY -
        -- 'dag_multi' (tra cứu 1 chỉ tiêu cho >=2 mã cùng lúc, sửa bug P3
        -- id=42 - âm thầm bỏ sót mã) và 'out_of_scope' (dự đoán giá tương
        -- lai / xin ý kiến bên thứ ba / xin tư vấn đầu tư - trả lời
        -- deterministic, không qua RAG/LLM). Thêm NGAY vào CHECK này (chứ
        -- không tạo migration số mới - xem README.md's bảng ánh xạ, số 015+
        -- đã dành riêng cho roadmap Master FINAL) để KHÔNG tái diễn đúng bug
        -- "CHECK thiếu nhãn -> INSERT bị nuốt lỗi -> answer_run_id=None ->
        -- mất ô đánh giá -> 0 dữ liệu HITL, hoàn toàn im lặng" mà migration
        -- này (014) BAN ĐẦU được viết ra để sửa (xem phần đầu file). File
        -- này vẫn idempotent (DROP CONSTRAINT IF EXISTS ở trên) - chạy lại
        -- an toàn dù đã áp dụng trước đó.
        'dag_multi', 'out_of_scope',
        -- 2026-09-15: 'ratio_lookup' (financial_ratios - P/E, ROE, current
        -- ratio, đòn bẩy...) - CÙNG loại bug thiếu-giá-trị-CHECK như 2 dòng
        -- trên, thêm ngay khi thêm nhánh mới vào rules.py, không đợi phát
        -- hiện qua lỗi INSERT im lặng.
        'ratio_lookup',
        -- 2026-09-15: 'bank_rate_lookup' (xếp hạng lãi suất tiền gửi theo
        -- ngân hàng, macro_indicators.DEPOSIT_RATES) - cùng loại bug CHECK
        -- thiếu giá trị như các dòng trên.
        'bank_rate_lookup',
        -- 2026-09-21: 'price_snapshot' (giá cổ phiếu EOD gần nhất qua
        -- company_price_snapshot, src/causal/market_factors.py's
        -- latest_price_snapshot() - câu hỏi "giá/cổ phiếu X hôm nay") -
        -- cùng loại bug CHECK thiếu giá trị như các dòng trên. Thêm NGAY
        -- khi thêm nhánh vào rules.py, đúng bài học của chính migration
        -- này (xem phần đầu file) - không đợi phát hiện qua INSERT bị nuốt
        -- lỗi.
        'price_snapshot',
        -- nhánh nhận diện được ý định nhưng thiếu dữ kiện để chạy
        'whatif_unsupported', 'causal_unsupported', 'compare_unsupported',
        'absa_unsupported',
        -- dự phòng cho Phase P2 (chưa nối vào router) + nhãn cũ của 013
        'predictive', 'unsupported'
    ));

-- fallback_from nhận cùng tập nhãn (ghi nhánh ĐẦU TIÊN được chọn khi có
-- fallback, vd 'dag' hoặc 'dag_trend' -> RAG). 013 để cột này không có
-- CHECK nên không cần sửa, chỉ nới độ dài cho khớp.
ALTER TABLE feedback.answer_runs
    ALTER COLUMN fallback_from TYPE VARCHAR(32);

-- --- Vấn đề 2: persona + username ---
ALTER TABLE feedback.answer_runs
    ADD COLUMN IF NOT EXISTS persona VARCHAR(32);

ALTER TABLE feedback.answer_runs
    ADD COLUMN IF NOT EXISTS username VARCHAR(64);

-- Không đặt CHECK cứng trên persona: danh sách persona là quyết định
-- NGHIỆP VỤ còn đang định hình (4 persona chốt 2026-08-07, có thể thêm khi
-- phỏng vấn thêm người dùng thật). Một CHECK ở đây sẽ biến mỗi lần thêm
-- persona thành một migration - trong khi giá trị sai duy nhất có thể xảy
-- ra là do chính code của ta gửi lên, đã kiểm soát ở src/core/personas.py.
COMMENT ON COLUMN feedback.answer_runs.persona IS
    'Vai trò NGHIỆP VỤ của người hỏi (analyst/auditor/executive/investor) - '
    'xem src/core/personas.py. KHÁC role kiểm soát truy cập (user/admin) '
    'của gateway. NULL = lượt ghi trước 2026-08-07 hoặc không xác định.';

COMMENT ON COLUMN feedback.answer_runs.username IS
    'Tài khoản đăng nhập đã hỏi câu này. NULL = lượt ghi trước 2026-08-07.';

CREATE INDEX IF NOT EXISTS idx_answer_runs_persona
    ON feedback.answer_runs (persona);
