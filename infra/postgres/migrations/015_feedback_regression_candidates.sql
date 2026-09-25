-- Migration 015 - curated regression candidates from reviewed feedback.
--
-- Ratings and a review-queue status alone do not improve FinEx: neither is
-- an executable expected result. This table records the human-approved
-- expected answer for a faulty/uncertain run, so it can later become a
-- fixed regression test. It deliberately does NOT trigger fine-tuning.
--
-- Run against an existing database:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/015_feedback_regression_candidates.sql

CREATE TABLE IF NOT EXISTS feedback.regression_candidates (
    id                    BIGSERIAL PRIMARY KEY,
    review_id             BIGINT NOT NULL UNIQUE REFERENCES feedback.review_queue_lite(id),
    answer_run_id         BIGINT NOT NULL REFERENCES feedback.answer_runs(id),
    question_text         TEXT NOT NULL,
    expected_answer       TEXT NOT NULL,
    expected_sources      TEXT,
    reviewer_note         TEXT,
    status                VARCHAR(20) NOT NULL DEFAULT 'candidate'
                          CHECK (status IN ('candidate', 'accepted', 'rejected')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at            TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_regression_candidates_status
    ON feedback.regression_candidates (status, created_at DESC);

COMMENT ON TABLE feedback.regression_candidates IS
    'Human-curated expected answers promoted from review items. These are regression-test candidates, not automatic training data.';


-- ============================================================================
-- 2026-09-22: provenance + tách vai (separation of duties).
--
-- === Vấn đề: bộ fixture có thể bị nhiễu mà KHÔNG truy được, và người viết
-- === đáp án tự duyệt được đáp án của chính mình ===
--
-- Bảng này sinh ra ground truth: mỗi dòng 'accepted' được
-- scripts/export_feedback_regression_candidates.py ghi thành một case trong
-- data/eval/feedback_regression.jsonl, và case đó về sau được chạy lại để
-- KẾT LUẬN hệ thống đúng hay sai. Nói cách khác, nội dung bảng này có thẩm
-- quyền cao hơn câu trả lời của hệ thống. Nhưng trước migration này:
--
--   1. KHÔNG ghi ai viết expected_answer, KHÔNG ghi ai bấm Accept. Nếu một
--      người nhập bừa (hoặc nhập nhầm - expected_answer không khớp
--      question_text, chuyện rất dễ xảy ra vì hai trường này đến từ hai
--      nguồn khác nhau: question_text lấy tự động từ answer_runs, còn
--      expected_answer do người gõ tay), thì không có cách nào lọc riêng
--      phần của người đó - chỉ còn cách nghi ngờ TOÀN BỘ bộ fixture.
--      Trớ trêu: migration 014 đã thêm persona/username vào answer_runs
--      ĐÚNG cho loại truy vết này; 015 viết sau lại không mang theo.
--
--   2. Mọi tài khoản trong ACCOUNTS (api/services/gateway/auth.py) đều có
--      role='user', và ROLE_FOR_SERVICE cho role='user' gọi được TOÀN BỘ
--      /api/feedback/* - tức là cùng một người gọi được cả
--      POST /review/promote (viết đáp án) lẫn
--      POST /review/regression/{id}/decision (duyệt đáp án đó).
--      Thiết kế 3 tầng của 013 trông như có hai lớp kiểm tra, nhưng đó là
--      hai BƯỚC, không phải hai NGƯỜI. Một người tự duyệt chính mình thì
--      lớp thứ hai không kiểm tra thêm được gì.
--
-- === Cách sửa ===
-- Ghi danh tính ở CẢ HAI bước, rồi chặn ở tầng ứng dụng khi promoted_by ==
-- decided_by (api/services/feedback/main.py). KHÔNG đặt CHECK constraint
-- cho luật đó: các dòng đã ghi trước migration này có promoted_by NULL, một
-- CHECK sẽ làm chúng thành không hợp lệ - và một migration làm hỏng dữ liệu
-- cũ là một migration sai. Luật tách vai cũng là quyết định NGHIỆP VỤ, có
-- thể phải nới (vd một người duy nhất dùng hệ thống lúc phát triển), nên để
-- ở tầng ứng dụng nơi nới được mà không cần migration.
--
-- persona ghi RIÊNG khỏi username, đúng phân biệt mà 014 đã thiết lập:
-- username = danh tính đăng nhập (truy vết), persona = vai trò nghiệp vụ
-- (phân tích: "auditor viết expected answer có khác investor không?").
--
-- NULLABLE có chủ đích: dòng ghi trước 2026-09-22 không có thông tin này và
-- KHÔNG được phép bịa - NULL = "không biết", trung thực hơn giá trị mặc định.
--
-- Idempotent (ADD COLUMN IF NOT EXISTS) - chạy lại an toàn.
ALTER TABLE feedback.regression_candidates
    ADD COLUMN IF NOT EXISTS promoted_by       VARCHAR(64);

ALTER TABLE feedback.regression_candidates
    ADD COLUMN IF NOT EXISTS promoted_persona  VARCHAR(32);

ALTER TABLE feedback.regression_candidates
    ADD COLUMN IF NOT EXISTS decided_by        VARCHAR(64);

ALTER TABLE feedback.regression_candidates
    ADD COLUMN IF NOT EXISTS decided_persona   VARCHAR(32);

COMMENT ON COLUMN feedback.regression_candidates.promoted_by IS
    'Tài khoản đã VIẾT expected_answer (POST /review/promote). '
    'NULL = dòng ghi trước 2026-09-22 hoặc gọi thẳng service không qua gateway.';

COMMENT ON COLUMN feedback.regression_candidates.decided_by IS
    'Tài khoản đã DUYỆT (accept/reject). Phải KHÁC promoted_by - tách vai, '
    'ép ở tầng ứng dụng (api/services/feedback/main.py), không phải CHECK, '
    'xem phần giải thích ở trên.';

COMMENT ON COLUMN feedback.regression_candidates.promoted_persona IS
    'Vai trò NGHIỆP VỤ của người viết đáp án (analyst/auditor/executive/'
    'investor) - xem src/core/personas.py. KHÁC role kiểm soát truy cập.';

-- Truy vết nhanh "những dòng nào do tài khoản X viết" khi cần lọc bỏ đóng
-- góp của một người review bị phát hiện nhập ẩu.
CREATE INDEX IF NOT EXISTS idx_regression_candidates_promoted_by
    ON feedback.regression_candidates (promoted_by);


-- ============================================================================
-- 2026-09-22 (phần 2): nhật ký quyết định CHỈ-GHI-THÊM + cho phép đảo ngược.
--
-- === Vấn đề 1: bảng này đang PHÁ nguyên tắc của chính dự án ===
-- Migration 013 tuyên bố rõ: "review_queue_lite là bảng DUY NHẤT được sửa
-- trạng thái trong schema feedback - mọi bảng còn lại chỉ ghi thêm
-- (append-only), để dấu vết đánh giá không bao giờ bị viết đè."
-- Nhưng regression_candidates (viết sau, ở chính file này) lại là bảng thứ
-- HAI bị sửa đè: decide_regression_candidate() UPDATE thẳng status/
-- decided_by/decided_at lên dòng đó, và reviewer_note còn bị COALESCE ghi
-- chồng. Hậu quả: chỉ đọc được TRẠNG THÁI CUỐI, không bao giờ đọc được
-- DIỄN BIẾN. Không trả lời được câu "ai đã duyệt cái này, lúc nào, và đã có
-- ai lật lại chưa" - đúng loại câu hỏi mà một bảng sinh ra ground truth
-- bắt buộc phải trả lời được.
--
-- === Vấn đề 2: duyệt sai là khoá vĩnh viễn ===
-- decide_regression_candidate() ném lỗi với mọi dòng đã rời trạng thái
-- 'candidate'. Nên nếu người duyệt bấm Accept nhầm một đáp án rác (vd
-- expected_answer không khớp question_text - rất dễ xảy ra vì hai trường
-- này đến từ hai nguồn khác nhau), dòng đó nằm vĩnh viễn trong fixture và
-- KHÔNG có đường nào gỡ ra qua giao diện. Một hệ thống chống nhiễu mà
-- không sửa được cái nhiễu đã lọt thì mới đi được nửa đường.
--
-- === Cách sửa ===
-- Bảng dưới đây ghi MỌI thao tác quyết định thành một dòng MỚI, không bao
-- giờ UPDATE/DELETE. Trạng thái hiện tại vẫn đọc từ regression_candidates
-- (nhanh, đơn giản); còn LỊCH SỬ thì nằm ở đây và không ai viết đè được.
-- Hai thứ khác nhau, cố ý tách: một cái trả lời "bây giờ thế nào", cái kia
-- trả lời "đã đi qua những gì".
--
-- Ai được đảo ngược: một tài khoản review KHÁC người đã duyệt (ép ở tầng
-- ứng dụng, cùng chỗ với luật tách vai). Admin KHÔNG đảo ngược - service
-- admin khai là read-only và giữ đúng như vậy; admin đọc được toàn bộ nhật
-- ký này nhưng không ghi gì.
--
-- KHÔNG có khoá ngoại tới answer_runs ở đây: dòng log phải sống sót kể cả
-- khi dữ liệu liên quan bị dọn - một nhật ký kiểm toán mà xoá theo dữ liệu
-- gốc thì không phải nhật ký kiểm toán.
CREATE TABLE IF NOT EXISTS feedback.regression_decisions (
    id              BIGSERIAL PRIMARY KEY,
    candidate_id    BIGINT      NOT NULL REFERENCES feedback.regression_candidates(id),
    -- 'accepted'/'rejected' = quyết định thường; 'reversed' = lật lại một
    -- quyết định trước đó. Tách 'reversed' ra thành nhãn riêng (thay vì chỉ
    -- ghi accepted/rejected lần nữa) để đếm được số lần phải sửa sai - một
    -- chỉ số CHẤT LƯỢNG của chính quy trình review, không phải chỉ số của
    -- hệ thống QA.
    decision        VARCHAR(20) NOT NULL
                    CHECK (decision IN ('accepted', 'rejected', 'reversed')),
    -- trạng thái TRƯỚC và SAU, để đọc được diễn biến mà không phải suy luận
    from_status     VARCHAR(20),
    to_status       VARCHAR(20) NOT NULL,
    actor           VARCHAR(64),      -- tài khoản thực hiện; NULL = gọi ngoài gateway
    actor_persona   VARCHAR(32),
    -- Bắt buộc về mặt tinh thần khi đảo ngược (ép ở tầng ứng dụng, không
    -- ép bằng CHECK vì quyết định thường không cần lý do): "vì sao lật lại"
    -- chính là phần giải trình được của thao tác này.
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_regression_decisions_candidate
    ON feedback.regression_decisions (candidate_id, created_at);

CREATE INDEX IF NOT EXISTS idx_regression_decisions_actor
    ON feedback.regression_decisions (actor, created_at DESC);

COMMENT ON TABLE feedback.regression_decisions IS
    'Nhật ký CHỈ-GHI-THÊM mọi quyết định accept/reject/đảo ngược trên '
    'regression_candidates. Không bao giờ UPDATE/DELETE - đây là dấu vết '
    'kiểm toán của các quyết định CON NGƯỜI sinh ra ground truth. Trạng '
    'thái hiện tại đọc ở regression_candidates; lịch sử đọc ở đây.';
