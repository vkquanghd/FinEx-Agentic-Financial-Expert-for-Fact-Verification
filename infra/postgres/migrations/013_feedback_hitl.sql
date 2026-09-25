-- Migration 013 - Human-in-the-Loop feedback store ("013-mini" from
-- docs/Thesis_Project_DSA_v5_2026-08-03.pdf section 6, the thesis-scope
-- subset of Dataflow FINAL's migrations 019 (HITL) + 023 (answer/evidence)).
--
-- WHY A SEPARATE SCHEMA (not just more tables in public): the feedback loop
-- is the piece most likely to move to its own database when a real backend
-- exists (multi-user, PII-adjacent free text, different retention policy).
-- CREATE SCHEMA feedback gives that boundary TODAY at zero operational cost:
-- separate GRANTs possible, separate pg_dump possible, and the eventual move
-- to its own instance is one dump/restore with no renames. This is the
-- "chua cho cho backend" (reserve room for the backend) decision from DSA v5.
--
-- RESEARCH BASIS for the loop design (full citations to add to the thesis
-- reference list, Australian Harvard style like Master FINAL Part III):
--
--   * Wu, X., Xiao, L., Sun, Y., Zhang, J., Ma, T. & He, L. 2022, 'A survey
--     of human-in-the-loop for machine learning', Future Generation Computer
--     Systems, vol. 135, pp. 364-381. Their taxonomy splits HITL into
--     (1) data-processing improvement, (2) interventional training,
--     (3) system-independent HITL design. THIS schema implements (1)+(3):
--     ratings are collected primarily as an EVALUATION signal and a
--     review-prioritization signal - they only become TRAINING data after
--     explicit human review, never automatically.
--
--   * Settles, B. 2009, 'Active learning literature survey', Computer
--     Sciences Technical Report 1648, University of Wisconsin-Madison.
--     Uncertainty sampling: humans should spend review effort on the items
--     the system is most uncertain/wrong about. review_queue_lite encodes
--     exactly that: low rating, router fallback events, and low retrieval
--     similarity are the uncertainty proxies that auto-enqueue items.
--
--   * Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S. &
--     Amodei, D. 2017, 'Deep reinforcement learning from human preferences',
--     in Advances in Neural Information Processing Systems 30. Human
--     preference signals are aggregated and used in BATCHES to fit a
--     quality/reward model - not applied per-label to the policy. Our
--     retrain_gate_log implements the same batch discipline: individual
--     ratings NEVER trigger training; a weekly aggregate either passes the
--     gate or keeps accumulating (also matches Master FINAL muc 19: "Khong
--     fine-tune/deploy sau tung nhan moi" - oscillation / catastrophic
--     forgetting / no-validation risks).
--
-- APPEND-ONLY DISCIPLINE: answer_runs, answer_feedback and retrain_gate_log
-- have NO update trigger on purpose - matching causal_experiment_runs
-- (migration 009). A rating is an observation at a point in time; corrections
-- happen by inserting a new row, never editing history. review_queue_lite is
-- the ONE mutable table (status transitions pending->reviewed->resolved),
-- because it is a WORK QUEUE, not a log - same distinction Dataflow FINAL
-- draws between review_items (mutable workflow) and review_decisions
-- (append-only judgments).
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/013_feedback_hitl.sql

CREATE SCHEMA IF NOT EXISTS feedback;

-- One row per question the demo answered - the audit log of every chat turn.
-- This is Dataflow FINAL's answer_runs (trang 31) in mini form: question/
-- answer hashes for dedup+integrity, evidence JSONB standing in for the
-- future answer_evidence_links table (its keys: chunk_ids[], prediction_id,
-- experiment_run_id, sql_text - so migrating to real link rows later is a
-- mechanical JSONB unnest, no information loss).
CREATE TABLE IF NOT EXISTS feedback.answer_runs (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          VARCHAR(64),              -- Streamlit session; groups a conversation
    asked_at            TIMESTAMPTZ  NOT NULL DEFAULT now(),
    question_text       TEXT         NOT NULL,
    question_hash       CHAR(64)     NOT NULL,    -- SHA-256 of normalized question (project-wide hash convention)
    resolved_question   TEXT,                     -- after RQ7 context resolver, NULL if unchanged
    branch              VARCHAR(20)  NOT NULL
                        CHECK (branch IN ('dag','rag','whatif','causal','compare','absa',
                                          'predictive','unsupported')),
    fallback_from       VARCHAR(20),              -- e.g. 'dag' when DAG->RAG fallback fired (router S3 signal)
    latency_ms          INT,
    answer_text         TEXT,
    answer_hash         CHAR(64),
    evidence            JSONB,                    -- {chunk_ids:[], prediction_id:, experiment_run_id:, sql_text:}
    router_version      VARCHAR(100),             -- rules.py version label at answer time
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_answer_runs_asked_at  ON feedback.answer_runs (asked_at);
CREATE INDEX IF NOT EXISTS idx_answer_runs_branch    ON feedback.answer_runs (branch);
CREATE INDEX IF NOT EXISTS idx_answer_runs_session   ON feedback.answer_runs (session_id);

-- One row per rating action - INCLUDING skips. A skip ("bo qua") is data:
-- skip rate per branch is itself an engagement metric, and treating skip as
-- NULL-rating-with-flag (instead of just not inserting) means "user saw the
-- widget and declined" is distinguishable from "widget never shown" -
-- without this, the denominator for response-rate is unknowable.
CREATE TABLE IF NOT EXISTS feedback.answer_feedback (
    id             BIGSERIAL PRIMARY KEY,
    answer_run_id  BIGINT      NOT NULL REFERENCES feedback.answer_runs(id),
    rating         SMALLINT    CHECK (rating BETWEEN 1 AND 5),
    is_skip        BOOLEAN     NOT NULL DEFAULT FALSE,
    reason_code    VARCHAR(40)
                   CHECK (reason_code IS NULL OR reason_code IN
                          ('good','wrong_number','wrong_branch','missing_evidence',
                           'hallucination','outdated','unclear','too_slow','other')),
    free_text      TEXT,
    rated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- a row must be either a real rating or an explicit skip:
    CONSTRAINT chk_rating_or_skip CHECK (is_skip OR rating IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_answer_feedback_run ON feedback.answer_feedback (answer_run_id);

-- Uncertainty-sampling work queue (Settles 2009): what should the human look
-- at next. Rows are auto-enqueued by triggers-in-code (NOT db triggers - the
-- Streamlit app / Airflow DAG inserts them) when: rating <= 2, router
-- fallback fired, or retrieval similarity below threshold. Manual adds
-- allowed (source='manual'). THE ONE MUTABLE TABLE in this schema - it is a
-- queue, not a log. Maps 1:1 onto review_items when migration 019 lands.
CREATE TABLE IF NOT EXISTS feedback.review_queue_lite (
    id               BIGSERIAL PRIMARY KEY,
    answer_run_id    BIGINT REFERENCES feedback.answer_runs(id),   -- nullable: manual items may have no run
    source           VARCHAR(20) NOT NULL
                     CHECK (source IN ('low_rating','fallback','low_similarity','manual')),
    priority         SMALLINT    NOT NULL DEFAULT 3,               -- 1 = highest
    status           VARCHAR(20) NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending','reviewed','resolved','dismissed')),
    resolution_note  TEXT,                                          -- what the human decided/did
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_review_queue_pending
    ON feedback.review_queue_lite (priority, created_at) WHERE status = 'pending';

-- Append-only log of every retrain-gate evaluation (weekly, by the
-- thesis_feedback_aggregate Airflow DAG). The gate rule lives HERE as data,
-- not hard-coded in the DAG, so the thesis can show the full decision trail:
-- "week W: branch B had n=42 mean=3.1 -> gate not triggered (n < 50)".
-- Batch discipline per Christiano et al. 2017 / Master FINAL muc 19:
-- gate_triggered=TRUE creates a review_queue_lite item (source='manual',
-- priority=1) for a HUMAN to decide - it never starts training by itself.
CREATE TABLE IF NOT EXISTS feedback.retrain_gate_log (
    id              BIGSERIAL PRIMARY KEY,
    window_start    DATE        NOT NULL,
    window_end      DATE        NOT NULL,
    branch          VARCHAR(20) NOT NULL,
    n_feedback      INT         NOT NULL,
    n_skip          INT         NOT NULL DEFAULT 0,
    mean_rating     NUMERIC,
    skip_rate       NUMERIC,
    gate_rule       VARCHAR(200) NOT NULL,   -- rule text evaluated, e.g. 'n>=50 AND mean<3.5'
    gate_triggered  BOOLEAN     NOT NULL,
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_retrain_gate_log_branch
    ON feedback.retrain_gate_log (branch, window_end);


-- ============================================================================
-- 2026-09-22: mở rộng CHECK reason_code. CÙNG loại bug mà 014 được viết ra để
-- sửa (xem đầu file 014_answer_runs_branch_and_persona.sql), chỉ khác cột.
--
-- === Bug THẬT, đang sống, phát hiện 2026-09-22 ===
-- CHECK inline ở CREATE TABLE phía trên cho phép 9 mã:
--   good, wrong_number, wrong_branch, missing_evidence, hallucination,
--   outdated, unclear, too_slow, other
-- Nhưng dropdown "Sai ở đâu?" trên giao diện (api/frontend/app.js's
-- _reasonsVi/_reasonsEn) lại đưa ra 3 mã KHÔNG có trong danh sách đó:
--   wrong_entity, wrong_period, missing_context
-- Hậu quả: người dùng chấm <=3 sao rồi chọn đúng một trong 3 lý do đó ->
-- INSERT vi phạm CHECK -> log_feedback() ném IntegrityError ->
-- api/services/feedback/main.py bắt và trả HTTP 502 "Không ghi được đánh
-- giá" -> MẤT TRẮNG tín hiệu đánh giá đó. Đây là 3/9 lựa chọn trong dropdown,
-- tức là khoảng một phần ba số lần người dùng chịu khó nói rõ SAI Ở ĐÂU -
-- đúng loại phản hồi giá trị nhất - thì lại là loại chắc chắn hỏng.
-- (Pydantic ReasonCode trong api/shared/schemas.py khớp với dropdown chứ
-- không khớp với DB, nên tầng API cho qua và lỗi chỉ nổ ở tầng DB.)
--
-- === Thêm mới: 'implicit' ===
-- 2026-09-22 bỏ cơ chế bắt buộc đánh giá trước khi hỏi tiếp (trước đây khoá
-- ô nhập sau mỗi câu trả lời). Lý lẽ phương pháp của 013 KHÔNG đổi - "Bỏ qua
-- CŨNG là dữ liệu, không có nó thì mọi thống kê rating đều là mẫu tự chọn"
-- (xem comment của chính bảng này phía trên) - nên khi người dùng hỏi câu
-- tiếp mà chưa chấm, frontend vẫn ghi đủ một bản ghi is_skip, với
-- reason_code='implicit'.
-- Cố ý TÁCH 'implicit' khỏi lượt bỏ qua chủ động (reason_code NULL): một cú
-- bấm "Bỏ qua" BỊ ÉP không mang cùng thông tin với việc người dùng tự nhiên
-- lướt qua. Gộp hai thứ này lại là mất một biến phân tích thật.
--
-- Giữ NGUYÊN cả 9 mã cũ (kể cả 'good'/'missing_evidence'/'outdated' hiện
-- không còn trên dropdown) - các dòng đã ghi trước đây phải tiếp tục hợp lệ;
-- một CHECK mới mà làm dữ liệu cũ thành không hợp lệ là một migration hỏng.
--
-- Idempotent (DROP CONSTRAINT IF EXISTS) - đúng quy ước mọi migration trong
-- dự án này. Thêm vào ĐÂY chứ không tạo file số mới, theo đúng tiền lệ 014
-- đã ghi rõ: số 015+ đã dành cho roadmap Master FINAL (xem README.md).
ALTER TABLE feedback.answer_feedback
    DROP CONSTRAINT IF EXISTS answer_feedback_reason_code_check;

ALTER TABLE feedback.answer_feedback
    ADD CONSTRAINT answer_feedback_reason_code_check CHECK (reason_code IS NULL OR reason_code IN (
        -- 9 mã gốc của 013 (giữ nguyên để dòng cũ vẫn hợp lệ)
        'good', 'wrong_number', 'wrong_branch', 'missing_evidence',
        'hallucination', 'outdated', 'unclear', 'too_slow', 'other',
        -- 3 mã dropdown vẫn đưa ra nhưng CHECK cũ chặn (bug ở trên)
        'wrong_entity', 'wrong_period', 'missing_context',
        -- bỏ qua ngầm: hỏi câu tiếp mà chưa chấm
        'implicit'
    ));

COMMENT ON COLUMN feedback.answer_feedback.reason_code IS
    'Lý do chấm điểm thấp, lấy từ dropdown ở giao diện. '
    '''implicit'' = bỏ qua NGẦM (người dùng hỏi câu tiếp mà chưa chấm) - '
    'khác NULL + is_skip=true, vốn là lượt CHỦ ĐỘNG bấm "Bỏ qua".';
