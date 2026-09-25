-- 002_chunk_text_fts.sql (2026-08-06)
--
-- GIN index phục vụ nhánh lexical (sparse) của hybrid_search() trong
-- src/retrieval/vector_store.py - xem comment ở LEXICAL_SQL trong file đó
-- cho lý do thiết kế (RRF hybrid: Cormack et al. SIGIR 2009; Blended RAG
-- arXiv:2404.07220) và vì sao dùng config 'simple' cho tiếng Việt (không
-- có stemmer tiếng Việt trong Postgres core; 'simple' giữ dấu + lowercase,
-- đủ cho khớp-nguyên-văn tên kiểm toán/con số/tên riêng).
--
-- KHÔNG bắt buộc phải chạy trước khi dùng hybrid_search - các LEXICAL_SQL_*
-- dùng biểu thức to_tsvector('simple', chunk_text) nên vẫn chạy đúng bằng
-- seq scan khi chưa có index; migration này chỉ làm nó nhanh (đặc biệt cho
-- truy vấn KHÔNG lọc ticker, quét cả ~18k+ dòng).
--
-- Expression index (không thêm cột generated): tránh ALTER TABLE ghi lại
-- toàn bộ bảng đang có embedding nặng, và biểu thức trong index khớp đúng
-- nguyên văn biểu thức trong LEXICAL_SQL_* nên planner dùng được ngay.

CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_fts
    ON disclosure_chunks
    USING gin (to_tsvector('simple', chunk_text));
