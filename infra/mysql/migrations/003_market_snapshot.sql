-- DEPRECATED 2026-07-26 - superseded by infra/postgres/migrations/001_init_schema.sql
-- (single consolidated Postgres database - see docs/PROGRESS.md). Kept for
-- historical reference only, no longer applied by docker-compose.yml.
--
-- Daily "real-time market factor" data (Step 11 - thesis_market_snapshot DAG):
-- per-company daily price/volume, plus VN-Index/gold/forex/oil going into the
-- EXISTING macro_indicators table (period_type='DAILY', new MKT_* categories -
-- no schema change needed there, see scripts/market_snapshot.py).
--
-- Why a separate table for price instead of macro_indicators: macro_indicators
-- has no ticker column (national/industry aggregates only, see
-- 002_macro_indicators.sql) - per-company daily price is a different shape
-- (one row per ticker per day), so it gets its own table with a ticker FK,
-- same pattern as financial_line_items/financial_ratios/disclosures.
--
-- NOTE: this file is only auto-applied by docker-entrypoint-initdb.d on a
-- FRESH volume. If your DB already exists (it does, per Step 1), run this
-- by hand:
--   docker exec -i thesis_mysql mysql -u thesis_user -p financial_db < infra/mysql/migrations/003_market_snapshot.sql
-- and separately widen ingestion_log.file_type (also can't be altered by
-- rerunning init scripts on a non-empty volume):
--   ALTER TABLE ingestion_log MODIFY COLUMN file_type
--       ENUM('bao_cao_tai_chinh','chi_so_tai_chinh','du_lieu_doanh_nghiep','pdf_disclosure','du_lieu_vi_mo','market_snapshot') NOT NULL;

CREATE TABLE IF NOT EXISTS company_price_snapshot (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker          VARCHAR(10)  NOT NULL,
    trade_date      DATE         NOT NULL,
    open_price      DECIMAL(18,2),
    high_price      DECIMAL(18,2),
    low_price       DECIMAL(18,2),
    close_price     DECIMAL(18,2),
    volume          BIGINT,
    source          VARCHAR(20)  NOT NULL DEFAULT 'vnstock_vci',  -- data provider, in case a ticker's source ever changes (see vnstock VCI/KBS choice in market_snapshot.py)
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES companies(ticker),
    UNIQUE KEY uq_price (ticker, trade_date)
);

CREATE INDEX idx_price_trade_date ON company_price_snapshot (trade_date);
