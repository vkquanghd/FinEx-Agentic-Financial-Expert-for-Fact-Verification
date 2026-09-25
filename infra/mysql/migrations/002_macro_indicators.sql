-- DEPRECATED 2026-07-26 - superseded by infra/postgres/migrations/001_init_schema.sql
-- (single consolidated Postgres database - see docs/PROGRESS.md). Kept for
-- historical reference only, no longer applied by docker-compose.yml.
--
-- Macro/national-economy indicators (GDP, CPI, PPI, PMI, IIP by industry,
-- etc.) from Vietstock's "Du lieu vi mo" exports. Decoupled from `companies`
-- (no ticker FK) since these are national/industry aggregates that apply
-- the same value across all companies at a given period - used as (a)
-- optional DAG-queryable facts, (b) covariates in src/causal/treatment_effect.py,
-- (c) real inputs for src/structured/calculator.py what-if scenarios.
--
-- NOTE: this file is only auto-applied by docker-entrypoint-initdb.d on a
-- FRESH volume. If your DB already exists (it does, per Step 1), run this
-- by hand:
--   docker exec -i thesis_mysql mysql -u thesis_user -p financial_db < infra/mysql/migrations/002_macro_indicators.sql
-- (or paste the CREATE TABLE below into MySQL Workbench with financial_db selected)
-- and separately widen ingestion_log.file_type (also can't be altered by
-- rerunning init scripts on a non-empty volume):
--   ALTER TABLE ingestion_log MODIFY COLUMN file_type
--       ENUM('bao_cao_tai_chinh','chi_so_tai_chinh','du_lieu_doanh_nghiep','pdf_disclosure','du_lieu_vi_mo') NOT NULL;
--
-- If you already ran an earlier version of this file (period_type missing
-- 'DAILY' and/or 'YTD' - 'DAILY' is for exchange-rate/interest-rate exports
-- that report a value per calendar day (e.g. "TỶ GIÁ TRUNG TÂM USD/VND");
-- 'YTD' is for state-budget exports that report a YEAR-TO-DATE CUMULATIVE
-- figure at 3/6/9/12 months (e.g. "9 Tháng/2021" = cumulative through 9
-- months of 2021 - a different thing from a single month's value, hence a
-- separate period_type rather than silently relabeling it as a quarter),
-- widen it:
--   ALTER TABLE macro_indicators MODIFY COLUMN period_type
--       ENUM('QUARTER','MONTH','YEAR','DAILY','YTD') NOT NULL;
--
-- If you already ran an earlier version with `unit VARCHAR(20)` (too short -
-- e.g. 'Số bé trai/100 bé gái' = 22 chars, causes error 1406 Data too long),
-- widen it:
--   ALTER TABLE macro_indicators MODIFY COLUMN unit VARCHAR(50);

CREATE TABLE IF NOT EXISTS macro_indicators (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    category        VARCHAR(50)  NOT NULL,   -- 'CPI','PPI','GDP_REAL_GROWTH','IIP_BY_INDUSTRY', etc.
    indicator_name  VARCHAR(255) NOT NULL,   -- row label as reported, e.g. 'Sản xuất kim loại'
    unit            VARCHAR(50),
    period_type     ENUM('QUARTER','MONTH','YEAR','DAILY','YTD') NOT NULL,
    fiscal_year     SMALLINT     NOT NULL,
    period          VARCHAR(10)  NOT NULL,   -- 'Q1'..'Q4' / 'M1'..'M12' / 'YEAR' / 'DmmDD' / 'YTD3'..'YTD12'
    value           DECIMAL(24,4),
    source_file     VARCHAR(255),
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_macro (category, indicator_name, period_type, fiscal_year, period)
);
