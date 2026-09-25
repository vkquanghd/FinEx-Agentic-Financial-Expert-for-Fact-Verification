"""Nhánh B (Week 6 of the locked 10-week plan, RQ2 baseline) - agentic
Text-to-SQL: the LLM generates a raw SQL SELECT statement directly against
the PostgreSQL schema (migrated 2026-07-26 off MySQL, see docs/PROGRESS.md)
in natural language, WITHOUT going through Router or
schema.py's fixed alias table. This is the baseline the schema-constrained
DAG lookup (Nhánh A - src/structured/query_builder.py + schema.py) is
compared against for RQ2: "Định tuyến schema-constrained (Nhánh A, DAG
lookup) có chính xác và ít hallucination hơn agentic text-to-SQL tự do
(Nhánh B) không?"

Deliberately NOT wired into src/router/query_router.py - per
Thesis_Direction_Redesign.pdf section 7's baseline table, "Text-to-SQL
Agent-only (Nhánh B)" is its own standalone ablation baseline ("không qua
Router"), called directly by the Week 8 evaluation harness, not one of the
Router's live dispatch branches (dag/rag/whatif/causal/compare).

Academic basis (Thesis_Direction_Redesign.pdf section 6.4):
- "Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL"
  (arXiv:2406.08426) - LLM-generated SQL hallucinates most often on JOINs
  and on column/table names that don't actually exist in the target schema.
- "Before Generation, Align it!" (arXiv:2405.15307) - checking a generated
  query's referenced identifiers against the real schema BEFORE execution
  catches a meaningful share of hallucinations without needing to run (and
  fail on) the query first. _check_schema_alignment() below is a
  best-effort version of that idea - a regex-based identifier check, NOT a
  full SQL parser, consistent with this project's existing preference for
  transparent rule-based logic over a heavyweight new dependency (same
  choice already made for src/router/rules.py's classifier).

Every generated query passes through up to 3 gates before any real data
comes back, each measuring a DIFFERENT, separately-reportable failure mode
for RQ2 (see answer()'s `gate_failed` field):
  1. unsafe_statement - not a plain read-only SELECT (rejects
     INSERT/UPDATE/DELETE/DROP/ALTER/... outright). A safety guard first,
     not really a "hallucination" measurement - but worth tracking
     separately since the LLM attempting a WRITE is a more severe failure
     than a wrong column name.
  2. schema_alignment - _check_schema_alignment() catches a reference to a
     table/column that doesn't exist in the real schema, BEFORE execution.
  3. execution - gates 1-2 passed but Postgres itself still rejected the
     query at runtime (e.g. a hallucinated column the regex check missed,
     a genuine syntax error, a bad JOIN). This is the ground-truth signal
     schema_alignment is trying to anticipate - comparing how often
     failures land at gate 2 vs only surface at gate 3 is itself a usable
     RQ2 result (does the cheap pre-check actually help, and how much).

Known, real limitation of gate 2 (confirmed via testing, not just a
theoretical caveat): _check_schema_alignment() only validates QUALIFIED
references (`alias.column` / `table.column`, via _DOTTED_COL_RE) - a
hallucinated BARE column name with no table prefix (e.g. plain `SELECT
faulty_col FROM financial_line_items`) is NOT checked and will slip
through to gate 3, only caught by Postgres's own "column ... does not
exist" error at real execution (was MySQL's "Unknown column" error before
the 2026-07-26 migration - same blind spot, different error string).
Validating unqualified columns correctly would require
tracking which clause (SELECT/WHERE/GROUP BY/...) each bare identifier
appears in and resolving it against every table in scope - a real SQL
parser's job, not a regex's - so this gap is accepted rather than solved,
same "best-effort, not a full parser" scoping as the rest of this module.

**Gate 2b - AST-based cross-check (added 2026-07-26), OBSERVATIONAL ONLY,
does not change what stops execution.** `_check_schema_alignment_ast()`
uses `sqlglot` to parse the generated SQL into a real AST and check EVERY
column reference against SCHEMA, including bare/unqualified ones that gate
2's regex cannot see. This is deliberately wired in as a side-channel
measurement, not a replacement for gate 2: gate 2's bare-column blind spot
is exactly what produces this module's core RQ2 evidence (SQL that looks
syntactically fine, passes gate 2, and only fails at gate 3 with a real
Postgres error) - silently fixing that blind spot would change the very
phenomenon RQ2 is measuring and invalidate comparisons against data already
collected with the original gate 2. So `answer()` still gates on gate 2
(regex) only; gate 2b's result is attached to the return value
(`gate2b_ast_check`) purely so the evaluation harness can report, as a
SEPARATE number, "how many gate-3 (execution) failures would gate 2b have
caught before execution, that gate 2 missed" - the actual production-
readiness question, kept distinct from the RQ2 baseline-comparison
question. Degrades gracefully (`{"available": False, ...}`) if `sqlglot`
isn't installed - same "optional dependency, don't crash without it"
pattern as `causalml` in `src/causal/treatment_effect.py`.
"""
import re
import time

from sqlalchemy import text

from src.core.constants import CORE_MS_CODES
from src.core.db import get_session
from src.models import llm_factory

# Real schema (infra/postgres/migrations/001_init_schema.sql) - table/column
# list only, no row data. Kept in sync by hand (small, stable schema, 7
# structured tables, not counting disclosure_chunks which this Nhánh B
# baseline never queries) - used
# both as the LLM's prompt context (SCHEMA_DESCRIPTION) AND as the ground
# truth for _check_schema_alignment()'s hallucination check.
SCHEMA = {
    "companies": ["ticker", "company_name", "exchange", "sector", "created_at"],
    "financial_line_items": [
        "id", "ticker", "fiscal_year", "period", "ms_code", "item_name",
        "statement_type", "consolidation", "value", "unit", "updated_at",
    ],
    "financial_ratios": [
        "id", "ticker", "fiscal_year", "period", "ratio_group", "ratio_name",
        "value", "unit", "updated_at",
    ],
    "disclosures": [
        "id", "ticker", "fiscal_year", "period", "consolidation", "report_type",
        "audit_status", "auditor", "audit_opinion", "disclosure_date",
        "signed_date", "source_pdf_path", "structure_valid", "updated_at",
    ],
    "macro_indicators": [
        "id", "category", "indicator_name", "unit", "period_type",
        "fiscal_year", "period", "value", "source_file", "updated_at",
    ],
    "company_price_snapshot": [
        "id", "ticker", "trade_date", "open_price", "high_price", "low_price",
        "close_price", "volume", "source", "updated_at",
    ],
    "ingestion_log": ["id", "file_name", "file_type", "ticker", "row_count", "ingested_at", "notes"],
}

SCHEMA_DESCRIPTION = """
Bảng companies(ticker PK, company_name, exchange, sector, created_at) - danh sách công ty.
Bảng financial_line_items(id, ticker FK->companies, fiscal_year, period['Q1'..'Q4','YEAR','H1','9M'], ms_code, item_name, statement_type['balance_sheet','income_statement','cash_flow_indirect','cash_flow_direct'], consolidation, value, unit, updated_at) - từng dòng chỉ tiêu BCTC theo Thông tư 200.
Bảng financial_ratios(id, ticker FK, fiscal_year, period, ratio_group, ratio_name, value, unit, updated_at) - các chỉ số tài chính (P/E, ROEA, Beta...).
Bảng disclosures(id, ticker FK, fiscal_year, period, consolidation, report_type['nam','ban_nien','quy'], audit_status, auditor, audit_opinion, disclosure_date, signed_date, source_pdf_path, structure_valid, updated_at) - metadata công bố thông tin.
Bảng macro_indicators(id, category, indicator_name, unit, period_type['QUARTER','MONTH','YEAR','DAILY','YTD'], fiscal_year, period, value, source_file, updated_at) - chỉ số vĩ mô quốc gia/ngành, KHÔNG có cột ticker.
Bảng company_price_snapshot(id, ticker FK, trade_date, open_price, high_price, low_price, close_price, volume, source, updated_at) - giá cổ phiếu theo ngày.
""".strip()

SYSTEM_PROMPT = (
    "Bạn là một công cụ sinh câu lệnh SQL (PostgreSQL) cho hệ thống dữ liệu tài chính "
    "doanh nghiệp Việt Nam. CHỈ được dùng các bảng/cột sau, KHÔNG được bịa thêm "
    "bảng hay cột nào không có trong danh sách:\n\n" + SCHEMA_DESCRIPTION + "\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. CHỈ trả về một câu lệnh SELECT duy nhất, KHÔNG giải thích, KHÔNG dùng "
    "markdown/code fence, KHÔNG viết bất kỳ chữ nào khác ngoài chính câu SQL.\n"
    "2. TUYỆT ĐỐI không sinh INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE - "
    "chỉ đọc dữ liệu (SELECT).\n"
    "3. Nếu câu hỏi cần lọc theo mã cổ phiếu, dùng cột ticker (viết hoa, vd 'HPG').\n"
    "4. Nếu câu hỏi không thể trả lời bằng schema trên, trả về đúng chuỗi: "
    "NOT_ANSWERABLE\n"
)


def _build_ms_code_block() -> str:
    """Bảng tra cứu ms_code -> tên khoản mục CHÍNH THỨC (Thông tư 200), lấy
    trực tiếp từ CORE_MS_CODES (src/core/constants.py) - CÙNG một nguồn dữ
    liệu mà Nhánh A (src/structured/schema.py's INDICATOR_ALIASES) dùng để
    tra `item_name` thật trong DB. Chỉ dùng cho ablation (with_ms_code_hint=
    True) - xem answer()'s docstring và scripts/ablation_nhanh_b_ms_code.py.

    Mục đích của ablation này (2026-08-07, theo phát hiện ở eval run #1 -
    xem docs/eval_run1_results.md §RQ2): Nhánh B mặc định KHÔNG được cấp tập
    giá trị hợp lệ của `item_name` (SCHEMA_DESCRIPTION chỉ liệt kê enum cho
    `period`/`statement_type`, không cho `item_name` vì có hàng trăm khoản
    mục) - hệ quả là LLM đoán chuỗi gần đúng ('Doanh thu thuần') thay vì tên
    chính thức đầy đủ ('Doanh thu thuần về bán hàng và cung cấp dịch vụ',
    ms_code='10'), khớp WHERE tuyệt đối thất bại thầm lặng, trả về 0 dòng mà
    không gate nào bắt được. Cấp đúng bảng ms_code->tên chính thức này vào
    prompt kiểm tra trực tiếp giả thuyết "bảng alias có kiểm soát mới là
    thành phần chịu lực" (không phải bản thân LLM kém khả năng sinh SQL).
    """
    lines = ["Bảng tra cứu mã số khoản mục CHÍNH THỨC (ms_code) theo statement_type - "
             "PHẢI dùng ĐÚNG NGUYÊN VĂN các tên khoản mục dưới đây khi lọc theo "
             "item_name, KHÔNG được rút gọn hay viết lại:"]
    for stype, codes in CORE_MS_CODES.items():
        lines.append(f"- {stype}:")
        for ms_code, name in codes.items():
            lines.append(f"    ms_code='{ms_code}' -> item_name='{name}'")
    return "\n".join(lines)


_MS_CODE_BLOCK = _build_ms_code_block()

# Biến thể ablation (2026-08-07) - CHỈ khác SYSTEM_PROMPT ở việc cấp thêm
# _MS_CODE_BLOCK + quy tắc 5. answer()'s mặc định (with_ms_code_hint=False)
# vẫn dùng SYSTEM_PROMPT gốc, giữ nguyên hành vi/kết quả đã thu thập ở eval
# run #1 - biến thể này chỉ được dùng khi gọi tường minh, không đổi baseline.
SYSTEM_PROMPT_MS_CODE_HINT = (
    SYSTEM_PROMPT + "\n" + _MS_CODE_BLOCK + "\n\n"
    "5. Khi lọc theo item_name, ƯU TIÊN dùng đúng nguyên văn tên trong bảng "
    "tra cứu ms_code ở trên (khớp chính xác WHERE item_name = '...'). Nếu "
    "khoản mục cần tìm không có trong bảng tra cứu, mới được viết tên theo "
    "cách hiểu thông thường.\n"
)

_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|exec|call|merge)\b",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*|\s*```", re.IGNORECASE)
_FROM_JOIN_RE = re.compile(r'\b(?:from|join)\s+[`"]?(\w+)[`"]?(?:\s+(?:as\s+)?[`"]?(\w+)[`"]?)?', re.IGNORECASE)
_DOTTED_COL_RE = re.compile(r"\b(\w+)\.(\w+)\b")

_MAX_ROWS = 50


def _extract_sql(raw_llm_output: str) -> str:
    """Strip markdown code fences the LLM sometimes adds despite rule 1,
    and trailing semicolons/whitespace."""
    cleaned = _CODE_FENCE_RE.sub("", raw_llm_output).strip()
    return cleaned.rstrip(";").strip()


def _split_statements(sql: str) -> tuple:
    """Returns (first_statement, note). `note` is set (not raised as a
    failure) if the LLM produced more than one ';'-separated statement
    despite rule 1 asking for exactly one - only the first is ever
    executed, but this is still a real instruction-following miss worth
    recording for RQ2, distinct from a hallucination or a safety
    violation."""
    parts = [s.strip() for s in sql.split(";") if s.strip()]
    if not parts:
        return sql, None
    if len(parts) > 1:
        return parts[0], f"LLM sinh {len(parts)} câu lệnh dù chỉ được yêu cầu 1 - chỉ thực thi câu đầu tiên."
    return parts[0], None


def _is_safe_select(sql: str) -> tuple:
    """Gate 1 - read-only enforcement. Returns (ok, reason)."""
    if not sql or not _SELECT_RE.match(sql):
        return False, "Không phải câu lệnh SELECT (hoặc rỗng) - từ chối thực thi."
    forbidden = _FORBIDDEN_RE.findall(sql)
    if forbidden:
        return False, f"Chứa từ khóa ghi/thay đổi dữ liệu bị cấm: {sorted(set(w.upper() for w in forbidden))}."
    return True, None


def _check_schema_alignment(sql: str) -> list:
    """Gate 2 (arXiv:2405.15307-inspired) - best-effort, regex-based check
    (NOT a full SQL parser): extract every FROM/JOIN table (+ optional
    alias) and every alias.column / table.column reference, flag any whose
    TABLE isn't in SCHEMA at all, or whose COLUMN isn't a real column of
    the table it resolves to (when the alias is resolvable). Returns a
    list of human-readable problem strings (empty = nothing flagged - but
    this does NOT prove the SQL is otherwise valid; real execution is the
    only ground truth for that, see gate 3)."""
    problems = []
    alias_to_table = {}
    for table, alias in _FROM_JOIN_RE.findall(sql):
        table_low = table.lower()
        if table_low not in SCHEMA:
            problems.append(f"Bảng không tồn tại: '{table}'")
            continue
        alias_to_table[table_low] = table_low
        if alias:
            alias_to_table[alias.lower()] = table_low

    for prefix, col in _DOTTED_COL_RE.findall(sql):
        table = alias_to_table.get(prefix.lower())
        if table is None:
            continue  # prefix isn't a known table/alias (function call, db-qualified name, etc.) - not enough info to judge, stay conservative
        if col.lower() not in SCHEMA[table]:
            problems.append(f"Cột không tồn tại: '{prefix}.{col}' (bảng '{table}' không có cột này)")
    return problems


def _check_schema_alignment_ast(sql: str) -> dict:
    """Gate 2b - real AST-based schema check via `sqlglot`, OBSERVATIONAL
    ONLY (see module docstring for why this doesn't replace gate 2). Unlike
    _check_schema_alignment()'s regex, this also validates BARE
    (unqualified) column references - a bare column is flagged if it
    doesn't exist in ANY table referenced by the query (can't always
    resolve a bare column to exactly ONE table without full binder-style
    scope tracking, so "exists in at least one in-scope table" is the
    check, same conservative spirit as the regex gate's "unknown alias ->
    skip, don't guess").

    Returns {"available": bool, "problems": list[str]|None,
    "parse_error": bool, "reason": str|None}. `available=False` (sqlglot
    not installed) and `parse_error=True` (sqlglot itself couldn't parse
    the SQL - a real, separately-interesting signal for RQ2: gate 1
    already confirmed it starts with SELECT, so a sqlglot parse failure
    here means genuinely malformed SQL, not just an unknown identifier)
    are both non-fatal - caller attaches whichever result it gets, never
    raises."""
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return {
            "available": False, "problems": None, "parse_error": False,
            "reason": "sqlglot chưa được cài đặt (pip install sqlglot) - Cổng 2b không chạy được.",
        }

    try:
        # dialect updated 2026-07-26 (mysql -> postgres) to match the
        # project's database migration - see module docstring and
        # docs/PROGRESS.md. sqlglot's "postgres" reader also accepts plain
        # ANSI-ish SQL without backticks/MySQL-only quirks, so this is not
        # expected to make Gate 2b stricter or looser in practice, just
        # accurate about which engine's grammar it's checking against.
        parsed = sqlglot.parse_one(sql, read="postgres")
    except Exception as e:
        return {
            "available": True, "problems": None, "parse_error": True,
            "reason": f"sqlglot không parse được câu SQL: {e}",
        }

    alias_to_table = {}
    tables_in_scope = set()
    for table_exp in parsed.find_all(exp.Table):
        table_name = table_exp.name.lower()
        if not table_name:
            continue
        tables_in_scope.add(table_name)
        alias_to_table[table_name] = table_name
        alias = (table_exp.alias_or_name or "").lower()
        if alias:
            alias_to_table[alias] = table_name

    problems = []
    for table_name in tables_in_scope:
        if table_name not in SCHEMA:
            problems.append(f"[AST] Bảng không tồn tại: '{table_name}'")

    all_scope_columns = set()
    for t in tables_in_scope:
        if t in SCHEMA:
            all_scope_columns |= {c.lower() for c in SCHEMA[t]}

    for col_exp in parsed.find_all(exp.Column):
        col_name = (col_exp.name or "").lower()
        if not col_name:
            continue
        table_prefix = (col_exp.table or "").lower()
        if table_prefix:
            resolved_table = alias_to_table.get(table_prefix)
            if resolved_table is None:
                continue  # unknown prefix - not enough info, same conservative choice as the regex gate
            if resolved_table in SCHEMA and col_name not in {c.lower() for c in SCHEMA[resolved_table]}:
                problems.append(
                    f"[AST] Cột không tồn tại: '{table_prefix}.{col_name}' (bảng '{resolved_table}')"
                )
        else:
            # Bare column - exactly what _check_schema_alignment()'s regex cannot see.
            if all_scope_columns and col_name not in all_scope_columns:
                problems.append(f"[AST] Cột trần không khớp bảng nào đang dùng trong câu truy vấn: '{col_name}'")

    return {"available": True, "problems": problems, "parse_error": False, "reason": None}


def answer(question: str, with_ms_code_hint: bool = False) -> dict:
    """Returns:
    {"question", "generated_sql", "raw_llm_output", "ok", "gate_failed",
     "problems", "notes", "rows", "row_count", "truncated", "error",
     "elapsed_s", "with_ms_code_hint"}

    `gate_failed` in {None, "not_answerable", "unsafe_statement",
    "schema_alignment", "execution"} - which gate stopped this query (see
    module docstring), or None if it succeeded. `problems` holds the
    specific issues found at whichever gate failed; `notes` holds
    non-fatal observations (currently: multi-statement generation) that
    don't block execution but are still worth recording.

    `with_ms_code_hint` (default False, preserves the exact prompt/behavior
    used for eval run #1 - docs/eval_run1_results.md) - when True, uses
    SYSTEM_PROMPT_MS_CODE_HINT instead, which additionally supplies the
    official ms_code -> item_name lookup table (see _build_ms_code_block()).
    This is the RQ2 ablation flagged in eval run #1's results: does Nhánh B's
    27/27 "success but 0 rows" failure mode (wrong item_name string guess)
    go away once it's given the same controlled name table Nhánh A already
    uses? Only set True from scripts/ablation_nhanh_b_ms_code.py or
    run_eval.run_nhanh_a_vs_b(..., with_ms_code_hint=True) - never the
    default, so existing collected results stay reproducible."""
    system_prompt = SYSTEM_PROMPT_MS_CODE_HINT if with_ms_code_hint else SYSTEM_PROMPT
    start = time.time()
    raw = llm_factory.generate(question, system=system_prompt)
    sql = _extract_sql(raw)
    elapsed = time.time() - start
    base = {
        "question": question, "generated_sql": sql, "raw_llm_output": raw,
        "elapsed_s": round(elapsed, 2), "with_ms_code_hint": with_ms_code_hint,
    }

    def _fail(gate_failed: str, problems: list, notes: list = None) -> dict:
        return {**base, "ok": False, "gate_failed": gate_failed, "problems": problems,
                "notes": notes or [], "rows": None, "row_count": 0, "truncated": False, "error": None}

    if sql.upper() == "NOT_ANSWERABLE":
        return _fail("not_answerable", ["LLM xác định câu hỏi không thể trả lời bằng schema hiện có."])

    sql, note = _split_statements(sql)
    notes = [note] if note else []

    safe, reason = _is_safe_select(sql)
    if not safe:
        return _fail("unsafe_statement", [reason], notes)

    # Gate 2b (AST, sqlglot) - computed alongside gate 2, attached to every
    # outcome below regardless of which gate stops execution. Observational
    # only - see module docstring for why this must NOT change gate_failed.
    gate2b = _check_schema_alignment_ast(sql)

    problems = _check_schema_alignment(sql)
    if problems:
        return {**_fail("schema_alignment", problems, notes), "gate2b_ast_check": gate2b}

    try:
        with get_session() as session:
            result = session.execute(text(sql))
            cols = list(result.keys())
            fetched = result.fetchmany(_MAX_ROWS + 1)  # 1 extra row = cheap truncation signal, no separate COUNT(*)
    except Exception as e:
        return {
            **_fail("execution", [f"Lỗi thực thi SQL thật (schema alignment KHÔNG bắt được lỗi này): {e}"], notes),
            "gate2b_ast_check": gate2b,
        }

    truncated = len(fetched) > _MAX_ROWS
    rows = [dict(zip(cols, row)) for row in fetched[:_MAX_ROWS]]
    return {**base, "ok": True, "gate_failed": None, "problems": [], "notes": notes,
            "rows": rows, "row_count": len(rows), "truncated": truncated, "error": None,
            "gate2b_ast_check": gate2b}


def _format_gate2b(gate2b: dict) -> list:
    """Renders gate2b_ast_check for CLI display - purely informational, see
    module docstring for why this never affects `ok`/`gate_failed` above."""
    if gate2b is None:
        return []
    if not gate2b.get("available"):
        return [f"[Cổng 2b/AST - sqlglot]: {gate2b.get('reason')}"]
    if gate2b.get("parse_error"):
        return [f"[Cổng 2b/AST - sqlglot]: lỗi parse - {gate2b.get('reason')}"]
    problems = gate2b.get("problems") or []
    if not problems:
        return ["[Cổng 2b/AST - sqlglot]: không phát hiện vấn đề."]
    lines = ["[Cổng 2b/AST - sqlglot] phát hiện (quan sát, KHÔNG chặn thực thi):"]
    for p in problems:
        lines.append(f"    {p}")
    return lines


def format_answer(result: dict) -> str:
    lines = [f"SQL sinh ra:\n{result['generated_sql']}"]
    for n in result.get("notes") or []:
        lines.append(f"[Ghi chú: {n}]")
    if not result["ok"]:
        lines.append(f"\n[THẤT BẠI ở gate={result['gate_failed']}]")
        for p in result["problems"]:
            lines.append(f"  - {p}")
        lines.extend(_format_gate2b(result.get("gate2b_ast_check")))
        return "\n".join(lines)
    trunc_note = " (đã cắt ở " + str(_MAX_ROWS) + " dòng đầu)" if result["truncated"] else ""
    lines.append(f"\n{result['row_count']} dòng kết quả{trunc_note}:")
    for row in result["rows"][:10]:
        lines.append(f"  {row}")
    if result["row_count"] > 10:
        lines.append(f"  ... (+{result['row_count'] - 10} dòng nữa, xem result['rows'] đầy đủ)")
    lines.extend(_format_gate2b(result.get("gate2b_ast_check")))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Tổng tài sản của HPG năm 2024 là bao nhiêu?"
    print(f"Q: {q}")
    res = answer(q)
    print(format_answer(res))
    print(f"\n(elapsed: {res['elapsed_s']}s)")
