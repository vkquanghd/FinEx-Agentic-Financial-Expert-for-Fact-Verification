"""One-off regression check (2026-09-15): verify every question in
scripts/final_benchmark_100_en.py's QUESTIONS list routes to its own
expected_branch, using classify_intent() directly (pure regex, no DB/LLM),
same sandboxed DB-stub technique used earlier this session for
final_benchmark_en.py's 26 questions and the 60-question audit set.

sys.modules stubbing rationale: classify_intent() -> parse_question() ->
_extract_ticker() -> is_known_ticker() makes a real DB call; in this
DB-less sandbox that call raises, and _extract_ticker()'s own documented
fail-open behavior (see src/structured/query_builder.py) then trusts the
bare regex match instead - which is exactly the behavior we want to check
branch-routing logic in isolation from DB/network availability.

Run from the Thesis_Project root: python3 scripts/verify_100_regression.py
"""
import sys
import types

# --- stub sqlalchemy so src.core.db (and anything importing `text`/
# `create_engine`/`sessionmaker`) can be imported without a real driver ---
fake_sqlalchemy = types.ModuleType("sqlalchemy")
fake_sqlalchemy.text = lambda s: s
fake_sqlalchemy.create_engine = lambda *a, **k: None
sys.modules["sqlalchemy"] = fake_sqlalchemy

fake_orm = types.ModuleType("sqlalchemy.orm")


def _fake_sessionmaker(*a, **k):
    def _factory():
        raise RuntimeError("no DB in sandbox")
    return _factory


fake_orm.sessionmaker = _fake_sessionmaker
sys.modules["sqlalchemy.orm"] = fake_orm

fake_db = types.ModuleType("src.core.db")


def _get_session():
    raise RuntimeError("no DB in sandbox")


fake_db.get_session = _get_session
fake_db.engine = None
fake_db.SessionLocal = None
sys.modules["src.core.db"] = fake_db

sys.path.insert(0, ".")

from src.router.rules import classify_intent  # noqa: E402
from scripts.final_benchmark_100_en import QUESTIONS  # noqa: E402

# CORE_16 structured_lookup questions: classify_intent() cannot see the
# CORE_16 dag->rag runtime fallback (dag_execute() finding zero rows in
# financial_line_items, then route() relabeling branch to "rag" and setting
# fallback_from="dag") - that only happens with a real DB connection.
# classify_intent() alone correctly returns "dag" for these (ticker/period/
# ms_code all syntactically resolve, same as any PANEL ticker) since it has
# no way to know the table is empty for this ticker. Known, expected
# sandbox limitation - not a question-design bug - so these ids are
# reported in their own bucket instead of FAILURES.
KNOWN_SANDBOX_LIMITATION_IDS = {2, 4, 6, 8, 10, 12, 14, 16}

passed, failed, limitation = 0, [], []
category_counts = {}
for q in QUESTIONS:
    category_counts[q["category"]] = category_counts.get(q["category"], 0) + 1
    try:
        intent = classify_intent(q["question"])
        actual = intent.branch
    except Exception as e:
        actual = f"EXC:{type(e).__name__}:{e}"
    expected = q["expected_branch"]
    ok = actual == expected
    if ok:
        passed += 1
    elif q["id"] in KNOWN_SANDBOX_LIMITATION_IDS and actual == "dag" and expected == "rag":
        limitation.append((q["id"], q["category"], q["lang"], expected, actual, q["question"]))
    else:
        failed.append((q["id"], q["category"], q["lang"], expected, actual, q["question"]))

print(f"TOTAL {len(QUESTIONS)}  PASS {passed}  FAIL {len(failed)}  KNOWN_SANDBOX_LIMITATION {len(limitation)}")
print("\nCategory counts:")
for cat, n in category_counts.items():
    print(f"  {cat}: {n}")

if failed:
    print("\nFAILURES:")
    for fid, cat, lang, exp, act, question in failed:
        print(f"  id={fid} [{cat}/{lang}] expected={exp} actual={act}")
        print(f"    Q: {question}")

if limitation:
    print("\nKNOWN SANDBOX LIMITATIONS (expected dag->rag runtime fallback, needs live DB to confirm):")
    for fid, cat, lang, exp, act, question in limitation:
        print(f"  id={fid} [{cat}/{lang}] classify_intent={act}, real route() should fall back to {exp}")

# duplicate-id check
ids = [q["id"] for q in QUESTIONS]
dupes = {i for i in ids if ids.count(i) > 1}
if dupes:
    print(f"\nDUPLICATE IDS: {dupes}")
else:
    print(f"\nNo duplicate ids. id range: {min(ids)}-{max(ids)}, count={len(ids)}")
