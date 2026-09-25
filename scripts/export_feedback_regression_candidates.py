"""Export human-accepted feedback cases as a JSONL regression fixture.

Only ``accepted`` records are exported.  Candidate/rejected rows remain in
PostgreSQL for audit and never affect a benchmark or model automatically.

Run from the project root after the reviewer accepts at least one candidate:
    PYTHONPATH=. .venv/bin/python scripts/export_feedback_regression_candidates.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import text

from src.core.db import get_session


def export_accepted(output: Path) -> int:
    with get_session() as session:
        rows = session.execute(text("""
            SELECT id, question_text, expected_answer, expected_sources,
                   reviewer_note, created_at,
                   promoted_by, promoted_persona, decided_by, decided_persona
            FROM feedback.regression_candidates
            WHERE status = 'accepted'
            ORDER BY id
        """)).mappings().all()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "id": f"feedback-{row['id']}",
                "question": row["question_text"],
                "expected_answer": row["expected_answer"],
                "expected_sources": row["expected_sources"],
                "reviewer_note": row["reviewer_note"],
                "origin": "human_reviewed_feedback",
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                # Provenance (2026-09-22). Carried INTO the fixture, not just
                # left in Postgres: the .jsonl is what a future reader (or
                # examiner) actually inspects, and "who decided this was the
                # right answer" is part of whether the case can be trusted.
                # It also makes the fixture filterable - if one reviewer is
                # later found to have entered careless expected answers, their
                # cases can be dropped by name instead of discarding the set.
                "promoted_by": row["promoted_by"],
                "promoted_persona": row["promoted_persona"],
                "decided_by": row["decided_by"],
                "decided_persona": row["decided_persona"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="data/eval/feedback_regression.jsonl",
        help="JSONL fixture path, relative to the project root by default.",
    )
    args = parser.parse_args()
    count = export_accepted(Path(args.output))
    print(f"Exported {count} accepted feedback regression case(s) to {args.output}")


if __name__ == "__main__":
    main()
