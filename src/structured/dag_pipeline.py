"""End-to-end DAG structured-lookup pipeline: question -> parsed entities ->
SQL lookup -> answer with citation. This is System S2 ("DAG-only") in the
thesis's 3-system comparison (S1 RAG-only / S2 DAG-only / S3 Router).

Usage:
    python -m src.structured.dag_pipeline "Tổng tài sản của VNM quý 1 năm 2026 là bao nhiêu?"
"""
from src.structured.query_builder import execute, parse_question


def answer(question: str, lang: str | None = None) -> str:
    """`lang` added 2026-09-25, OPTIONAL and Vietnamese by default.

    This is System S2, a frozen ablation baseline: Chapter 4's S2 numbers
    must stay reproducible, so every existing caller - benchmark scripts,
    CLI - passes no `lang` and gets exactly the Vietnamese output it always
    got. Only the QA service passes lang="en", so an English demo no longer
    shows one Vietnamese answer in the middle of the S1/S2/S3 comparison.
    The lookup itself (parse_question + execute) is identical either way;
    only the wording around the number changes.
    """
    parsed = parse_question(question)
    result = execute(parsed)

    if lang == "en":
        from src.router.display_labels import error_en, line_item_label

        if not result["ok"]:
            return (
                f"Could not answer via DAG: {error_en(result['error'])}. "
                f"Recognised: ticker={parsed.ticker}, year={parsed.fiscal_year}, "
                f"period={parsed.period}, matched line item='{parsed.matched_alias}'."
            )
        return (
            f"{result['value']:,} {result['unit']} "
            f"(source: {result['ticker']}, "
            f"{line_item_label(result['statement_type'], result['ms_code'], result['item_name'])} "
            f"[{result['statement_type']} MS={result['ms_code']}], "
            f"period {result['period']}/{result['fiscal_year']}, {result['consolidation']})"
        )

    if not result["ok"]:
        return (
            f"Không trả lời được qua DAG: {result['error']}. "
            f"Đã nhận diện: mã={parsed.ticker}, năm={parsed.fiscal_year}, "
            f"kỳ={parsed.period}, chỉ tiêu khớp='{parsed.matched_alias}'."
        )

    return (
        f"{result['value']:,} {result['unit']} "
        f"(nguồn: {result['ticker']}, {result['item_name']} [{result['statement_type']} "
        f"MS={result['ms_code']}], kỳ {result['period']}/{result['fiscal_year']}, {result['consolidation']})"
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Tổng tài sản của VNM quý 1 năm 2026 là bao nhiêu?"
    print(f"Q: {q}")
    print(f"A: {answer(q)}")
