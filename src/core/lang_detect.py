"""Rule-based Vietnamese/English question-language detector - added
2026-09-08 to support bilingual (VI/EN) chat for the English-language thesis
defense demo (see docs/PROGRESS.md same-date entry).

Deliberately NOT using a third-party library (langdetect/fasttext/etc.):
Vietnamese with full diacritics is unambiguous from its Unicode character
set alone (no other language commonly asked in this project's demo uses
these tone-marked Latin characters), and a demo/presentation question is
always cleanly written in one language, not code-mixed - so a zero-dependency
character-set check is both sufficient and, for a thesis defense, easier to
explain/defend than an opaque ML language-ID model would be (same
"documented judgment call over unnecessary dependency" philosophy as
rag_pipeline.py's RECENCY_PENALTY_PER_YEAR and friends).

Scope note: this only ever returns "vi" or "en" - the project has no third
language in scope. A question with zero Vietnamese diacritics defaults to
"en" (not any other detection), including edge cases like a bare ticker
("VNM?") or a number-only question - those default to "en" too, but in
practice such fragments render identically regardless of which system
prompt fires (no natural-language sentence to translate), so the default
side is a low-stakes judgment call, not a real failure mode.
"""

_VN_DIACRITIC_CHARS = set(
    "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữỳýỵỷỹđ"
    "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    "ÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
)


def detect_language(text: str) -> str:
    """Returns "vi" if `text` contains any Vietnamese-specific diacritic
    character, else "en". Case-sensitive set covers both cases so no
    .lower() call is needed (also avoids an unnecessary allocation on every
    query-time call)."""
    if not text:
        return "en"
    return "vi" if any(ch in _VN_DIACRITIC_CHARS for ch in text) else "en"


# A "real sentence" needs at least this many alphabetic word tokens before we
# treat the ABSENCE of Vietnamese diacritics as positive evidence of English.
# 2 is deliberately low: "Total assets of VNZ?" is 4 tokens, while the
# fragments this guard exists for ("VNZ?", "VNZ 2025", "280") are 0-1.
_MIN_TOKENS_FOR_POSITIVE_EN = 2


def resolve_answer_language(question: str, ui_lang: str | None = None) -> str:
    """Decide which language the ANSWER should be written in.

    Added 2026-09-22. detect_language() above answers "what script is this
    written in", which is the right question for a full sentence but the
    wrong one for a fragment: it defaults anything without Vietnamese
    diacritics to "en", so a bare ticker ("VNZ?") reads as English even when
    the user has the whole interface set to Vietnamese. That default is
    harmless for the RAG prompt it was written for (see this module's header:
    a fragment has no prose to translate), but it is NOT harmless once the
    structured branches are bilingual, because those branches answer a
    fragment with several sentences of generated scaffolding.

    So this resolves in three steps, most confident signal first:
      1. Vietnamese diacritics present  -> "vi". An unambiguous positive
         signal, and it OVERRIDES `ui_lang`: someone typing Vietnamese wants
         a Vietnamese answer even if they left the toggle on English.
      2. No diacritics, but >= _MIN_TOKENS_FOR_POSITIVE_EN alphabetic word
         tokens -> "en". A real English sentence, also a positive signal.
      3. Neither (bare ticker, digits, single word) -> `ui_lang`, i.e. the
         interface toggle, falling back to "vi" when no toggle was sent.

    Known limitation, inherited from detect_language() and accepted for the
    same documented reason: Vietnamese typed WITHOUT diacritics ("Tong tai
    san cua VNZ nam 2025") hits step 2 and resolves to "en". This module's
    header already scopes the detector to cleanly-written demo questions;
    fixing it would need a Vietnamese word-list, i.e. exactly the dependency
    this module exists to avoid.
    """
    if question and any(ch in _VN_DIACRITIC_CHARS for ch in question):
        return "vi"
    tokens = [t for t in (question or "").split() if any(ch.isalpha() for ch in t)]
    if len(tokens) >= _MIN_TOKENS_FOR_POSITIVE_EN:
        return "en"
    return "en" if ui_lang == "en" else "vi"


if __name__ == "__main__":
    # python3 -m src.core.lang_detect
    tests = [
        ("Rủi ro pháp lý của VNM là gì?", "vi"),
        ("What are VNM's legal risks?", "en"),
        ("Tổng tài sản của CTD năm 2025 là bao nhiêu?", "vi"),
        ("What was CTD's total assets in 2025?", "en"),
        ("VNM", "en"),
    ]
    for text, expected in tests:
        got = detect_language(text)
        status = "OK" if got == expected else "MISMATCH"
        print(f"[{status}] detect_language({text!r}) = {got!r} (expected {expected!r})")

    print()
    resolve_tests = [
        # (question, ui_lang, expected) - see resolve_answer_language()'s 3 steps
        ("Tổng tài sản của VNZ năm 2025?", "en", "vi"),   # diacritics override the toggle
        ("What is VNZ's total assets in 2025?", "vi", "en"),  # real EN sentence overrides too
        ("VNZ?", "en", "en"),                            # fragment -> toggle
        ("VNZ?", "vi", "vi"),                            # same fragment, other toggle
        ("VNZ 2025", None, "vi"),                        # fragment, no toggle -> vi default
    ]
    for text, ui, expected in resolve_tests:
        got = resolve_answer_language(text, ui)
        status = "OK" if got == expected else "MISMATCH"
        print(f"[{status}] resolve_answer_language({text!r}, ui={ui!r}) = {got!r} (expected {expected!r})")
