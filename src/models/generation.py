"""Answer generation with citation, given retrieved RAG chunks. Used by
src/retrieval/rag_pipeline.py (System S1 "RAG-only" in the thesis's 3-system
comparison - see src/structured/dag_pipeline.py's docstring for S2, the
same framing).

Design principle: the LLM is instructed to answer ONLY from the provided
excerpts and to say explicitly when the excerpts don't cover the question,
rather than filling gaps from its own training knowledge - this is the
whole point of RAG for financial QA (grounded, checkable answers over
un-checkable "the model just knows this"), and it's what RAGAS's
"faithfulness" metric (cited in Thesis_Direction_Redesign.pdf section 6.6)
actually measures.
"""
from src.core.lang_detect import detect_language
from src.models import llm_factory
from src.models.callbacks import timed_generation

# BILINGUAL SUPPORT (added 2026-09-08) - the thesis defense is presented in
# English, but the corpus (disclosure_chunks) and every prompt/label in this
# file were Vietnamese-only until now. Retrieval itself needs NO change: the
# embedding model is BAAI/bge-m3 (src/retrieval/embeddings.py), whose
# training explicitly targets cross-lingual alignment across 100+ languages
# (Chen et al. 2024, "BGE M3-Embedding: Multi-Lingual, Multi-Functionality,
# Multi-Granularity Text Embeddings Through Self-Knowledge Distillation",
# arXiv:2402.03216) - an English query embeds into the SAME semantic space
# as the Vietnamese-language chunks it needs to retrieve, so no second
# English index/corpus is needed (see docs/PROGRESS.md 2026-09-08 entry for
# the full academic framing to cite in the thesis). The generator
# (Qwen3, src/models/llm_factory.py) is likewise natively multilingual
# (Qwen Team, "Qwen3 Technical Report", 2025) - it already follows an
# English instruction/question correctly, it just needs one.
# The actual gap closed here is narrower: this file's SYSTEM_PROMPT (citation
# rules, faithfulness rules) was hard-written in Vietnamese, so an English
# question could get a Vietnamese-instructed answer, or a mixed-language one.
# detect_language() (rule-based, zero extra dependency - see its own
# docstring) picks SYSTEM_PROMPT vs SYSTEM_PROMPT_EN per question, so
# citation-rule compliance is never weakened by an instruction/answer
# language mismatch. Note: the ROUTER (src/router/rules.py) and DAG indicator
# resolution (src/structured/schema.py's _ENGLISH_ALIASES) already had
# partial English marker/alias coverage before this change - this file was
# the one part of the RAG answer path still Vietnamese-only.
SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi tài chính dựa trên tài liệu công bố thông tin "
    "của doanh nghiệp niêm yết tại Việt Nam. QUY TẮC BẮT BUỘC:\n"
    "1. Chỉ trả lời dựa trên các đoạn trích được cung cấp bên dưới - KHÔNG dùng "
    "kiến thức nền của bạn để bổ sung số liệu hay sự kiện không có trong đoạn trích.\n"
    "2. Nếu các đoạn trích không đủ thông tin để trả lời, hãy nói rõ "
    "\"Không tìm thấy thông tin trong tài liệu được cung cấp\" thay vì đoán.\n"
    "3. Khi trả lời, với MỖI số liệu hoặc khẳng định cụ thể lấy từ đoạn trích, "
    "phải ghi [Nguồn N] VÀ kèm theo một cụm trích dẫn ngắn (nguyên văn hoặc rất "
    "sát nguyên văn, tối đa khoảng 15 từ) lấy trực tiếp từ đúng đoạn trích N đó "
    "để cho thấy rõ căn cứ - ví dụ: \"...tổng tài sản đạt 23.723 tỷ đồng "
    "[Nguồn 2: 'Tổng cộng tài sản... 23.723.796.477.060']\". KHÔNG chỉ ghi "
    "[Nguồn N] suông mà không có trích dẫn kèm theo - người đọc phải thấy được "
    "chính xác câu/số nào trong tài liệu dẫn tới câu trả lời, không chỉ số thứ "
    "tự nguồn.\n"
    "4. Trả lời ngắn gọn, đúng trọng tâm câu hỏi.\n"
    "5. Mỗi đoạn trích có ghi rõ năm của báo cáo nguồn - LUÔN nêu rõ năm đó "
    "trong câu trả lời (vd \"Theo BCTC năm 2024...\"), đặc biệt với các thông tin "
    "có thể thay đổi theo thời gian (thành viên HĐQT, ban điều hành, cổ đông, "
    "v.v.) - không trả lời như thể đó là thông tin cố định/hiện tại nếu không rõ. "
    "Nếu các đoạn trích thuộc nhiều năm khác nhau, hãy nói rõ năm nào ứng với "
    "thông tin nào thay vì gộp chung.\n"
    "6. Giữ NGUYÊN VĂN tên riêng, chức danh, và dấu tiếng Việt đúng như trong "
    "đoạn trích (vd \"Bà Cao Thị Ngọc Dung\", KHÔNG viết thành \"Mrs. Cao Thi Ngoc "
    "Dung\") - không dịch, không bỏ dấu, không Anh-hóa bất kỳ tên người/chức danh nào.\n"
    "7. Với tên viết tắt/thương hiệu (vd \"pwc\", \"iCPA\" trong đoạn trích), giữ "
    "NGUYÊN dạng hoa/thường như xuất hiện trong đoạn trích và viết LIỀN thành một "
    "CHUỖI DUY NHẤT, không chèn khoảng trắng ở bất kỳ vị trí nào bên trong - kể cả "
    "giữa một chữ cái thường và cụm chữ hoa theo sau (vd đoạn trích viết \"iCPA\" thì "
    "trả lời đúng \"iCPA\", TUYỆT ĐỐI KHÔNG viết \"i CPA\"; đoạn trích viết \"pwc\" thì "
    "viết hoa thành \"PwC\", TUYỆT ĐỐI KHÔNG viết \"Pw C\" hay bất kỳ dạng tách rời "
    "nào khác của tên viết tắt)."
)


SYSTEM_PROMPT_EN = (
    "You are a financial QA assistant answering questions based on official "
    "disclosure documents of Vietnamese listed companies. MANDATORY RULES:\n"
    "0. Write your ENTIRE answer in English, and ONLY English - this is "
    "non-negotiable even though every excerpt below is written in Vietnamese. "
    "Do not slip into Vietnamese for any sentence, clause, or explanation - the "
    "ONLY Vietnamese allowed anywhere in your answer is a proper name/title/direct "
    "quote you are required to preserve verbatim under rule 6/7 below. "
    "(Added 2026-09-08 after a real test run showed the model answering an English "
    "question in Vietnamese because every source excerpt is Vietnamese - if you "
    "notice yourself about to write a Vietnamese sentence that is not a quoted "
    "proper noun, stop and write it in English instead.)\n"
    "1. Answer ONLY from the excerpts provided below - do NOT use your own "
    "background knowledge to add numbers or facts that are not in the excerpts.\n"
    "2. If the excerpts do not contain enough information to answer, say clearly "
    "\"No information found in the provided documents\" instead of guessing.\n"
    "3. For EVERY specific figure or claim taken from an excerpt, you MUST cite "
    "[Source N] AND include a short direct quote (verbatim or very close to "
    "verbatim, at most ~15 words) taken directly from excerpt N, so the basis "
    "is visible - e.g. \"...total assets reached VND 23,723 billion "
    "[Source 2: 'Total assets... 23,723,796,477,060']\". Do NOT write a bare "
    "[Source N] with no accompanying quote - the reader must see exactly which "
    "sentence/number in the document led to the answer, not just a source index.\n"
    "4. Answer concisely, staying on the point of the question.\n"
    "5. Each excerpt states the fiscal year of its source report - ALWAYS state "
    "that year in your answer (e.g. \"According to the 2024 financial statements...\"), "
    "especially for information that can change over time (board members, "
    "executives, shareholders, etc.) - never answer as if it were fixed/current "
    "information when that is not clear. If excerpts span multiple years, state "
    "clearly which year each piece of information belongs to instead of merging them.\n"
    "6. Keep proper names, titles, and Vietnamese diacritics EXACTLY as they "
    "appear in the excerpt (e.g. keep \"Bà Cao Thị Ngọc Dung\" as written, do "
    "NOT anglicize, romanize, or strip diacritics from any person's name or title) "
    "- translate the surrounding sentence into English, but never the proper noun itself.\n"
    "7. For abbreviations/brand names (e.g. \"pwc\", \"iCPA\" in an excerpt), "
    "keep the EXACT casing as it appears in the excerpt and write it as ONE "
    "UNBROKEN token, with no space inserted anywhere inside it - including "
    "between a lowercase letter and a following uppercase cluster (e.g. if the "
    "excerpt says \"iCPA\", answer must say \"iCPA\", NEVER \"i CPA\"; if the "
    "excerpt says \"pwc\", capitalize it as \"PwC\", NEVER \"Pw C\" or any other "
    "split form of the abbreviation)."
)


def build_prompt(question: str, chunks: list, lang: str = None) -> str:
    """`lang` picks the scaffold/label language ("vi" or "en"); auto-detected
    from `question` via detect_language() when not given explicitly (2026-09-08 -
    see the bilingual-support comment above SYSTEM_PROMPT). Existing callers
    that don't pass `lang` keep exactly the old Vietnamese-only behavior for
    any Vietnamese question - only English-detected questions get the new
    English scaffold, so this is additive, not a behavior change for the
    existing (Vietnamese) benchmark.jsonl items."""
    if lang is None:
        lang = detect_language(question)

    if lang == "en":
        if not chunks:
            context = "(no excerpt was retrieved)"
        else:
            parts = []
            for i, c in enumerate(chunks, start=1):
                year_part = f", year {c['year']}" if c.get("year") else ""
                parts.append(
                    f"[Source {i}: ticker {c['ticker']}, document type {c['doctype']}{year_part}, "
                    f"page {c['page']}]\n{c['chunk_text']}"
                )
            context = "\n\n".join(parts)
        return (
            f"Document excerpts:\n\n{context}\n\n"
            f"---\n\nQuestion: {question}\n\nAnswer:"
        )

    if not chunks:
        # Still build a prompt (rather than skip generation) so the model's
        # own "không tìm thấy" phrasing stays consistent - see NO_CONTEXT_MSG
        # in rag_pipeline.py for the short-circuit that skips this entirely.
        context = "(không có đoạn trích nào được truy xuất)"
    else:
        parts = []
        for i, c in enumerate(chunks, start=1):
            year_part = f", năm {c['year']}" if c.get("year") else ""
            parts.append(
                f"[Nguồn {i}: mã {c['ticker']}, loại tài liệu {c['doctype']}{year_part}, "
                f"trang {c['page']}]\n{c['chunk_text']}"
            )
        context = "\n\n".join(parts)

    return (
        f"Các đoạn trích tài liệu:\n\n{context}\n\n"
        f"---\n\nCâu hỏi: {question}\n\nTrả lời:"
    )


def answer_with_citation(question: str, chunks: list) -> dict:
    """Returns {"question", "answer", "sources", "elapsed_s"}. `sources` is
    the same chunk list passed in (ticker/doctype/source_file/page/similarity)
    so the caller can render citations even if the LLM's own [Nguồn N]/
    [Source N] references are imperfect - the ground truth of what was
    actually retrieved doesn't depend on the LLM repeating it correctly.

    2026-09-08: `question`'s language is auto-detected (detect_language()) to
    pick SYSTEM_PROMPT (vi) vs SYSTEM_PROMPT_EN (en) and the matching
    build_prompt() scaffold - see the bilingual-support comment above
    SYSTEM_PROMPT for why. A Vietnamese question's behavior/output is
    byte-for-byte unchanged from before this change."""
    lang = detect_language(question)
    system_prompt = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT
    prompt = build_prompt(question, chunks, lang=lang)
    generation_fallback = None
    try:
        answer_text, elapsed = timed_generation(
            llm_factory.generate, prompt, system=system_prompt
        )
    except llm_factory.OllamaNotReachable as exc:
        # Retrieval may still have succeeded when local generation is slow or
        # unavailable.  Preserve that evidence instead of turning a usable
        # source list into a generic server failure.  This is explicitly not
        # an answer synthesized from the excerpts: the caller/UI can render
        # the returned official-document pages for a reviewer to inspect.
        #
        # 2026-09-24 - NÊU ĐÍCH DANH NGUYÊN NHÂN. Bản cũ nói "chưa hoàn tất
        # phần tổng hợp có căn cứ" cho MỌI kiểu hỏng, nên ba tình huống rất
        # khác nhau - Ollama chưa chạy, model chưa pull, và Ollama chạy
        # nhưng quá tải/timeout - đều hiện y hệt một câu. Suốt quá trình
        # thử nghiệm điều đó dẫn tới việc đi tìm lỗi ở router và ở dữ liệu,
        # trong khi nguyên nhân thật chỉ là chưa chạy `ollama serve`.
        # llm_factory đã soạn sẵn câu giải thích có ích ("Is Ollama running
        # (`ollama serve`)...") - trước đây nó bị vứt đi hoàn toàn.
        detail = str(exc).strip()
        if lang == "en":
            answer_text = (
                "The relevant disclosure excerpts were retrieved, but the local "
                "Qwen service did not complete a grounded synthesis, so no answer "
                "was generated from them. Review the official disclosure pages "
                f"listed below.\n\nReason: {detail}"
            )
        else:
            answer_text = (
                "Đã truy xuất được các đoạn trích công bố liên quan, nhưng Qwen cục bộ "
                "chưa hoàn tất phần tổng hợp có căn cứ nên chưa sinh ra câu trả lời. "
                "Hãy xem các trang tài liệu công bố chính thức được liệt kê bên dưới."
                f"\n\nNguyên nhân: {detail}"
            )
        elapsed = 0.0
        generation_fallback = "ollama_unavailable"
    return {
        "question": question,
        "answer": answer_text,
        "sources": [
            {
                "ticker": c["ticker"], "doctype": c["doctype"],
                "source_file": c["source_file"], "page": c["page"],
                "year": c.get("year"),
                "similarity": round(c["similarity"], 4),
            }
            for c in chunks
        ],
        "elapsed_s": round(elapsed, 2),
        "answer_lang": lang,
        "generation_fallback": generation_fallback,
    }
