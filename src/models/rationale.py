"""Natural-language rationale generation for the Causal/What-if branch's
already-computed, fully-transparent numeric results (bridge decomposition,
sensitivity analysis, ATE estimates).

NOT explainable AI (XAI) in the formal sense (SHAP/LIME/Anchors), and this
module is deliberately NOT named/framed that way - see the 2026-07-19
discussion in docs/PROGRESS.md for the full reasoning. XAI techniques exist
to probe an OPAQUE model's internals ("why did this black box predict X").
src/structured/calculator.py's formulas aren't opaque - gross_profit =
revenue - cogs is fully transparent already; there's nothing to reverse-
engineer. What this module does is narrower and more honestly described:
turn an already-correct, already-interpretable numeric result into readable
Vietnamese prose for a reader who isn't going to parse a formula - the same
"verbalize an interpretable signal" pattern used by Ploutos
(arXiv:2403.00782, PloutosGPT) for stock-movement rationale generation, not
"open a black box" (there isn't one here).

If genuine XAI is wanted somewhere in this project, RQ5's ABSA branch is the
right place - IF it ends up using a learned/black-box sentiment classifier,
LIME or Anchors could explain individual classifications. That's separate,
unbuilt work, not this module.

Grounding rule, same principle as src/models/generation.py's RAG prompt: the
LLM is given the EXACT computed numbers (the already-formatted Vietnamese
string from query_router.py's _format_whatif()/_format_causal()) and
instructed to explain them, not invent new figures, causes, or assumptions.
The deterministic calculator remains the single source of truth for every
number that appears - this module's only job is presentation.
"""
from src.models import llm_factory
from src.models.callbacks import timed_generation

SYSTEM_PROMPT = (
    "Bạn là trợ lý giải thích kết quả tính toán tài chính cho người đọc không "
    "chuyên. QUY TẮC BẮT BUỘC:\n"
    "1. Chỉ dùng CHÍNH XÁC các số liệu đã cho bên dưới - KHÔNG bịa thêm số, "
    "sự kiện, hay nguyên nhân nào không có trong dữ liệu.\n"
    "2. Giải thích bằng ngôn ngữ tự nhiên, dễ hiểu, như đang nói với một "
    "khách hàng không rành tài chính - đừng liệt kê lại số khô khan, hãy "
    "diễn giải Ý NGHĨA của các con số.\n"
    "3. Nếu có nhiều yếu tố, sắp xếp theo mức độ ảnh hưởng.\n"
    "4. Không đưa ra khuyến nghị đầu tư - chỉ giải thích, không tư vấn.\n"
    "5. Trả lời ngắn gọn (3-5 câu), không sao chép nguyên văn dữ liệu đã cho."
)


def build_prompt(question: str, computed_result: str) -> str:
    return (
        f"Câu hỏi gốc: {question}\n\n"
        f"Kết quả tính toán (đã tính chính xác, chỉ cần diễn giải, không tính lại):\n"
        f"{computed_result}\n\n"
        f"---\n\nHãy giải thích kết quả trên bằng lời văn tự nhiên, dễ hiểu:"
    )


def generate_rationale(question: str, computed_result: str) -> dict:
    """Returns {"rationale", "elapsed_s"}. `computed_result` should be the
    already-formatted deterministic string (e.g. query_router.py's
    _format_whatif()/_format_causal() output) - this function only narrates
    it in plain language, it never computes anything itself and is not
    trusted to introduce new numbers."""
    prompt = build_prompt(question, computed_result)
    text, elapsed = timed_generation(llm_factory.generate, prompt, system=SYSTEM_PROMPT)
    return {"rationale": text, "elapsed_s": round(elapsed, 2)}
