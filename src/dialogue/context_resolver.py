"""RQ7 Conversational Context Resolver - dataclasses + the 5 baselines
compared in the RQ7 evaluation (see docs/Ghi_chu_thao_luan_phan_bien_2026-07-26.md
section 8.10 for the design decision this implements, chốt 2026-07-28).

Placement in the pipeline: this module sits BEFORE src/router/rules.py.
Every question first passes through a resolver here, which returns a
ResolvedQuery whose `.standalone_question` is what actually gets handed to
`classify_intent()`. A single-turn question with no history is unaffected
(NoHistoryResolver / any resolver on a self_contained conversation should
return the input unchanged).

Two things this module deliberately keeps SEPARATE (a recurring point in the
discussion notes, since conflating them was an early design mistake):
  1. GENERAL RULES of how to rewrite - this is what gets fine-tuned (mT5 +
     LoRA, see scripts/finetune_context_resolver.py) or hard-coded (the 3
     non-learned baselines below). Fixed after training; does not change
     within or across sessions.
  2. PER-SESSION CONVERSATION MEMORY - which ticker/year/indicator THIS
     particular conversation has been talking about so far. This is just
     data (a small in-memory/JSON-serializable object, see
     `SessionMemory` below), never part of the model's weights. A model
     fine-tuned once must still be told "in THIS conversation, the last
     ticker mentioned was HPG" via the memory object passed alongside the
     current utterance - it cannot have memorized any real session's
     history at training time.

Zero top-level heavy imports for the parts that don't need them (dataclasses,
regex, the 3 non-LLM baselines) - same "testable everywhere" principle as
src/evaluation/metrics.py. The two baselines that need an external
model/service (LLMPromptOnlyResolver -> Ollama, FineTunedContextResolver ->
a local mT5+LoRA checkpoint) import their dependencies lazily, inside the
method that needs them, so importing this module never requires either to
be installed/running.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

# Deliberately duplicated from src.structured.query_builder's TICKER_RE/
# NON_TICKER_ABBREVIATIONS rather than imported - query_builder.py imports
# sqlalchemy at module level (a real DB dependency), which would make this
# module (and its baseline resolvers, which have nothing to do with the DB)
# un-importable/un-unit-testable without a live Postgres connection. Keep
# this list in sync by hand if query_builder.py's list changes - a small,
# accepted duplication cost for keeping this module dependency-free (same
# tradeoff already made deliberately for src/evaluation/metrics.py).
_TICKER_RE = re.compile(r"\b([A-Z]{3,4})\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_NON_TICKER_ABBREVIATIONS = {
    "EPS", "ROE", "ROA", "ROI", "ROS", "EBIT", "EBITDA", "CEO", "CFO", "CFA",
    "GTGT", "TNDN", "KQKD", "BCTC", "LNST", "LNTT", "VND", "USD", "TSCĐ",
    "HOSE", "HNX", "MS", "PE", "PB", "PS", "COGS", "OCF",
}

# Pronoun/ellipsis markers that signal "this question depends on prior
# context" - used only by the non-learned baselines (2/RawConcat doesn't
# need this list; NoHistory doesn't either). Not exhaustive by design - the
# whole point of RQ7 is that a fixed keyword list like this one is a WEAK
# baseline, expected to be beaten by the fine-tuned resolver.
_REFERENCE_MARKERS = [
    r"\bnó\b", r"\bcon số đó\b", r"\bcái đó\b", r"\bđiều đó\b",
    r"\bcòn\b.*\bthì sao\b", r"\bcòn\b.*\bthế nào\b", r"\bthì sao\b",
]


def _norm_tokens(text: str) -> list[str]:
    """Token thường hoá, bỏ dấu câu - dùng cho so khớp từ vựng trong
    SessionMemory.resolve_ticker_for(). Cố ý KHÔNG bỏ dấu tiếng Việt và KHÔNG
    tách từ ghép: hai câu đang so là cùng ngôn ngữ, cùng khung câu, nên trùng
    token thô đã đủ phân biệt (xem resolve_ticker_for's docstring)."""
    return re.sub(r"[^\w\s]", " ", (text or "").lower()).split()


def extract_entities(question: str) -> dict:
    """Best-effort ticker/year extraction from a single utterance, no DB
    round-trip (company-NAME resolution, unlike query_builder._extract_ticker,
    is intentionally NOT attempted here - this function is used to score
    what a resolver's OUTPUT already contains, not to do the resolving
    itself)."""
    ticker = None
    for m in _TICKER_RE.finditer(question):
        if m.group(1) not in _NON_TICKER_ABBREVIATIONS:
            ticker = m.group(1)
            break
    years = [int(y) for y in _YEAR_RE.findall(question)]
    return {"ticker": ticker, "year": years[-1] if years else None}


@dataclass
class ConversationTurn:
    turn_idx: int
    speaker: str  # "user" - resolvers only ever look at user turns in this project's scope
    utterance: str
    source_benchmark_id: Optional[str] = None  # traces back to data/eval/benchmark.jsonl, when applicable


@dataclass
class EntityMention:
    """Một lần một thực thể được nhắc tới trong phiên, kèm ngữ cảnh câu nói
    chứa nó. Là đơn vị lưu của SessionMemory.mentions (nhiều ô) - xem
    SessionMemory's docstring cho lý do phải nhiều ô."""
    turn_idx: int
    utterance: str
    ticker: Optional[str] = None
    year: Optional[int] = None


@dataclass
class SessionMemory:
    """Per-session state - see module docstring's point 2. Deliberately a
    plain dataclass (JSON-serializable via dataclasses.asdict()), not part of
    any model - a fresh SessionMemory() per conversation, updated turn by
    turn as resolved entities come in.

    **Sửa 2026-08-14 - từ MỘT ô ghi đè sang NHIỀU ô + chọn theo liên quan.**

    Thiết kế cũ chỉ có `resolved_ticker` (1 ô, luôn ghi đè):

        if resolved.ticker:
            self.resolved_ticker = resolved.ticker

    Đo thật trên cả 8 item `return_to_old_topic` của
    data/eval/rq7_conversational_benchmark.jsonl: **SAI 7/8 (88%)**. Lý do cấu
    trúc, không phải lỗi cài đặt: hội thoại A -> B (ngắt quãng) -> "nó" (ý là A)
    thì sau lượt B ô nhớ đã bị B ghi đè, không còn đường nào lấy lại A. Vd
    rq7_return_topic_033: CTD -> PNJ -> "COGS của nó" (gold = CTD), ô nhớ = PNJ.

    Cách sửa KHÔNG cần học máy: giữ TOÀN BỘ lịch sử nhắc tới (`mentions`,
    append-only) rồi khi cần giải một đại từ thì CHỌN theo độ liên quan giữa
    câu hiện tại và câu chứa từng thực thể (`resolve_ticker_for()`), thay vì
    mặc định lấy cái mới nhất.

    `resolved_ticker`/`resolved_year`/`resolved_indicator` GIỮ NGUYÊN ngữ nghĩa
    "cái gần nhất" để không phá code/test đã có - chúng vẫn là field của
    dataclass (không đổi thành property) vì test hiện tại khởi tạo trực tiếp
    `SessionMemory(session_id=..., resolved_ticker="HPG")`.
    """
    session_id: str
    resolved_ticker: Optional[str] = None
    resolved_indicator: Optional[str] = None
    resolved_year: Optional[int] = None
    previous_branch: Optional[str] = None
    previous_sources: list = field(default_factory=list)
    mentions: list = field(default_factory=list)  # list[EntityMention], append-only

    def update_from(self, resolved: "ResolvedQuery", turn_idx: Optional[int] = None,
                    utterance: Optional[str] = None) -> None:
        """Cập nhật cả 2 tầng: các ô "gần nhất" (tương thích ngược) VÀ lịch sử
        `mentions` (dùng cho resolve_ticker_for). `turn_idx`/`utterance` là tuỳ
        chọn để mọi lời gọi cũ (chỉ truyền `resolved`) vẫn chạy nguyên."""
        if resolved.ticker:
            self.resolved_ticker = resolved.ticker
        if resolved.indicator:
            self.resolved_indicator = resolved.indicator
        if resolved.fiscal_year:
            self.resolved_year = resolved.fiscal_year
        if resolved.ticker or resolved.fiscal_year:
            self.mentions.append(EntityMention(
                turn_idx=turn_idx if turn_idx is not None else len(self.mentions),
                utterance=utterance if utterance is not None else resolved.standalone_question,
                ticker=resolved.ticker,
                year=resolved.fiscal_year,
            ))

    def resolve_ticker_for(self, current_utterance: str) -> Optional[str]:
        """Chọn ticker phù hợp NHẤT với `current_utterance` trong số các mã đã
        được nhắc, dựa trên độ trùng từ vựng giữa câu hiện tại và câu đã chứa
        mã đó.

        Vì sao trùng từ vựng là đủ (không cần embedding): ca `return_to_old_topic`
        là "quay lại đúng chủ đề cũ", nên câu quay lại dùng gần như NGUYÊN VĂN
        khung câu cũ, chỉ thay tên mã bằng đại từ. Vd:
            cũ  : "Nếu COGS của CTD tăng 5% trong 2025, lợi nhuận gộp sẽ ra sao?"
            mới : "Nếu COGS của nó  tăng 5% trong 2025, lợi nhuận gộp sẽ ra sao?"
        Trùng từ vựng ở đây rất cao, trong khi câu ngắt quãng ("Công ty kiểm toán
        của PNJ năm 2024 là ai?") gần như không trùng. Đây cũng đúng hướng bằng
        chứng ngoài: trên RAGRouter-Bench (7,727 truy vấn), đặc trưng từ vựng
        TF-IDF THẮNG sentence-embedding ngữ nghĩa 3.1 điểm macro-F1
        (arXiv:2604.03455) - xem docs/AgeMem_phan_tich_2026-08-11.md §6.1.

        Khi hoà điểm (vd chỉ có đúng 1 mã từng được nhắc, hoặc 2 câu trùng bằng
        nhau) thì lấy lượt GẦN NHẤT - giữ đúng hành vi cũ cho các ca đơn giản
        (pronoun_reference, ellipsis) vốn đã đúng, để bản sửa này không gây hồi quy.
        """
        candidates = [m for m in self.mentions if m.ticker]
        if not candidates:
            return None
        cur = set(_norm_tokens(current_utterance))
        best, best_key = None, None
        for m in candidates:
            overlap = len(cur & set(_norm_tokens(m.utterance)))
            key = (overlap, m.turn_idx)  # trùng nhiều hơn thắng; hoà thì lượt sau thắng
            if best_key is None or key > best_key:
                best, best_key = m.ticker, key
        return best


@dataclass
class ResolvedQuery:
    standalone_question: str
    ticker: Optional[str] = None
    period: Optional[str] = None
    fiscal_year: Optional[int] = None
    indicator: Optional[str] = None
    confidence: float = 1.0
    resolver_name: str = ""


# ---------------------------------------------------------------------------
# Baseline 1/5: no history at all - the naive floor. Passes the current
# utterance straight to the router unchanged. Expected to fail on every
# context-dependent turn (pronoun/ellipsis/subject-switch/self-correction),
# and succeed trivially on self_contained/noisy_irrelevant_history turns
# (since those are already standalone regardless of history).
# ---------------------------------------------------------------------------
class NoHistoryResolver:
    name = "no_history"

    def resolve(self, history: list[ConversationTurn], current: str,
                memory: Optional[SessionMemory] = None) -> ResolvedQuery:
        entities = extract_entities(current)
        return ResolvedQuery(standalone_question=current, ticker=entities["ticker"],
                              fiscal_year=entities["year"], resolver_name=self.name)


# ---------------------------------------------------------------------------
# Baseline 2/5: raw concatenation of the last k turns. Cheap, no model call,
# but produces an ungrammatical/run-on "standalone" question and carries
# forward EVERY entity mentioned in the window (including irrelevant ones -
# this is exactly what the noisy_irrelevant_history error category is
# designed to penalize).
# ---------------------------------------------------------------------------
class RawConcatResolver:
    name = "raw_concat"

    def __init__(self, k: int = 3):
        self.k = k

    def resolve(self, history: list[ConversationTurn], current: str,
                memory: Optional[SessionMemory] = None) -> ResolvedQuery:
        window = [t.utterance for t in history[-self.k:]] + [current]
        concatenated = " ".join(window)
        entities = extract_entities(concatenated)
        return ResolvedQuery(standalone_question=concatenated, ticker=entities["ticker"],
                              fiscal_year=entities["year"], resolver_name=self.name)


# ---------------------------------------------------------------------------
# Baseline 3/5: LLM-prompt-only rewriting - the SAME local Qwen3 model
# already used everywhere else in this project (src/models/llm_factory.py,
# via Ollama), asked via a zero-shot prompt to rewrite the follow-up into a
# standalone question. No fine-tuning at all - this is the baseline that
# tests whether prompting alone (no training) is already good enough, which
# is exactly the question RQ7 has to answer to justify fine-tuning as a real
# contribution (see docs/Ghi_chu_thao_luan_phan_bien_2026-07-26.md 8.10 -
# "why fine-tuning alone doesn't solve long-context problems" cuts both
# ways: it also means a good prompt might not need fine-tuning either, and
# that has to be measured, not assumed).
# ---------------------------------------------------------------------------
class LLMPromptOnlyResolver:
    name = "llm_prompt_only"

    _SYSTEM = (
        "Bạn là bộ giải quyết ngữ cảnh hội thoại cho hệ thống hỏi-đáp tài chính. "
        "Nhiệm vụ DUY NHẤT: viết lại câu hỏi cuối cùng của người dùng thành một câu hỏi "
        "độc lập, đầy đủ, không cần ngữ cảnh trước đó, dựa vào lịch sử hội thoại. "
        "KHÔNG trả lời câu hỏi. CHỈ trả về câu hỏi đã viết lại, không giải thích thêm."
    )

    # 2026-09-24: bản tiếng Anh của _SYSTEM. CHỈ dùng khi caller truyền
    # lang="en" - mặc định vẫn là prompt tiếng Việt y nguyên, nên mọi
    # baseline/benchmark đã chạy (resolver này là baseline 3/5 của RQ3, xem
    # docstring module) giữ nguyên hành vi và con số vẫn tái lập được.
    #
    # Vì sao cần: prompt tiếng Việt yêu cầu "viết lại thành câu hỏi độc
    # lập" nên model có xu hướng trả lời BẰNG TIẾNG VIỆT kể cả khi câu hỏi
    # gốc là tiếng Anh. Câu trả lời cuối vẫn ra tiếng Anh (route() ghim
    # ngôn ngữ theo câu người dùng gõ), nhưng dòng "Understood as:" hiện
    # trên giao diện lại là tiếng Việt - giữa một buổi bảo vệ bằng tiếng
    # Anh thì đó đúng là chỗ người ta nhìn thấy đầu tiên.
    _SYSTEM_EN = (
        "You are a conversation-context resolver for a financial question-answering "
        "system. Your ONLY task: rewrite the user's last question into a standalone, "
        "self-contained question that needs no prior context, using the conversation "
        "history. Do NOT answer the question. Return ONLY the rewritten question, with "
        "no explanation. Write the rewritten question in English."
    )

    def resolve(self, history: list[ConversationTurn], current: str,
                memory: Optional[SessionMemory] = None,
                lang: Optional[str] = None) -> ResolvedQuery:
        from src.models.llm_factory import generate  # lazy import - see module docstring

        history_text = "\n".join(f"- {t.utterance}" for t in history)
        if lang == "en":
            prompt = (
                f"Conversation history:\n{history_text}\n\n"
                f"Last question to rewrite: {current}\n\n"
                f"Standalone question:"
            )
            system = self._SYSTEM_EN
        else:
            prompt = (
                f"Lịch sử hội thoại:\n{history_text}\n\n"
                f"Câu hỏi cuối cùng cần viết lại: {current}\n\n"
                f"Câu hỏi độc lập:"
            )
            system = self._SYSTEM
        rewritten = generate(prompt, system=system, temperature=0.0).strip()
        entities = extract_entities(rewritten)
        return ResolvedQuery(standalone_question=rewritten, ticker=entities["ticker"],
                              fiscal_year=entities["year"], resolver_name=self.name)


# ---------------------------------------------------------------------------
# Baseline 4/5: the actual fine-tuned model this RQ is about - mT5-small
# (300M, default) or mT5-base (580M, stretch goal) + LoRA
# (rank r=8/16, alpha=16/32, dropout=0.05, encoder-decoder Q/V attention
# matrices only, ~0.5-1% of params updated), trained by
# scripts/finetune_context_resolver.py on data/eval/rq7_conversational_benchmark.jsonl.
#
# NOT runnable inside this sandbox: downloading google/mt5-small from the
# HuggingFace Hub needs network access this sandbox does not have, and
# training needs a GPU (or at least a fast CPU/MPS) this sandbox does not
# have either - both must happen on the project owner's own machine (see
# scripts/finetune_context_resolver.py's own docstring for exact commands).
# This class is a real, correctly-shaped implementation (loads a real
# checkpoint via transformers+peft when one exists), not a mock - it just
# has nothing to load until the fine-tuning script has actually been run
# once, hence the clear error below rather than a silent fallback.
# ---------------------------------------------------------------------------
class FineTunedContextResolver:
    name = "finetuned_mt5_lora"
    DEFAULT_CHECKPOINT_DIR = "models/context_resolver_lora"

    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.checkpoint_dir = checkpoint_dir or self.DEFAULT_CHECKPOINT_DIR
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import os
        if not os.path.isdir(self.checkpoint_dir):
            raise FileNotFoundError(
                f"No fine-tuned checkpoint at {self.checkpoint_dir!r}. Run "
                f"scripts/finetune_context_resolver.py first (needs network to "
                f"download google/mt5-small + a GPU/MPS device - see that "
                f"script's docstring for the exact command)."
            )
        from peft import PeftModel  # lazy import - only needed once a checkpoint exists
        from transformers import AutoTokenizer, MT5ForConditionalGeneration

        base = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        self._model = PeftModel.from_pretrained(base, self.checkpoint_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)

    def resolve(self, history: list[ConversationTurn], current: str,
                memory: Optional[SessionMemory] = None) -> ResolvedQuery:
        self._load()
        history_text = " | ".join(t.utterance for t in history)
        input_text = f"resolve: {history_text} </s> {current}"
        inputs = self._tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
        output_ids = self._model.generate(**inputs, max_new_tokens=64)
        rewritten = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        entities = extract_entities(rewritten)
        return ResolvedQuery(standalone_question=rewritten, ticker=entities["ticker"],
                              fiscal_year=entities["year"], resolver_name=self.name)


# ---------------------------------------------------------------------------
# Baseline 5/5: human oracle - the upper bound. Only usable against
# data/eval/rq7_conversational_benchmark.jsonl (or any dataset shaped the
# same way), which already carries a `gold_standalone_question` per item
# written by scripts/build_rq7_dataset.py FROM a real, already-verified
# single-turn benchmark.jsonl question - never a freshly-invented value (see
# that script's docstring: "no invented ground truth" applies here too).
# ---------------------------------------------------------------------------
class HumanOracleResolver:
    name = "human_oracle"

    def resolve_from_item(self, item: dict) -> ResolvedQuery:
        entities = extract_entities(item["gold_standalone_question"])
        return ResolvedQuery(
            standalone_question=item["gold_standalone_question"],
            ticker=item.get("gold_ticker") or entities["ticker"],
            fiscal_year=entities["year"],
            resolver_name=self.name,
        )


# ---------------------------------------------------------------------------
# Baseline 6/6 (thêm 2026-08-14): giải ngữ cảnh bằng BỘ NHỚ NHIỀU Ô, không học,
# không gọi LLM. Đây là baseline còn thiếu trong thiết kế RQ7 ban đầu: 3
# baseline không-học đầu tiên đều BỎ QUA tham số `memory` (kiểm chứng:
# `grep "memory\."` trước 14/08 trả về 0 dòng), nên `SessionMemory` tuy được
# viết và test đầy đủ nhưng chưa từng được đường chạy nào dùng tới.
#
# Vì sao baseline này đáng có RIÊNG, thay vì chỉ chờ mô hình fine-tune:
# nếu một cơ chế nhớ không học đã giải được phần lớn 8 loại lỗi, thì phần
# fine-tune (mT5+LoRA trên 64 item) phải chứng minh nó thêm được gì NGOÀI cái
# này - nếu không, "cải thiện nhờ fine-tune" thực chất chỉ là cái mà một
# cấu trúc dữ liệu đúng đã cho không mất đồng nào. Đây chính là dạng đối chứng
# mà MemDelta (arXiv:2606.29914) chỉ ra là hay bị bỏ sót trong literature về
# agent memory: "agent self-memory (42%) thua basic retrieval (47%)".
# ---------------------------------------------------------------------------
class MemoryAwareResolver:
    """Quy tắc, theo thứ tự:

    1. Nếu câu hiện tại TỰ NÓ đã nêu mã -> dùng luôn mã đó, KHÔNG hỏi bộ nhớ.
       Đây là điều kiện giữ đúng `subject_switch` ("Còn HPG thì sao?") và
       `self_correction` ("À không, ý tôi là XYZ cơ") - ở hai loại này việc
       ĐỔI chủ thể mới là hành vi đúng, nên bộ nhớ không được phép can thiệp.
    2. Ngược lại (đại từ/tỉnh lược) -> hỏi `memory.resolve_ticker_for()`, chọn
       theo độ liên quan chứ không mặc định lấy mã gần nhất. Đây là điều kiện
       giải được `return_to_old_topic`.
    3. Năm: lấy trong câu nếu có, nếu không thì lấy năm của chính lượt đã cung
       cấp mã được chọn (không lấy năm gần nhất một cách mù quáng).

    Không sinh lại câu hỏi bằng LLM - trả về câu gốc kèm thực thể đã giải. Với
    RQ7 thì `predicted_ticker`/`predicted_route` mới là chỉ số quyết định
    (xem src/evaluation/context_resolver_metrics.py), không phải câu chữ.
    """
    name = "memory_aware"

    def resolve(self, history: list[ConversationTurn], current: str,
                memory: Optional[SessionMemory] = None) -> ResolvedQuery:
        entities = extract_entities(current)
        ticker, year = entities["ticker"], entities["year"]

        mem = memory
        if mem is None:
            # Không được cấp bộ nhớ -> tự dựng từ history, để resolver dùng
            # được cả trong khung đánh giá chỉ truyền (history, current).
            mem = SessionMemory(session_id="ad-hoc")
            for t in history:
                e = extract_entities(t.utterance)
                mem.update_from(
                    ResolvedQuery(standalone_question=t.utterance,
                                  ticker=e["ticker"], fiscal_year=e["year"]),
                    turn_idx=t.turn_idx, utterance=t.utterance,
                )

        if ticker is None:
            ticker = mem.resolve_ticker_for(current)
            if year is None and ticker is not None:
                for m in reversed(mem.mentions):
                    if m.ticker == ticker and m.year:
                        year = m.year
                        break
        if year is None:
            year = mem.resolved_year

        return ResolvedQuery(standalone_question=current, ticker=ticker,
                             fiscal_year=year, resolver_name=self.name)


ALL_BASELINES = [
    NoHistoryResolver, RawConcatResolver, LLMPromptOnlyResolver,
    FineTunedContextResolver, HumanOracleResolver, MemoryAwareResolver,
]
