"""Test cho bản sửa SessionMemory 1-ô -> NHIỀU-ô (2026-08-14).

Bối cảnh: `SessionMemory.update_from()` cũ chỉ giữ MỘT ô `resolved_ticker` và
luôn ghi đè. Mô phỏng trên cả 8 item `return_to_old_topic` của
data/eval/rq7_conversational_benchmark.jsonl cho kết quả SAI 7/8 (88%) - lỗi
cấu trúc, không phải lỗi cài đặt: hội thoại A -> B (ngắt quãng) -> "nó" (ý là A)
thì sau lượt B, A đã bị ghi đè và không còn đường lấy lại.

HAI NHÓM TEST QUAN TRỌNG NGANG NHAU:

  - nhóm "phải sửa được": `return_to_old_topic` giờ phải chọn đúng mã cũ.
  - nhóm "KHÔNG được làm hỏng": `subject_switch` và `self_correction` là những
    loại mà ĐỔI chủ thể mới là hành vi ĐÚNG. Một bản sửa quá tay (luôn ưu tiên
    mã cũ vì "liên quan hơn") sẽ sửa được nhóm 1 nhưng phá nhóm 2 - đây là đánh
    đổi thật đã được cảnh báo trước khi viết code, nên phải có lưới an toàn.

Mọi ca đều lấy TỪ DỮ LIỆU THẬT trong rq7_conversational_benchmark.jsonl.

Chạy:
    python3 -m unittest tests.test_dialogue.test_session_memory_multislot -v
Không cần DB/LLM/mạng.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dialogue.context_resolver import (  # noqa: E402
    ConversationTurn,
    MemoryAwareResolver,
    ResolvedQuery,
    SessionMemory,
    extract_entities,
)

BENCHMARK = ROOT / "data" / "eval" / "rq7_conversational_benchmark.jsonl"


def _load(category: str) -> list:
    if not BENCHMARK.exists():
        return []
    items = [json.loads(l) for l in BENCHMARK.open(encoding="utf-8") if l.strip()]
    return [i for i in items if i.get("error_category") == category]


def _resolve_item(item: dict) -> str | None:
    """Chạy MemoryAwareResolver đúng như khung đánh giá sẽ chạy: nạp mọi lượt
    TRƯỚC lượt cuối vào bộ nhớ, rồi giải lượt cuối."""
    turns = item["turns"]
    history = [ConversationTurn(turn_idx=t["turn_idx"], speaker=t["speaker"],
                                utterance=t["utterance"]) for t in turns[:-1]]
    memory = SessionMemory(session_id=item["conv_id"])
    for t in history:
        e = extract_entities(t.utterance)
        memory.update_from(
            ResolvedQuery(standalone_question=t.utterance,
                          ticker=e["ticker"], fiscal_year=e["year"]),
            turn_idx=t.turn_idx, utterance=t.utterance,
        )
    return MemoryAwareResolver().resolve(history, turns[-1]["utterance"], memory).ticker


class TestOldSingleSlotBehaviourWasBroken(unittest.TestCase):
    """Ghi lại BẰNG CHỨNG của lỗi cũ, để nếu ai đó quay lại thiết kế 1-ô thì
    test này chỉ ra ngay vì sao không được."""

    def test_single_slot_overwrite_loses_the_earlier_topic(self):
        mem = SessionMemory(session_id="demo")
        mem.update_from(ResolvedQuery(standalone_question="Doanh thu CTD 2025?", ticker="CTD"),
                        turn_idx=0, utterance="Doanh thu CTD 2025?")
        mem.update_from(ResolvedQuery(standalone_question="Kiểm toán PNJ 2024?", ticker="PNJ"),
                        turn_idx=1, utterance="Kiểm toán PNJ 2024?")
        # Ô "gần nhất" vẫn giữ nguyên ngữ nghĩa cũ (tương thích ngược):
        self.assertEqual(mem.resolved_ticker, "PNJ")
        # ...nhưng mã cũ KHÔNG còn mất nữa - đây chính là điểm sửa:
        self.assertEqual({m.ticker for m in mem.mentions}, {"CTD", "PNJ"})


class TestReturnToOldTopicNowFixed(unittest.TestCase):
    """Nhóm 'phải sửa được'. Trước bản sửa: sai 7/8. Sau: đúng 4/8.

    4 ca còn lại KHÔNG phải lỗi cài đặt mà là **bất khả phân giải bằng từ vựng**,
    và đây là phát hiện về chất lượng benchmark chứ không phải về resolver:

        [0] "Công ty kiểm toán của CTD năm 2025 là ai?"
        [1] "Công ty kiểm toán của CSC năm 2025 là ai?"   <- CÙNG khung câu
        [2] "Công ty kiểm toán của nó  năm 2025 là ai?"   <- gold = CTD

    Trùng từ vựng giữa lượt 2 với lượt 0 và với lượt 1 đều = 9, hoà tuyệt đối.
    Thông tin để chọn CTD thay vì CSC **không nằm trong câu chữ**. Một người đọc
    cũng sẽ hiểu "nó" là CSC (lượt gần nhất) - tức nhãn vàng của 4 ca này đáng
    ngờ, vì scripts/build_rq7_dataset.py sinh chúng bằng cách chèn một lượt ngắt
    quãng CÙNG khuôn mẫu, khiến "quay lại chủ đề cũ" không phân biệt được với
    "tiếp tục chủ đề mới".

    4 ca giải được là 4 ca lượt ngắt quãng dùng KHUNG CÂU KHÁC (vd whatif chen
    giữa hai câu rag) - khi đó trùng từ vựng phân biệt rõ ràng.

    Test này KHÓA lại đúng ranh giới đó: 4 ca phân biệt được thì phải đúng; 4 ca
    hoà thì ghi nhận là giới hạn đã biết, KHÔNG hack tie-break để ép xanh (làm
    vậy sẽ phá `subject_switch`, nơi lượt gần nhất mới đúng).
    """

    @staticmethod
    def _same_template(item) -> bool:
        t0, t1 = item["turns"][0]["utterance"], item["turns"][1]["utterance"]
        return t0.split()[:3] == t1.split()[:3]

    def test_lexically_distinguishable_items_are_fixed(self):
        items = [i for i in _load("return_to_old_topic") if not self._same_template(i)]
        self.assertTrue(items, "không đọc được benchmark RQ7")
        wrong = [(i["conv_id"], _resolve_item(i), i["gold_ticker"])
                 for i in items if _resolve_item(i) != i["gold_ticker"]]
        self.assertEqual(wrong, [], f"ca phân biệt được mà vẫn sai: {wrong}")

    def test_ambiguous_items_documented_not_silently_passing(self):
        """Ghi nhận giới hạn: 4 ca cùng khung câu vẫn chọn lượt gần nhất.
        Nếu ai đó sửa được chúng MÀ KHÔNG phá subject_switch, test này sẽ đỏ và
        đó là tin tốt - hãy cập nhật lại con số kèm bằng chứng."""
        items = [i for i in _load("return_to_old_topic") if self._same_template(i)]
        n_wrong = sum(1 for i in items if _resolve_item(i) != i["gold_ticker"])
        self.assertEqual(
            n_wrong, len(items),
            "Có ca hoà từ vựng bỗng đúng - kiểm tra xem có phải do tie-break bị "
            "sửa thiên vị lượt cũ không (sẽ phá subject_switch).",
        )


class TestMustNotRegress(unittest.TestCase):
    """Nhóm 'KHÔNG được làm hỏng' - lưới an toàn cho đánh đổi đã cảnh báo trước.

    Ở subject_switch và self_correction, ĐỔI mã mới là đúng. Nếu ai sửa
    resolve_ticker_for() theo hướng luôn thiên vị mã cũ, các test này vỡ TRƯỚC
    khi con số tổng thể trông có vẻ đẹp lên.
    """

    def _assert_category_all_correct(self, category: str):
        items = _load(category)
        self.assertTrue(items, f"không có item nào cho {category}")
        wrong = []
        for it in items:
            got = _resolve_item(it)
            if got != it["gold_ticker"]:
                wrong.append((it["conv_id"], got, it["gold_ticker"]))
        self.assertEqual(wrong, [], f"{category} bị hồi quy: {wrong}")

    def test_subject_switch_still_follows_the_new_ticker(self):
        """6/8 đúng. 2 ca sai KHÔNG do bộ nhớ mà do `extract_entities()` không
        nhận ra mã ngay trong chính câu hiện tại - đã sai SẴN trước bản sửa này
        (kiểm chứng: NoHistoryResolver cũng trả None cho cả hai):

          - rq7_subj_switch_027, gold 'CEO'  : 'CEO' nằm trong
            _NON_TICKER_ABBREVIATIONS (chức danh giám đốc), nhưng CEO cũng là mã
            thật (CTCP Tập đoàn C.E.O) -> nhập nhằng thật của tiếng Việt.
          - rq7_subj_switch_031, gold 'AISEC': 5 ký tự, trong khi
            _TICKER_RE chỉ khớp 3-4 ký tự.

        Cố ý KHÔNG sửa hai chỗ đó ở đây: nới _TICKER_RE lên 5 ký tự hoặc bỏ 'CEO'
        khỏi danh sách loại trừ sẽ tạo false positive ở nơi khác, phải đo riêng.
        Ghi nhận là việc tồn đọng của extract_entities, không phải của SessionMemory.
        """
        items = _load("subject_switch")
        wrong = [(i["conv_id"], _resolve_item(i), i["gold_ticker"])
                 for i in items if _resolve_item(i) != i["gold_ticker"]]
        self.assertEqual(
            sorted(c for c, _, _ in wrong),
            ["rq7_subj_switch_027", "rq7_subj_switch_031"],
            f"tập ca sai đã đổi - kiểm tra hồi quy thật: {wrong}",
        )

    def test_self_correction_still_follows_the_corrected_ticker(self):
        self._assert_category_all_correct("self_correction")

    def test_pronoun_reference_still_correct(self):
        self._assert_category_all_correct("pronoun_reference")

    def test_self_contained_still_correct(self):
        self._assert_category_all_correct("self_contained")


class TestBackwardCompatibility(unittest.TestCase):
    """Chữ ký cũ `update_from(resolved)` (không truyền turn_idx/utterance) phải
    vẫn chạy - mọi lời gọi cũ trong repo/test dựa vào điều này."""

    def test_update_from_works_without_new_optional_args(self):
        mem = SessionMemory(session_id="s1")
        mem.update_from(ResolvedQuery(standalone_question="q", ticker="HPG",
                                      fiscal_year=2025, indicator="revenue"))
        self.assertEqual(mem.resolved_ticker, "HPG")
        self.assertEqual(mem.resolved_year, 2025)
        self.assertEqual(len(mem.mentions), 1)

    def test_none_does_not_clobber_existing_slot(self):
        mem = SessionMemory(session_id="s1", resolved_ticker="HPG")
        mem.update_from(ResolvedQuery(standalone_question="q", ticker=None))
        self.assertEqual(mem.resolved_ticker, "HPG")

    def test_resolve_returns_none_when_memory_empty(self):
        mem = SessionMemory(session_id="empty")
        self.assertIsNone(mem.resolve_ticker_for("Doanh thu của nó là bao nhiêu?"))


if __name__ == "__main__":
    unittest.main()
