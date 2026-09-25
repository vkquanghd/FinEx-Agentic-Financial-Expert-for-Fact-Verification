"""Persona (vai trò NGHIỆP VỤ) của người dùng - nguồn định nghĩa DUY NHẤT.

Chốt 2026-08-07 sau câu hỏi của người dùng: "chưa xác định được RAG của ta
thực sự phục vụ ai trong công ty, cho role nào cụ thể". Đây không phải chi
tiết trang trí cho chương mở đầu - nó là LUẬN CỨ cho chính kiến trúc:

    Router chỉ có giá trị khi MỘT người dùng hỏi CẢ câu định lượng LẪN định
    tính trong cùng một phiên làm việc. Nếu persona chỉ hỏi số -> BI/SQL là
    đủ, không cần router. Nếu chỉ đọc văn bản -> RAG thuần là đủ.

Vì sao KHÔNG dùng lại `role` của gateway (user/admin): đó là vai trò KIỂM
SOÁT TRUY CẬP (ai được gọi API nào), khác hẳn vai trò nghiệp vụ (ai đang
hỏi, để làm gì). Gộp hai thứ này sẽ làm hỏng phân tích vì mọi persona đều
có role='user'. Xem infra/postgres/migrations/014's comment.

Ràng buộc bối cảnh (quyết định phạm vi): người dùng đã nêu từ đầu "khách
hàng cần bảo mật dữ liệu, không được call API ngoài". Ràng buộc đó tự loại
bỏ nhóm người dùng bên ngoài (nhà đầu tư cá nhân đọc báo cáo công khai -
nhóm đó dùng thẳng ChatGPT là xong, không ai tự dựng RAG local). Nó chỉ có
nghĩa khi người hỏi CÓ QUYỀN truy cập dữ liệu tài chính nội bộ/nhạy cảm ->
mọi persona dưới đây đều là người dùng NỘI BỘ doanh nghiệp hoặc định chế
tài chính.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Persona:
    key: str
    label: str            # nhãn tiếng Việt hiển thị trên giao diện
    description: str      # họ làm gì, hỏi kiểu gì
    primary_branches: list  # nhánh router phục vụ họ nhiều nhất
    needs: list = field(default_factory=list)   # yêu cầu cụ thể
    gaps: list = field(default_factory=list)    # thứ hệ thống CHƯA đáp ứng


PERSONAS: dict[str, Persona] = {
    "analyst": Persona(
        key="analyst",
        label="Chuyên viên phân tích tài chính",
        description=(
            "Phân tích nội bộ (FP&A) hoặc phân tích đầu tư. Hỏi dày đặc câu "
            "định lượng, thường theo chuỗi: tra số -> so kỳ trước -> so đối "
            "thủ -> thử kịch bản."
        ),
        primary_branches=["dag", "dag_trend", "whatif", "compare"],
        needs=[
            "Tra số liệu BCTC theo mã/kỳ nhanh, chính xác tuyệt đối",
            "Xu hướng nhiều kỳ kèm %QoQ/%YoY, không trả lời một con số trơ",
            "So sánh nhiều mã cùng ngành",
            "Kịch bản ceteris-paribus (nếu COGS tăng x% thì biên LN gộp ra sao)",
        ],
        gaps=[
            "Chưa có chỉ số định giá (P/E, P/B, EV/EBITDA)",
            "Chưa xuất được bảng/CSV để đưa thẳng vào mô hình Excel",
        ],
    ),
    "auditor": Persona(
        key="auditor",
        label="Kiểm toán & kế toán nội bộ",
        description=(
            "Kiểm toán nội bộ, kiểm soát tuân thủ, kế toán tổng hợp. Câu hỏi "
            "xoay quanh BẰNG CHỨNG và TÍNH NHẤT QUÁN, không phải xu hướng: "
            "số này lấy từ đâu, ai ký, vì sao hai nguồn lệch nhau."
        ),
        primary_branches=["rag", "auditor_lookup", "dag"],
        needs=[
            "Truy vết tới đúng tài liệu + số trang (không chấp nhận câu trả lời không nguồn)",
            "Biết ai kiểm toán, ý kiến kiểm toán loại gì, đổi hãng kiểm toán năm nào",
            "Phát hiện số liệu đá nhau giữa các nguồn (BCTC năm đã kiểm toán vs BCTC quý)",
            "Đọc được thuyết minh, chính sách kế toán, thay đổi ước tính",
        ],
        gaps=[
            "Chưa có cảnh báo tự động khi cùng một chỉ tiêu lệch giữa nhiều nguồn "
            "(hiện chỉ có note thủ công cho balance_sheet YEAR vs Q4)",
            "Chưa truy vết được thay đổi giữa các bản công bố (restatement history)",
        ],
    ),
    "executive": Persona(
        key="executive",
        label="Lãnh đạo / thiên về business",
        description=(
            "CFO, giám đốc khối, trưởng bộ phận. Không đọc bảng số dài - cần "
            "câu trả lời ngắn, có kết luận, kèm 'vì sao' bằng ngôn ngữ kinh "
            "doanh chứ không phải công thức."
        ),
        primary_branches=["dag_trend", "whatif", "absa", "compare"],
        needs=[
            "Trả lời ngắn gọn, dẫn tới kết luận, không bắt đọc bảng thô",
            "Diễn giải bằng lời vì sao con số thay đổi (đã có rationale cho whatif/causal)",
            "Đọc nhanh sắc thái/rủi ro trong tài liệu công bố (ABSA 5 khía cạnh)",
        ],
        gaps=[
            "rationale hiện TẮT mặc định (with_rationale=False) - persona này lẽ ra nên bật",
            "Chưa có bản tóm tắt điều hành đa chỉ tiêu trong một câu trả lời",
        ],
    ),
    "investor": Persona(
        key="investor",
        label="Định hướng đầu tư",
        description=(
            "Người ra quyết định phân bổ vốn / tư vấn đầu tư nội bộ. Quan tâm "
            "quan hệ nhân quả và bối cảnh thị trường hơn là một con số đơn lẻ."
        ),
        primary_branches=["causal", "compare", "rag", "dag_trend"],
        needs=[
            "So sánh hiệu suất giá nhiều mã, có kiểm soát bối cảnh thị trường",
            "Ước lượng tác động (đòn bẩy -> biên LN ròng) có kiểm soát biến gây nhiễu",
            "Nhạy cảm với chu kỳ thị trường khi xếp hạng nguồn (RQ3)",
            "Kịch bản theo yếu tố vĩ mô thật (giá vàng cho PNJ, giá dầu cho GAS)",
        ],
        gaps=[
            "Panel causal mới có leverage/net_margin và vài tỷ số - còn hẹp",
            "Chưa có nhánh dự báo (predictive, Phase P2 - chưa nối vào router)",
        ],
    ),
}

DEFAULT_PERSONA = "analyst"


def is_valid(key: str) -> bool:
    return key in PERSONAS


def labels() -> dict[str, str]:
    """{key: nhãn tiếng Việt} - dùng cho dropdown giao diện và nhãn biểu đồ."""
    return {k: p.label for k, p in PERSONAS.items()}
