from src.ingestion.vietstock_premium import (
    company_tab_url,
    extract_company_links,
    company_detail_url,
    extract_page_links,
    has_login_wall,
    is_direct_pdf,
    is_company_tab_url,
    merge_link_candidates,
    parse_published_date,
    parse_company_document_links,
    parse_company_news_links,
    safe_error,
)
from src.ingestion.vietstock_structured import (
    extract_financial_subsection_links,
    parse_structured_tables,
    structured_tab_url,
)


def test_extract_company_links_accepts_company_detail_urls_and_deduplicates():
    html = """
    <a href="/VNM-ctcp-sua-viet-nam.htm">Vinamilk</a>
    <a href="https://finance.vietstock.vn/HPG-ctcp-tap-doan-hoa-phat.htm">Hoa Phat</a>
    <a href="/doanh-nghiep-a-z?page=2">Next</a>
    <a href="/VNM-ctcp-sua-viet-nam.htm">Duplicate</a>
    """
    rows = extract_company_links(html, "https://finance.vietstock.vn/doanh-nghiep-a-z?page=1", {"VNM", "HPG"})
    assert [(row.ticker, row.name) for row in rows] == [("VNM", "Vinamilk"), ("HPG", "Hoa Phat")]


def test_company_link_parser_rejects_navigation_slugs_when_a_ticker_whitelist_is_given():
    html = """
    <a href="/du-lieu-nganh.htm">VS 100</a>
    <a href="/lich-su-kien.htm">Event calendar</a>
    <a href="/VNM-ctcp-sua-viet-nam.htm">Vinamilk</a>
    """
    rows = extract_company_links(html, "https://finance.vietstock.vn/doanh-nghiep-a-z?page=1", {"VNM"})
    assert [row.ticker for row in rows] == ["VNM"]


def test_known_company_url_fallback_transliterates_vietnamese_name():
    assert company_detail_url("VNM", "CTCP Sữa Việt Nam") == "https://finance.vietstock.vn/VNM-ctcp-sua-viet-nam.htm"


def test_extract_page_links_keeps_document_and_news_candidates_separate():
    html = """
    <a href="/download/vnm-report.pdf">Báo cáo tài chính 2025</a>
    <a href="/tin-tuc/vnm-event.htm">Tin tức sự kiện 15/08/2026</a>
    <a href="javascript:void(0)">Tài liệu ignored</a>
    """
    docs, news = extract_page_links(html, "https://finance.vietstock.vn/VNM-ctcp-sua-viet-nam.htm")
    assert docs == [{"url": "https://finance.vietstock.vn/download/vnm-report.pdf", "title": "Báo cáo tài chính 2025"}]
    assert news == [{"url": "https://finance.vietstock.vn/tin-tuc/vnm-event.htm", "title": "Tin tức sự kiện 15/08/2026"}]


def test_parse_published_date_is_best_effort_and_does_not_guess():
    assert parse_published_date("Cập nhật 15/08/2026") == __import__("datetime").date(2026, 8, 15)
    assert parse_published_date("No date") is None


def test_candidate_merge_and_pdf_detection_keep_navigation_out_of_pdf_set():
    direct = {"url": "https://finance.vietstock.vn/download/vnm.pdf", "title": "Report"}
    section = {"url": "https://finance.vietstock.vn/VNM-tai-lieu.htm", "title": "Tài liệu"}
    assert is_direct_pdf(direct)
    assert not is_direct_pdf(section)
    assert merge_link_candidates([direct, section], [direct]) == [direct, section]


def test_company_tab_urls_are_deterministic_and_ticker_scoped():
    assert company_tab_url("vnm", "documents") == "https://finance.vietstock.vn/VNM/tai-tai-lieu.htm"
    assert is_company_tab_url("https://finance.vietstock.vn/VNM/tin-tuc-su-kien.htm", "VNM", "news")
    assert not is_company_tab_url("https://finance.vietstock.vn/tai-lieu/bao-cao-tai-chinh.htm", "VNM", "documents")


def test_document_parser_accepts_only_listing_rows_and_not_page_navigation():
    html = """
    <nav><a href="/tai-lieu/bao-cao-tai-chinh.htm">Báo cáo tài chính toàn thị trường</a></nav>
    <table><tr><td>15/08/2026</td><td>Báo cáo tài chính VNM</td>
      <td><a href="/download/document?id=123">Tải file</a></td></tr></table>
    """
    assert parse_company_document_links(html, "https://finance.vietstock.vn/VNM/tai-tai-lieu.htm") == [{
        "url": "https://finance.vietstock.vn/download/document?id=123", "title": "Tải file",
    }]


def test_document_parser_reads_vietstock_document_title_anchors():
    html = """
    <nav><a href="/tai-lieu/bao-cao-tai-chinh.htm">Global category</a></nav>
    <div class="p-t-xs"><p class="i-b-d bg-hover-yellow">
      <a class="doc__ttl--text-link text-link" target="_blank"
         href="https://static2.vietstock.vn/data/HOSE/2025/BCTN/VN/VNM_Baocaothuongnien_2025.pdf">
        <span class="doc__ttl-file-name">Báo cáo thường niên năm 2025</span>
      </a>
    </p></div>
    """
    assert parse_company_document_links(html, "https://finance.vietstock.vn/VNM/tai-tai-lieu.htm") == [{
        "url": "https://static2.vietstock.vn/data/HOSE/2025/BCTN/VN/VNM_Baocaothuongnien_2025.pdf",
        "title": "Báo cáo thường niên năm 2025",
    }]


def test_news_parser_accepts_only_vietstock_company_news_titles():
    html = """
    <a href="/tin-tuc/global.htm">Tin tức điều hướng</a>
    <a class="stock-news__title" href="/VNM/tin-tuc/vnm-result.htm">VNM announces results</a>
    """
    assert parse_company_news_links(html, "https://finance.vietstock.vn/VNM/tin-tuc-su-kien.htm") == [{
        "url": "https://finance.vietstock.vn/VNM/tin-tuc/vnm-result.htm", "title": "VNM announces results",
    }]


def test_login_wall_detection_does_not_treat_ordinary_navigation_as_authentication():
    assert has_login_wall('<a href="/Account/Login">Đăng nhập</a><p>Vui lòng đăng nhập để xem tài liệu</p>')
    assert not has_login_wall('<a href="/Account/Login">Đăng nhập</a><p>Trang doanh nghiệp</p>')


def test_safe_error_drops_playwright_call_log_before_persistence():
    error = RuntimeError("request failed\nCall log:\nheader: Cookie=secret")
    assert safe_error(error) == "RuntimeError: request failed"


def test_structured_routes_are_ticker_scoped():
    assert structured_tab_url("vnm", "financial") == "https://finance.vietstock.vn/VNM/tai-chinh.htm"
    assert structured_tab_url("VNM", "trading") == "https://finance.vietstock.vn/VNM/thong-ke-giao-dich.htm"


def test_financial_subsection_parser_excludes_global_navigation_and_chart():
    html = """
    <nav><a href="/tai-lieu/bao-cao-tai-chinh.htm">Báo cáo tài chính</a></nav>
    <div class="tabs">
      <a href="/VNM/tai-chinh.htm?tab=summary">BCTC Tóm tắt</a>
      <a href="/VNM/tai-chinh.htm?tab=chart">Biểu đồ</a>
      <a href="/VNM/tai-chinh.htm?tab=ratios">Chỉ số tài chính</a>
    </div>
    """
    links = extract_financial_subsection_links(html, "https://finance.vietstock.vn/VNM/tai-chinh.htm", "VNM")
    assert [item["title"] for item in links] == ["BCTC Tóm tắt", "Chỉ số tài chính"]


def test_structured_table_parser_preserves_headers_rows_and_hash():
    html = """
    <h2>Kết quả kinh doanh</h2>
    <table><caption>Kết quả kinh doanh</caption>
      <tr><th>Chỉ tiêu</th><th>Năm 2024</th><th>Năm 2023</th></tr>
      <tr><td>Doanh thu</td><td>100</td><td>90</td></tr>
    </table>
    """
    tables = parse_structured_tables(html, ticker="VNM", section="financial",
                                     subsection="Kết quả kinh doanh",
                                     source_url="https://finance.vietstock.vn/VNM/tai-chinh.htm")
    assert len(tables) == 1
    assert tables[0].headers == ["Chỉ tiêu", "Năm 2024", "Năm 2023"]
    assert tables[0].rows == [["Doanh thu", "100", "90"]]
    assert len(tables[0].table_hash) == 64
