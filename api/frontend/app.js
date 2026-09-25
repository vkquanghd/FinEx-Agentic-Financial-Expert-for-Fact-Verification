/* =========================================================================
   Financial QA - lớp giao diện
   -------------------------------------------------------------------------
   Alpine.js cho trạng thái, Chart.js cho biểu đồ HITL, Lucide cho biểu
   tượng. Không có bước build: đây là tầng trình bày của một demo luận văn,
   thêm toolchain chỉ làm tăng thứ phải bảo trì mà không đổi kết quả.

   Mọi lời gọi mạng đi qua gateway (/api/...), không service nào được gọi
   thẳng - nhờ vậy đổi cổng nội bộ không phải sửa frontend.
   ========================================================================= */

const API = {
  qa: "/api/qa",
  feedback: "/api/feedback",
  auth: "/api/auth",
  // Service "admin" ở gateway MANG router có prefix "/admin" bên trong -
  // nên URL thật sự có "admin" lặp lại 2 lần (không phải lỗi gõ nhầm).
  admin: "/api/admin/admin",
};

// fetch() mặc định gửi kèm cookie same-origin (credentials: "same-origin")
// - đủ để phiên đăng nhập (cookie fqa_session) tự đi kèm mọi request, KHÔNG
// cần cấu hình gì thêm vì frontend và API cùng origin (đều qua gateway).
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    const err = new Error(`${r.status} ${await r.text()}`);
    err.status = r.status;
    throw err;
  }
  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = new Error(`${r.status} ${await r.text()}`);
    err.status = r.status;
    throw err;
  }
  return r.json();
}

function app() {
  return {
    /* ---------------- đăng nhập / vai trò ---------------- */
    // "loading" (đang kiểm tra phiên) -> "login" | "chat" (role=user) |
    // "admin" (role=admin). Ba khối HTML trong index.html tương ứng đều
    // dùng <template x-if>, nên chỉ MỘT cây DOM tồn tại tại một thời điểm
    // - không lo control của view này lẫn với view khác.
    view: "loading",
    session: null, // {username, role} sau khi đăng nhập, null nếu chưa
    loginUsername: "",
    loginPassword: "",
    loginError: "",
    loginBusy: false,
    // Trang trí cho màn đăng nhập - chỉ mã + tên ngành THẬT (không phải
    // giá/số liệu bịa), lấy từ đúng bộ CORE_16 + VNZ đã ghi trong DSA v6.
    tickerStripItems: [
      { k: "VNM", t: "VNM", s: "Tiêu dùng thiết yếu" },
      { k: "SAB", t: "SAB", s: "Tiêu dùng thiết yếu" },
      { k: "MSN", t: "MSN", s: "Tiêu dùng thiết yếu" },
      { k: "VIC", t: "VIC", s: "Bất động sản" },
      { k: "VHM", t: "VHM", s: "Bất động sản" },
      { k: "HPG", t: "HPG", s: "Công nghiệp chế biến" },
      { k: "GAS", t: "GAS", s: "Công nghiệp chế biến" },
      { k: "MWG", t: "MWG", s: "Bán lẻ" },
      { k: "PNJ", t: "PNJ", s: "Bán lẻ" },
      { k: "VNZ", t: "VNZ", s: "Công nghệ thông tin" },
    ],

    /* ---------------- trạng thái Chat ---------------- */
    sessionId: "",
    theme: localStorage.getItem("fqa-theme") || "dark",
    uiLanguage: localStorage.getItem("fqa-language") || "en",
    windowDays: Number(localStorage.getItem("fqa-window") || 30),
    health: "unknown",
    _degradedList: [],     // service báo "degraded" (vd qa khi Ollama chết)
    _degradedDetail: "",   // nguyên văn detail của chúng

    turns: [],
    draft: "",
    strategy: "s3",
    busy: false,
    stageText: "",
    activeBranch: null,
    railState: "idle", // idle | classifying | working | locked

    tab: "rate",
    pending: null,          // câu trả lời đang chờ đánh giá
    rating: null,
    reasonCode: "",
    freeText: "",
    sending: false,

    queue: [],
    regressionCandidates: [],
    toast: "",

    /* ---------------- trạng thái Admin ---------------- */
    // "overview" | "coverage" | "models" | "tables" | "files" - trước xếp
    // chồng 5 panel trên 1 trang dài, giờ tách tab (2026-09-08) theo đúng
    // yêu cầu "chia nhiều mục ra nhiều tab để chuyển trang xem".
    adminTab: "overview",
    pipelineSummary: {},
    coverage: [],
    models: [],
    adminLoading: false,
    // 2026-09-08: kiểm kê bảng + trình duyệt file theo mã (yêu cầu "xem các
    // bảng đã có những gì, mã nào đã có file gì, kể cả nguồn TiniX").
    tables: [],
    files: [],
    filesLoading: false,
    fileTickerFilter: "",
    vietstockSummary: {},
    vietstockRuns: [],
    vietstockTickers: [],
    // 2026-09-21: tab "Biến động giá" - company_price_snapshot (EOD OHLCV,
    // 1 dòng/mã/ngày, nạp qua scripts/market_snapshot.py). KHÔNG phải dữ
    // liệu trong ngày/tick-by-tick - xem docstring /admin/prices ở
    // api/services/admin/main.py trước khi diễn giải biểu đồ này như giá
    // thời gian thực.
    priceTickers: [],
    priceTickerSelected: "",
    priceSeries: [],
    pricesLoading: false,
    priceWindowDays: 180,
    _priceChart: null,

    copy: {
      vi: {
        product: "Hỏi đáp tài chính",
        productSub: "Công bố thông tin doanh nghiệp niêm yết Việt Nam",
        language: "Ngôn ngữ",
        window: "Cửa sổ",
        day: "ngày",
        appearance: "Giao diện",
        dark: "Tối",
        light: "Sáng",
        logout: "Đăng xuất",
        router: "Định tuyến",
        conversation: "Hội thoại",
        newSession: "Phiên mới",
        waiting: "CHỜ CÂU HỎI",
        classifying: "ĐANG PHÂN LOẠI",
        answered: "Đã trả lời",
        page: "tr.",
        fallback: "dự phòng từ",
        abbreviationExpanded: "đã khai triển viết tắt",
        enterHint: "gửi",
        newlineHint: "xuống dòng",
        system: "Hệ thống",
        healthy: "Toàn bộ dịch vụ hoạt động",
        degraded: "Suy giảm",
        unavailable: "Mất kết nối",
        checking: "Đang kiểm tra…",
        branchHints: { dag: "số liệu BCTC", rag: "tài liệu công bố", whatif: "kịch bản giả định", causal: "quan hệ nhân quả", compare: "đối chiếu nhiều mã" },
        latency: "thời gian trả lời",
        sourceCount: "trích dẫn nguồn",
        placeholder: "Ví dụ: TTS của VNZ năm 2025?",
        emptyTitle: "Hỏi về số liệu hoặc tài liệu công bố",
        emptyText: "Bộ định tuyến sẽ chọn nhánh phù hợp và cho bạn thấy nó chọn gì, vì sao.",
        sessionLabel: "phiên", abbreviationExpandedTooltip: "Viết tắt đã được khai triển bằng từ điển",
        contextResolved: "đã hiểu từ ngữ cảnh",
        contextResolvedTooltip: "Câu hỏi nối tiếp đã được giải nghĩa thành câu độc lập, dựa trên các lượt hỏi trước (RQ3 - bộ nhớ hội thoại)",
        resolvedAsLabel: "Hiểu thành:",
        numericWarnTitle: "Cảnh báo: có số liệu KHÔNG khớp đoạn trích đã dẫn",
        numericMismatch: "Câu trả lời ghi {said}, nhưng số gần nhất trong nguồn là {src} (lệch {off}).",
        numericNoSource: "Câu trả lời ghi {said}, nhưng không có số nào cùng loại trong các đoạn trích để đối chiếu.",
        numericWarnFoot: "Kiểm tra tự động bằng cách so số trong câu trả lời với số thật trong nguồn. Hãy tin con số trong nguồn, không phải trong câu trả lời.",
        pctPoints: "điểm %",
        blockedFeedbackMsg: "Đánh giá câu trả lời ở bảng bên phải trước khi hỏi tiếp — hoặc bấm “Bỏ qua”.",
        sendQuestionAria: "Gửi câu hỏi", chooseSystemAria: "Chọn hệ thống demo",
        routingStatusAria: "Trạng thái định tuyến truy xuất",
        sync: "Đồng bộ Vietstock",
        syncNote: "Trạng thái quan sát từ crawler Premium chạy cục bộ trên máy chủ.",
        notRun: "Chưa có lần đồng bộ Premium nào.",

        // ----- màn hình đăng nhập (2026-09-22, trước đây hardcode tiếng Việt) -----
        loginUsernameLabel: "Tài khoản", loginPasswordLabel: "Mật khẩu",
        loginUsernamePlaceholder: "user hoặc admin", loginPasswordPlaceholder: "giống hệt tài khoản",
        loginButton: "Đăng nhập", loginChecking: "Đang kiểm tra…",
        loginDemoHint: "Tài khoản demo — mật khẩu trùng tên tài khoản",
        personaAnalyst: "PHÂN TÍCH", personaAnalystDesc: "analyst — số liệu, xu hướng, kịch bản",
        personaAuditor: "KIỂM TOÁN", personaAuditorDesc: "auditor — truy vết tài liệu, bằng chứng",
        personaExecutive: "LÃNH ĐẠO", personaExecutiveDesc: "executive — tóm tắt, kết luận nhanh",
        personaInvestor: "ĐẦU TƯ", personaInvestorDesc: "investor — so sánh, nhân quả, thị trường",
        personaAdmin: "ADMIN", personaAdminDesc: "admin — kiểm tra database",
        loginErrorNotConfigured: "Đăng nhập chưa được cấu hình (thiếu SESSION_SECRET trong api/.env).",
        loginErrorInvalid: "Sai tài khoản hoặc mật khẩu.",

        // ----- HITL: tầng 1 (đánh giá) -----
        tabRate: "Đánh giá", tabQueue: "Hàng đợi", tabRegression: "Regression",
        tier1Title: "Tầng 1 · Thu tín hiệu",
        noPendingAnswer: "Chưa có câu trả lời nào chờ đánh giá. Hỏi một câu để bắt đầu.",
        ratingQuestion: "Câu trả lời này hữu ích tới đâu?",
        ratingHint0: "1 = vô dụng · 5 = dùng được ngay",
        ratingHint1: "Sai hoàn toàn hoặc bịa.", ratingHint2: "Sai phần cốt lõi.",
        ratingHint3: "Đúng một phần, còn thiếu.", ratingHint4: "Đúng, còn gợn nhỏ.",
        ratingHint5: "Đúng và đủ để dùng ngay.",
        wrongWhereLabel: "Sai ở đâu?", reasonPlaceholder: "— chọn nguyên nhân gần nhất —",
        noteLabel: "Ghi chú (không bắt buộc)", notePlaceholder: "Số đúng phải là… / thiếu kỳ nào…",
        submitFeedback: "Gửi đánh giá", skipFeedback: "Bỏ qua",
        skipExplain: "“Bỏ qua” được lưu thành dữ liệu thật (is_skip), không phải bỏ trống. Không có nó thì mọi thống kê điểm số đều là mẫu tự chọn — chỉ nghe được tiếng người bận tâm nhất.",
        implicitSkipExplain: "Hỏi câu tiếp mà chưa chấm cũng được: lượt đó được ghi thành “bỏ qua ngầm” (reason_code='implicit'), tách riêng khỏi lượt chủ động bấm “Bỏ qua” — nên mẫu số vẫn đầy đủ mà không phải ép ai bấm.",
        feedbackDestinationTitle: "Đánh giá đi về đâu",
        bstage1Title: "Lớp ứng viên", bstage1Desc: "Mọi lượt trả lời và đánh giá ghi thêm, không sửa đè",
        bstage2Title: "Người phán quyết", bstage2Desc: "Chỉ mục vào hàng đợi mới cần người xem, kèm lý do",
        bstage3Title: "Regression candidate", bstage3Desc: "Expected answer đã duyệt trở thành test tái chạy được",
        boundaryExplain: "Điểm số không phải chân lý — người dùng cũng chấm sai được. Vì vậy điểm số chỉ dẫn tới việc xem lại; quyết định cuối cùng thuộc về người, có ghi lý do. Đây là ranh giới dữ liệu đã duyệt trong thiết kế Amershi et al. 2014.",

        // ----- HITL: tầng 2 (hàng đợi review) -----
        tier2Title: "Tầng 2 · Lấy mẫu theo độ bất định",
        tier2Explain: "Hệ thống tự đưa vào đây những lượt nó kém chắc chắn nhất: bị chấm điểm thấp, phải chuyển nhánh dự phòng, hoặc độ tương đồng truy xuất thấp. Công sức của người được dồn vào chỗ đáng ngờ nhất thay vì rải đều — Settles 2009.",
        queueEmpty: "Hàng đợi trống. Không có lượt nào đang cần người xem lại.",
        expectedAnswerPlaceholder: "Câu trả lời mong đợi, dựa trên bằng chứng…",
        sourcesNeededPlaceholder: "Nguồn/trang cần có, nếu biết",
        reviewNotePlaceholder: "Ghi chú review (không bắt buộc)",
        createRegressionBtn: "Tạo regression test", creatingRegression: "Đang tạo…",
        reviewedBtn: "Đã xem", dismissBtn: "Bỏ qua",

        // ----- regression candidates -----
        regressionApprovedTitle: "Regression candidates đã duyệt",
        regressionExplain: "Một rating không sửa được hệ thống. Mỗi mục ở đây có câu hỏi gốc, câu trả lời mong đợi và nguồn cần kiểm tra, nên có thể trở thành test tái chạy được. Chỉ mục được chấp nhận mới được export thành fixture; không có fine-tuning tự động trong demo này.",
        regressionEmpty: "Chưa có regression candidate. Review một lỗi và ghi expected answer để tạo test đầu tiên.",
        expectedLabel: "Expected:", acceptExportBtn: "Chấp nhận để export", rejectBtn: "Từ chối",

        // ----- toasts -----
        toastEnqueued: (r) => `Đã ghi nhận · đưa vào hàng đợi (${r})`,
        toastSkipRecorded: "Đã ghi nhận lượt bỏ qua", toastRatingRecorded: "Đã ghi nhận đánh giá",
        toastFeedbackFailed: "Không gửi được đánh giá — kiểm tra dịch vụ feedback.",
        toastRegressionCreated: "Đã tạo regression candidate từ review này",
        toastRegressionCreateFailed: "Không tạo được regression candidate",
        toastReviewUpdated: "Đã cập nhật mục review", toastReviewUpdateFailed: "Không cập nhật được mục review",
        toastRegressionAccepted: "Đã chấp nhận cho regression export", toastRegressionRejected: "Đã từ chối candidate",
        toastRegressionUpdateFailed: "Không cập nhật được regression candidate",
        toastRegressionLoadFailed: "Không nạp được danh sách regression candidate",
        toastRegressionAlreadyDecided: "Mục này đã được quyết định rồi (trạng thái: {status}) — không thể đổi lại.",
        toastExpectedRequired: "Cần viết câu trả lời kỳ vọng trước khi promote.",
        toastExpectedTooShort: "Câu trả lời kỳ vọng quá ngắn (tối thiểu 8 ký tự).",
        errUnknown: "lỗi không rõ",
        regressionDecidedNote: "Đã quyết định: {status}. Chỉ mục 'accepted' mới được export thành fixture.",
        provenanceWrittenBy: "Đáp án do:",
        provenanceDecidedBy: "duyệt bởi:",
        provenanceUnknown: "không rõ",
        reverseBtn: "Lật lại quyết định",
        reverseReasonLabel: "Lý do lật lại (bắt buộc)",
        reverseReasonPlaceholder: "vd: đáp án kỳ vọng không khớp câu hỏi",
        reverseConfirmBtn: "Xác nhận lật thành {status}",
        cancelBtn: "Huỷ",
        reverseExplain: "Lật lại được ghi thành một dòng MỚI trong nhật ký (không xoá dòng cũ), kèm tên bạn và lý do. Người đã ra quyết định cũ không thể tự lật — cần một tài khoản review khác.",
        toastRegressionReversed: "Đã lật lại thành {status}",
        toastReverseNeedsReason: "Phải ghi lý do trước khi lật lại quyết định.",
        toastNothingToReverse: "Mục này chưa được quyết định nên không có gì để lật.",
        toastAdminLoadFailed: "Không tải được dữ liệu admin - kiểm tra dịch vụ admin/DB.",
        toastFilesLoadFailed: "Không tải được danh sách file (pdf_documents).",
        toastPriceTickersFailed: "Không tải được danh sách mã có dữ liệu giá (company_price_snapshot).",
        toastPriceSeriesFailed: "Không tải được chuỗi giá cho mã này.",

        // ----- admin dashboard -----
        adminTitle: "Quản trị hệ thống", adminSubtitle: "Kiểm tra database & model registry — chỉ đọc",
        pipelineOverview: "Tổng quan pipeline", refresh: "Làm mới", refreshing: "Đang tải…",
        tabOverview: "Tổng quan", tabCoverage: "Coverage theo mã", tabModels: "Model registry",
        tabTables: "Bảng dữ liệu", tabFiles: "Tài liệu theo mã", tabPrices: "Biến động giá",
        coverageTitle: "Coverage theo mã", coverageCountSuffix: "mã có dữ liệu",
        colTicker: "Mã", colStructuredPeriods: "Kỳ số liệu", colYears: "Năm",
        colChunks: "Chunk RAG", colDocTypes: "Loại tài liệu", colModality: "Modality",
        coverageEmpty: "Chưa có dữ liệu coverage nào - kiểm tra kết nối DB hoặc ETL đã chạy chưa.",
        modelsTitle: "Model registry", modelsCountSuffix: "model",
        colTask: "Task", colModel: "Model", colVersion: "Version", colCalibration: "Hiệu chỉnh",
        colChampion: "Champion", colFrozenAt: "Đóng băng lúc", colValidationMetrics: "Chỉ số validation",
        modelsEmpty: "Chưa có model nào trong registry (model_registry_lite).",
        tablesTitle: "Kiểm kê bảng dữ liệu", tablesCountSuffix: "bảng",
        colTable: "Bảng", colRowCount: "Số dòng", tablesEmpty: "Chưa tải được danh sách bảng.",
        errorChip: "lỗi",
        filesTitle: "Tài liệu theo mã", filesCountSuffix: "file",
        filterPlaceholder: "Lọc theo mã, vd VNM", filterBtn: "Lọc",
        filesExplain: "Nguồn dữ liệu là bảng pdf_documents - một dòng cho MỘT file vật lý (khác disclosure_chunks, vốn ở mức chunk). Cột Nguồn là suy đoán tốt nhất dựa trên hình dạng tên file (2 mốc năm + \"kiemtoan\"/\"baocaotaichinh\" → khả năng cao là TiniX), không phải trường lưu sẵn trong DB - xem _guess_file_origin() trong api/services/admin/main.py.",
        colFile: "File", colDocType: "Loại", colYear: "Năm", colPages: "Số trang",
        colSource: "Nguồn", colActive: "Hoạt động", colOriginalPdf: "PDF gốc", colUpdated: "Cập nhật",
        activeYes: "còn", activeNo: "đã xoá", pdfMissing: "thiếu", pdfPresent: "có",
        filesEmpty: "Không có file nào khớp bộ lọc hiện tại.",
        pricesCountSuffix: "mã có dữ liệu giá",
        pricesExplain: "Nguồn: bảng company_price_snapshot, nạp 1 lần/ngày qua scripts/market_snapshot.py (vnstock, DAG thesis_market_snapshot) - đây là giá cuối ngày (open/high/low/close/volume), không phải dữ liệu tick-by-tick/thời gian thực. \"% thay đổi\" là so với phiên giao dịch liền trước có trong dữ liệu, không phải so với \"hôm nay\" theo lịch nếu ngày gần nhất chưa được nạp.",
        pricesEmptyTickers: "Chưa có mã nào có dữ liệu trong company_price_snapshot - kiểm tra Airflow DAG thesis_market_snapshot đã chạy chưa.",
        colDaysAvailable: "Số ngày có dữ liệu", colFromDate: "Từ ngày", colToDate: "Đến ngày (gần nhất)",
        pricesEmptySeries: "Không có dòng giá nào cho mã này trong cửa sổ đã chọn.",
        daySuffix: "ngày",
      },
      en: {
        product: "Financial Q&A",
        productSub: "Vietnamese listed-company disclosures",
        language: "Language",
        window: "Window",
        day: "days",
        appearance: "Appearance",
        dark: "Dark",
        light: "Light",
        logout: "Log out",
        router: "Routing",
        conversation: "Conversation",
        newSession: "New session",
        waiting: "AWAITING QUESTION",
        classifying: "CLASSIFYING",
        answered: "Answered",
        page: "p.",
        fallback: "fallback from",
        abbreviationExpanded: "abbreviation expanded",
        enterHint: "send",
        newlineHint: "new line",
        system: "System",
        healthy: "All services operational",
        degraded: "Degraded",
        unavailable: "Unavailable",
        checking: "Checking…",
        branchHints: { dag: "financial figures", rag: "disclosures", whatif: "scenario analysis", causal: "causal evidence", compare: "multi-company comparison" },
        latency: "answer time",
        sourceCount: "source citations",
        placeholder: "Example: Compare VNM revenue and net profit in 2024 with the prior year.",
        emptyTitle: "Ask about a financial figure or disclosure",
        emptyText: "The router selects a suitable branch and shows the decision and rationale.",
        sessionLabel: "session", abbreviationExpandedTooltip: "Abbreviation expanded using the dictionary",
        contextResolved: "resolved from context",
        contextResolvedTooltip: "This follow-up was rewritten into a standalone question using the previous turns (RQ3 - conversational memory)",
        resolvedAsLabel: "Understood as:",
        numericWarnTitle: "Warning: a figure does NOT match the cited excerpts",
        numericMismatch: "The answer states {said}, but the closest figure in the source is {src} (off by {off}).",
        numericNoSource: "The answer states {said}, but no comparable figure appears in the cited excerpts.",
        numericWarnFoot: "Automatic check comparing figures in the answer with figures actually in the source. Trust the source figure, not the answer.",
        pctPoints: "pp",
        blockedFeedbackMsg: "Rate the answer in the panel on the right before asking again — or click “Skip”.",
        sendQuestionAria: "Send question", chooseSystemAria: "Choose the demo system",
        routingStatusAria: "Retrieval routing status",
        sync: "Vietstock sync",
        syncNote: "Observed status from the local Premium crawler on this host.",
        notRun: "No Premium sync has run yet.",

        // ----- login screen -----
        loginUsernameLabel: "Username", loginPasswordLabel: "Password",
        loginUsernamePlaceholder: "user or admin", loginPasswordPlaceholder: "same as the username",
        loginButton: "Log in", loginChecking: "Checking…",
        loginDemoHint: "Demo accounts — password is the same as the username",
        personaAnalyst: "ANALYST", personaAnalystDesc: "analyst — figures, trends, scenarios",
        personaAuditor: "AUDITOR", personaAuditorDesc: "auditor — document trail, evidence",
        personaExecutive: "EXECUTIVE", personaExecutiveDesc: "executive — summaries, quick conclusions",
        personaInvestor: "INVESTOR", personaInvestorDesc: "investor — comparisons, causal evidence, market",
        personaAdmin: "ADMIN", personaAdminDesc: "admin — inspect the database",
        loginErrorNotConfigured: "Login is not configured yet (SESSION_SECRET missing in api/.env).",
        loginErrorInvalid: "Wrong username or password.",

        // ----- HITL: tier 1 (rate) -----
        tabRate: "Rate", tabQueue: "Queue", tabRegression: "Regression",
        tier1Title: "Tier 1 · Signal collection",
        noPendingAnswer: "No answer is waiting for a rating yet. Ask a question to start.",
        ratingQuestion: "How useful was this answer?",
        ratingHint0: "1 = useless · 5 = usable right away",
        ratingHint1: "Completely wrong or fabricated.", ratingHint2: "Wrong on the core point.",
        ratingHint3: "Partly right, something is missing.", ratingHint4: "Right, with a minor rough edge.",
        ratingHint5: "Right and good enough to use as-is.",
        wrongWhereLabel: "Where did it go wrong?", reasonPlaceholder: "— pick the closest reason —",
        noteLabel: "Note (optional)", notePlaceholder: "The correct figure should be… / which period is missing…",
        submitFeedback: "Submit rating", skipFeedback: "Skip",
        skipExplain: "“Skip” is saved as real data (is_skip), not left blank. Without it, every score statistic would be a self-selected sample — only hearing from the people who cared most.",
        implicitSkipExplain: "Asking the next question without rating is fine: that turn is recorded as an “implicit skip” (reason_code='implicit'), kept separate from an explicit “Skip” click — so the denominator stays complete without forcing anyone to click.",
        feedbackDestinationTitle: "Where a rating goes",
        bstage1Title: "Candidate layer", bstage1Desc: "Every answer and rating is appended, never overwritten",
        bstage2Title: "Human adjudicator", bstage2Desc: "Only items entered into the queue need a human look, with a reason",
        bstage3Title: "Regression candidate", bstage3Desc: "An approved expected answer becomes a re-runnable test",
        boundaryExplain: "A rating is not ground truth — users can rate incorrectly too. So a rating only triggers a review; the final call belongs to a human, with a reason recorded. This is the reviewed-data boundary from Amershi et al. 2014's design.",

        // ----- HITL: tier 2 (review queue) -----
        tier2Title: "Tier 2 · Uncertainty sampling",
        tier2Explain: "The system automatically routes in the turns it is least confident about: rated low, fell back to a secondary branch, or had low retrieval similarity. Human effort goes to the most suspect cases instead of being spread evenly — Settles 2009.",
        queueEmpty: "The queue is empty. No turn currently needs a human look.",
        expectedAnswerPlaceholder: "Expected answer, grounded in evidence…",
        sourcesNeededPlaceholder: "Source/page that should be cited, if known",
        reviewNotePlaceholder: "Review note (optional)",
        createRegressionBtn: "Create regression test", creatingRegression: "Creating…",
        reviewedBtn: "Reviewed", dismissBtn: "Dismiss",

        // ----- regression candidates -----
        regressionApprovedTitle: "Approved regression candidates",
        regressionExplain: "A rating alone cannot fix the system. Each item here has the original question, an expected answer, and the source to check, so it can become a re-runnable test. Only accepted items are exported as fixtures; there is no automatic fine-tuning in this demo.",
        regressionEmpty: "No regression candidate yet. Review a failure and write an expected answer to create the first one.",
        expectedLabel: "Expected:", acceptExportBtn: "Accept for export", rejectBtn: "Reject",

        // ----- toasts -----
        toastEnqueued: (r) => `Recorded · queued for review (${r})`,
        toastSkipRecorded: "Skip recorded", toastRatingRecorded: "Rating recorded",
        toastFeedbackFailed: "Could not submit the rating — check the feedback service.",
        toastRegressionCreated: "Regression candidate created from this review",
        toastRegressionCreateFailed: "Could not create the regression candidate",
        toastReviewUpdated: "Review item updated", toastReviewUpdateFailed: "Could not update the review item",
        toastRegressionAccepted: "Accepted for regression export", toastRegressionRejected: "Candidate rejected",
        toastRegressionUpdateFailed: "Could not update the regression candidate",
        toastRegressionLoadFailed: "Could not load the regression candidate list",
        toastRegressionAlreadyDecided: "This item has already been decided (status: {status}) — it cannot be changed.",
        toastExpectedRequired: "Write the expected answer before promoting.",
        toastExpectedTooShort: "The expected answer is too short (8 characters minimum).",
        errUnknown: "unknown error",
        regressionDecidedNote: "Decided: {status}. Only 'accepted' items are exported as fixtures.",
        provenanceWrittenBy: "Expected answer by:",
        provenanceDecidedBy: "decided by:",
        provenanceUnknown: "unknown",
        reverseBtn: "Reverse this decision",
        reverseReasonLabel: "Reason for reversing (required)",
        reverseReasonPlaceholder: "e.g. expected answer does not match the question",
        reverseConfirmBtn: "Confirm reversal to {status}",
        cancelBtn: "Cancel",
        reverseExplain: "A reversal is written as a NEW row in the decision log (the old row is never deleted), with your name and reason. Whoever made the previous decision cannot reverse it themselves — a different reviewer account is required.",
        toastRegressionReversed: "Reversed to {status}",
        toastReverseNeedsReason: "Write a reason before reversing the decision.",
        toastNothingToReverse: "This item has not been decided yet, so there is nothing to reverse.",
        toastAdminLoadFailed: "Could not load admin data - check the admin service/DB.",
        toastFilesLoadFailed: "Could not load the file list (pdf_documents).",
        toastPriceTickersFailed: "Could not load the list of tickers with price data (company_price_snapshot).",
        toastPriceSeriesFailed: "Could not load the price series for this ticker.",

        // ----- admin dashboard -----
        adminTitle: "System administration", adminSubtitle: "Inspect the database & model registry — read-only",
        pipelineOverview: "Pipeline overview", refresh: "Refresh", refreshing: "Loading…",
        tabOverview: "Overview", tabCoverage: "Coverage by ticker", tabModels: "Model registry",
        tabTables: "Data tables", tabFiles: "Files by ticker", tabPrices: "Price movement",
        coverageTitle: "Coverage by ticker", coverageCountSuffix: "tickers with data",
        colTicker: "Ticker", colStructuredPeriods: "Structured periods", colYears: "Years",
        colChunks: "RAG chunks", colDocTypes: "Doc types", colModality: "Modality",
        coverageEmpty: "No coverage data yet - check the DB connection or whether the ETL has run.",
        modelsTitle: "Model registry", modelsCountSuffix: "models",
        colTask: "Task", colModel: "Model", colVersion: "Version", colCalibration: "Calibration",
        colChampion: "Champion", colFrozenAt: "Frozen at", colValidationMetrics: "Validation metrics",
        modelsEmpty: "No model in the registry yet (model_registry_lite).",
        tablesTitle: "Data table inventory", tablesCountSuffix: "tables",
        colTable: "Table", colRowCount: "Row count", tablesEmpty: "Could not load the table list.",
        errorChip: "error",
        filesTitle: "Files by ticker", filesCountSuffix: "files",
        filterPlaceholder: "Filter by ticker, e.g. VNM", filterBtn: "Filter",
        filesExplain: "The source is the pdf_documents table - one row per physical file (different grain from disclosure_chunks, which is chunk-level). The Source column is a best-effort guess based on the filename shape (two year stamps + \"kiemtoan\"/\"baocaotaichinh\" → likely TiniX), not a field actually stored in the DB - see _guess_file_origin() in api/services/admin/main.py.",
        colFile: "File", colDocType: "Type", colYear: "Year", colPages: "Pages",
        colSource: "Source", colActive: "Active", colOriginalPdf: "Original PDF", colUpdated: "Updated",
        activeYes: "active", activeNo: "removed", pdfMissing: "missing", pdfPresent: "present",
        filesEmpty: "No file matches the current filter.",
        pricesCountSuffix: "tickers with price data",
        pricesExplain: "Source: the company_price_snapshot table, loaded once/day via scripts/market_snapshot.py (vnstock, thesis_market_snapshot DAG) - this is end-of-day data (open/high/low/close/volume), not tick-by-tick/real-time data. \"% change\" is versus the previous trading session present in the data, not versus the calendar \"today\" if the latest day hasn't been loaded yet.",
        pricesEmptyTickers: "No ticker has data in company_price_snapshot yet - check whether the thesis_market_snapshot Airflow DAG has run.",
        colDaysAvailable: "Days with data", colFromDate: "From", colToDate: "To (latest)",
        pricesEmptySeries: "No price row for this ticker in the selected window.",
        daySuffix: "days",
      },
    },

    branches: [
      { id: "dag",     hint: "số liệu BCTC" },
      { id: "rag",     hint: "tài liệu công bố" },
      { id: "whatif",  hint: "kịch bản giả định" },
      { id: "causal",  hint: "quan hệ nhân quả" },
      { id: "compare", hint: "đối chiếu nhiều mã" },
    ],

    _reasonsVi: [
      { v: "wrong_number",   t: "Sai số liệu" },
      { v: "wrong_entity",   t: "Sai công ty / mã" },
      { v: "wrong_period",   t: "Sai kỳ báo cáo" },
      { v: "wrong_branch",   t: "Chọn sai nhánh xử lý" },
      { v: "missing_context",t: "Thiếu bối cảnh cần thiết" },
      { v: "hallucination",  t: "Bịa thông tin không có trong nguồn" },
      { v: "too_slow",       t: "Quá chậm" },
      { v: "unclear",        t: "Khó hiểu" },
      { v: "other",          t: "Khác" },
    ],
    _reasonsEn: [
      { v: "wrong_number",   t: "Wrong figure" },
      { v: "wrong_entity",   t: "Wrong company / ticker" },
      { v: "wrong_period",   t: "Wrong reporting period" },
      { v: "wrong_branch",   t: "Wrong branch selected" },
      { v: "missing_context",t: "Missing necessary context" },
      { v: "hallucination",  t: "Fabricated info not in the source" },
      { v: "too_slow",       t: "Too slow" },
      { v: "unclear",        t: "Unclear" },
      { v: "other",          t: "Other" },
    ],

    _samplesVi: [
      { b: "dag",     q: "TTS của VNZ năm 2025?" },
      { b: "rag",     q: "Công ty kiểm toán của VNZ năm 2025 là ai?" },
      { b: "causal",  q: "Đòn bẩy tài chính có tác động đến biên lợi nhuận không?" },
      { b: "compare", q: "So sánh HPG và GAS" },
    ],
    _samplesEn: [
      { b: "dag",     q: "What is VNZ's total assets in 2025?" },
      // 2026-09-22: was labelled "rag" because the EN phrasing "who audited"
      // used to miss rules.py's auditor gate (which only matched the noun
      // "auditor") and fell through to the RAG catch-all. That gate is fixed,
      // so this question now routes to auditor_lookup exactly like its
      // Vietnamese equivalent does.
      { b: "auditor_lookup", q: "Who audited VNZ in 2025?" },
      { b: "causal",  q: "Does financial leverage affect net margin?" },
      { b: "compare", q: "Compare HPG and GAS" },
    ],

    // 2026-09-22: giữ lại getter (nhiều chỗ vẫn đọc) nhưng nó KHÔNG còn
    // khoá ô nhập nữa - xem send()'s implicit-skip ở dưới. Trước đây mọi
    // binding :disabled đều gắn vào đây, nên sau MỖI câu trả lời người dùng
    // buộc phải bấm sang panel bên phải mới hỏi tiếp được.
    get awaitingFeedback() {
      return !!this.pending;
    },

    // Panel đánh giá gập lại mặc định (2026-09-22). Chỉ hiện đúng một dòng
    // sao nhỏ; bấm vào mới mở đủ lý do + ghi chú. Đặt ở state chứ không
    // phải CSS-only để `send()` còn tự gập lại sau khi đã ghi nhận.
    rateExpanded: false,

    /* ---------------- khởi động ---------------- */
    async boot() {
      // /api/auth/me quyết định view NGAY từ đầu - trang chỉ vẽ ra đúng
      // MỘT trong ba giao diện (login/chat/admin), không bao giờ vẽ chat
      // rồi mới ẩn đi nếu hoá ra chưa đăng nhập (tránh nháy nội dung).
      try {
        this.session = await getJSON(`${API.auth}/me`);
        await this._enterView(this.session.role);
      } catch {
        this.view = "login";
      }

      this.checkHealth();
      setInterval(() => this.checkHealth(), 30000);
      this.$nextTick(() => window.lucide?.createIcons());
      this.$watch("tab", () => this.$nextTick(() => window.lucide?.createIcons()));
    },

    tr(key) {
      return this.copy[this.uiLanguage]?.[key] || this.copy.en[key] || key;
    },

    setLanguage(language) {
      this.uiLanguage = language === "vi" ? "vi" : "en";
      localStorage.setItem("fqa-language", this.uiLanguage);
    },

    latencyLabel(ms) {
      return `${Number(ms || 0).toLocaleString(this.uiLanguage === "vi" ? "vi-VN" : "en-US")} ms ${this.tr("latency")}`;
    },

    windowLabel(days) {
      return this.uiLanguage === "vi" ? `${days} ${this.tr("day")}` : `${days} ${this.tr("day")}`;
    },

    branchHint(branch) {
      return this.copy[this.uiLanguage]?.branchHints?.[branch] || branch;
    },

    reasons() {
      return this.uiLanguage === "vi" ? this._reasonsVi : this._reasonsEn;
    },

    samples() {
      return this.uiLanguage === "vi" ? this._samplesVi : this._samplesEn;
    },

    /* ---------------- đăng nhập / đăng xuất ---------------- */
    async _enterView(role) {
      if (role === "admin") {
        this.view = "admin";
        await this.loadAdmin();
      } else {
        this.view = "chat";
        await this.initChatSession();
      }
    },

    async initChatSession() {
      this.sessionId =
        sessionStorage.getItem("fqa-session") ||
        (await getJSON(`${API.qa}/session/new`).then((d) => d.session_id).catch(() =>
          Math.random().toString(36).slice(2)
        ));
      sessionStorage.setItem("fqa-session", this.sessionId);
    },

    async login() {
      if (this.loginBusy) return;
      this.loginBusy = true;
      this.loginError = "";
      try {
        this.session = await postJSON(`${API.auth}/login`, {
          username: this.loginUsername.trim(),
          password: this.loginPassword,
        });
        this.loginPassword = "";
        await this._enterView(this.session.role);
      } catch (err) {
        this.loginError =
          err.status === 503
            ? this.tr("loginErrorNotConfigured")
            : this.tr("loginErrorInvalid");
      } finally {
        this.loginBusy = false;
      }
    },

    async logout() {
      try {
        await postJSON(`${API.auth}/logout`, {});
      } catch {
        // cookie có thể đã hết hạn sẵn - vẫn dọn trạng thái phía client
      }
      this.session = null;
      this.turns = [];
      this.pending = null;
      this.loginUsername = "";
      this.loginPassword = "";
      sessionStorage.removeItem("fqa-session");
      this.view = "login";
    },

    async checkHealth() {
      try {
        const rows = await getJSON("/health");
        const down = rows.filter((r) => r.status === "down").map((r) => r.service);
        // 2026-09-24: "degraded" trước đây bị BỎ QUA hoàn toàn ở đây - chỉ
        // "down" mới được đếm. Nên khi Ollama chết, qa service báo degraded
        // mà giao diện vẫn hiện "Bình thường", và cách duy nhất phát hiện là
        // hỏi một câu RAG rồi đợi tới 56 giây mới thấy thông báo fallback.
        const degraded = rows.filter((r) => r.status === "degraded");
        this.health = down.length ? "down" : (degraded.length ? "degraded" : "ok");
        this._downList = down;
        this._degradedList = degraded.map((r) => r.service);
        // Giữ nguyên văn detail của service - đó là chỗ nói rõ CÁI GÌ hỏng
        // ("Ollama không phản hồi") và cái gì vẫn chạy.
        this._degradedDetail = degraded.map((r) => r.detail).filter(Boolean).join(" · ");
      } catch {
        this.health = "down";
        this._downList = ["gateway"];
        this._degradedList = [];
        this._degradedDetail = "";
      }
    },

    healthLabel() {
      if (this.health === "ok") return this.tr("healthy");
      if (this.health === "down") return `${this.tr("unavailable")}: ${(this._downList || []).join(", ")}`;
      if (this.health === "degraded") return `${this.tr("degraded")}: ${(this._degradedList || []).join(", ")}`;
      return this.tr("checking");
    },

    // Dòng chi tiết hiện dưới chỉ báo sức khoẻ khi có service degraded -
    // để biết NGAY là Ollama chết, trước khi hỏi câu đầu tiên.
    healthDetail() {
      return this.health === "degraded" ? (this._degradedDetail || "") : "";
    },

    /* ---------------- cài đặt ---------------- */
    setTheme(t) {
      this.theme = t;
      localStorage.setItem("fqa-theme", this.theme);
      this.$nextTick(() => window.lucide?.createIcons());
    },

    setWindow(w) {
      this.windowDays = w;
      localStorage.setItem("fqa-window", String(w));
    },

    /* ---------------- hỏi đáp ---------------- */
    // Nhãn persona hiển thị - giữ đồng bộ THỦ CÔNG với src/core/personas.py
    // (frontend không import được Python). Nếu thêm persona mới, sửa cả 2
    // chỗ + ACCOUNTS trong api/services/gateway/auth.py.
    personaLabel() {
      const map = this.uiLanguage === "vi"
        ? { analyst: "PHÂN TÍCH", auditor: "KIỂM TOÁN", executive: "LÃNH ĐẠO", investor: "ĐẦU TƯ" }
        : { analyst: "ANALYST", auditor: "AUDITOR", executive: "EXECUTIVE", investor: "INVESTOR" };
      return map[this.session?.persona] || "";
    },

    baseBranch(b) {
      const base = (b || "rag").replace("_unsupported", "").split(" ")[0];
      // auditor_lookup, dag_trend (2026-08-07), price_snapshot (2026-09-21):
      // đều là biến thể của DAG về mặt Ý NGHĨA (số liệu có cấu trúc, chính
      // xác tuyệt đối, không qua RAG/LLM) - dùng chung màu/trạm định tuyến
      // với "dag" thay vì cần thêm màu + trạm rail riêng cho từng nhánh phụ
      // hiếm khi kích hoạt.
      return ["auditor_lookup", "dag_trend", "price_snapshot"].includes(base) ? "dag" : base;
    },

    activeColor() {
      const b = this.baseBranch(this.activeBranch || "rag");
      return `var(--br-${b})`;
    },

    railHeadline() {
      if (this.railState === "idle") return this.tr("waiting");
      if (this.railState === "classifying") return this.tr("classifying");
      if (this.railState === "working") return (this.activeBranch || "").toUpperCase();
      return (this.activeBranch || "").toUpperCase();
    },

    railDetail() {
      if (this.railState === "idle") return this.uiLanguage === "vi"
        ? "Bộ phân loại chọn nhánh trước khi truy xuất."
        : "The classifier selects a branch before retrieval.";
      const b = this.branches.find((x) => x.id === this.baseBranch(this.activeBranch));
      if (this.railState === "working") return `${this.uiLanguage === "vi" ? "Đang chạy" : "Running"} · ${this.branchHint(b?.id)}`;
      if (this.railState === "locked") return `${this.tr("answered")} · ${this.branchHint(b?.id)}`;
      return "";
    },

    autogrow(el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 128) + "px";
    },

    scrollDown() {
      this.$nextTick(() => {
        const el = this.$refs.stream;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    /* Ghi một lượt BỎ QUA NGẦM cho câu trả lời trước đó.
     *
     * 2026-09-22 - thay cho cái gate cũ khoá ô nhập sau mỗi câu trả lời.
     * Lý lẽ phương pháp KHÔNG đổi và đây chính là chỗ giữ nó: tài liệu của
     * feedback service ghi rõ "Bỏ qua (skip) CŨNG là dữ liệu: nó cho biết
     * mẫu số của tỷ lệ phản hồi" - nếu chỉ bỏ gate mà không ghi gì thì mọi
     * thống kê rating lại thành mẫu tự chọn, đúng thứ cái gate sinh ra để
     * tránh. Nên: người dùng hỏi tiếp mà chưa chấm điểm -> vẫn ghi đủ một
     * bản ghi is_skip, chỉ khác là không bắt ai phải bấm.
     *
     * reason_code='implicit' tách "chủ động bấm Bỏ qua" khỏi "lướt qua
     * luôn" - hai tín hiệu KHÁC nhau mà cái gate cũ gộp làm một (một cú bấm
     * Bỏ qua bị ép buộc không nói lên điều gì về câu trả lời). Về mặt phân
     * tích đây là dữ liệu chặt hơn bản cũ, không phải lỏng hơn.
     *
     * Fail-soft: lỗi mạng ở đây tuyệt đối không được chặn câu hỏi tiếp theo.
     */
    async _recordImplicitSkip() {
      const p = this.pending;
      if (!p) return;
      this.pending = null;          // xoá NGAY để không ghi trùng
      this.rateExpanded = false;
      try {
        await postJSON(`${API.feedback}/feedback`, {
          answer_run_id: p.answer_run_id,
          rating: null,
          is_skip: true,
          reason_code: "implicit",
          free_text: null,
        });
      } catch {
        // im lặng: mất một tín hiệu bỏ qua ngầm không đáng để chặn người dùng
      }
    },

    async send() {
      const q = this.draft.trim();
      if (!q || this.busy) return;
      // Câu trả lời trước chưa được chấm -> ghi nhận bỏ qua ngầm rồi đi tiếp.
      // KHÔNG await: request này không được phép làm chậm câu hỏi mới.
      if (this.pending) this._recordImplicitSkip();

      // Keep the query string bounded for SSE. The resolver treats this as
      // optional conversational context and falls back to the current text
      // when a rewrite would remove an explicit ticker or year.
      const history = this.turns
        .filter((turn) => turn.role === "user")
        .slice(-2)
        .map((turn) => turn.text.slice(-1200));
      this.turns.push({ role: "user", text: q });
      this.draft = "";
      if (this.$refs.input) this.$refs.input.style.height = "auto";
      this.busy = true;
      this.railState = "classifying";
      this.activeBranch = null;
      this.stageText = this.strategy === "s3"
        ? (this.uiLanguage === "vi" ? "Đang phân loại câu hỏi…" : "Classifying question…")
        : `${this.uiLanguage === "vi" ? "Đang chạy" : "Running"} ${this.strategy.toUpperCase()}…`;
      this.scrollDown();

      const params = new URLSearchParams({
        question: q,
        session_id: this.sessionId,
        strategy: this.strategy,
        // 2026-09-22: chỉ là TIE-BREAKER cho ngôn ngữ câu trả lời, không
        // phải lệnh ép - backend (lang_detect.resolve_answer_language) ưu
        // tiên ngôn ngữ của CHÍNH câu hỏi, chỉ dùng giá trị này khi câu hỏi
        // là mẩu cụt không đoán được ("VNZ?").
        ui_lang: this.uiLanguage,
      });
      history.forEach((turn) => params.append("history", turn));
      const url = `${API.qa}/ask/stream?${params.toString()}`;
      const es = new EventSource(url);

      es.addEventListener("expand", (e) => {
        const d = JSON.parse(e.data);
        this.stageText = `${this.uiLanguage === "vi" ? "Đã khai triển viết tắt" : "Expanded abbreviation"} → “${d.expanded}”`;
      });

      es.addEventListener("classify", (e) => {
        const d = JSON.parse(e.data);
        this.activeBranch = this.baseBranch(d.branch);
        this.railState = "working";
        const b = this.branches.find((x) => x.id === this.activeBranch);
        this.stageText = `${this.uiLanguage === "vi" ? "Nhánh" : "Branch"} ${this.activeBranch.toUpperCase()} · ${this.branchHint(b?.id)}`;
      });

      es.addEventListener("working", (e) => {
        const d = JSON.parse(e.data);
        this.stageText = d.note || this.stageText;
      });

      es.addEventListener("answer", (e) => {
        const d = JSON.parse(e.data);
        this.turns.push({ role: "bot", ...d });
        this.activeBranch = this.baseBranch(d.branch);
        this.railState = "locked";
        this.busy = false;
        es.close();

        if (d.answer_run_id) {
          this.pending = {
            answer_run_id: d.answer_run_id,
            question: d.question,
            answer: d.answer,
          };
          this.rating = null;
          this.reasonCode = "";
          this.freeText = "";
          this.rateExpanded = false;   // luôn mở lại ở dạng gập - xem rateExpanded
          this.tab = "rate";
        }
        this.scrollDown();
        this.$nextTick(() => window.lucide?.createIcons());
      });

      es.addEventListener("error", (e) => {
        let detail = this.uiLanguage === "vi"
          ? "Mất kết nối tới dịch vụ hỏi đáp."
          : "Lost connection to the Q&A service.";
        try {
          if (e.data) detail = JSON.parse(e.data).detail;
        } catch {}
        this.turns.push({
          role: "bot", branch: "rag", answer: detail,
          intent_reason: "", sources: [], latency_ms: 0,
        });
        this.busy = false;
        this.railState = "idle";
        es.close();
        this.scrollDown();
      });

      es.onerror = () => {
        if (this.busy) {
          this.busy = false;
          this.railState = "idle";
          this.stageText = "";
          es.close();
        }
      };
    },

    resetSession() {
      this.turns = [];
      this.pending = null;
      this.activeBranch = null;
      this.railState = "idle";
      sessionStorage.removeItem("fqa-session");
      this.initChatSession();
    },

    /* ---- tô màu câu trả lời theo quy ước bảng giá ---- */
    renderAnswer(text) {
      const esc = (text || "").replace(/[&<>]/g, (c) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])
      );
      return esc
        // % tăng/giảm: xanh lá / đỏ theo đúng quy ước bảng giá VN
        .replace(/([+\-]\d+[.,]?\d*%)/g, (m) =>
          m.startsWith("+")
            ? `<span class="delta-up">${m}</span>`
            : `<span class="delta-down">${m}</span>`
        )
        // số tiền lớn: chữ mono để các hàng xếp thẳng cột
        .replace(/(\d{1,3}(?:,\d{3}){2,})/g, '<span class="num">$1</span>')
        // mã cổ phiếu 3-4 chữ hoa
        .replace(/\b([A-Z]{3,4})\b(?=[\s,.:)]|$)/g, '<span class="tickr">$1</span>');
    },

    retrievalLabel(r) {
      const labels = this.uiLanguage === "vi"
        ? { both: "cả hai", lexical: "khớp chữ", vector: "ngữ nghĩa" }
        : { both: "both", lexical: "lexical", vector: "semantic" };
      return labels[r] || r;
    },

    /* ---------------- HITL: tầng 1 ---------------- */
    starColor(n) {
      // Thang màu đọc được ngay: thấp = đỏ (giảm), cao = xanh (tăng) -
      // cùng quy ước với bảng giá, không cần chú giải.
      return ["var(--px-down)", "var(--px-down)", "var(--px-ref)",
              "var(--px-up)", "var(--px-up)"][n - 1];
    },

    ratingHint() {
      if (!this.rating) return this.tr("ratingHint0");
      return this.tr(`ratingHint${this.rating}`);
    },

    async submitFeedback(isSkip) {
      if (!this.pending || this.sending) return;
      this.sending = true;
      try {
        const res = await postJSON(`${API.feedback}/feedback`, {
          answer_run_id: this.pending.answer_run_id,
          rating: isSkip ? null : this.rating,
          is_skip: !!isSkip,
          reason_code: isSkip ? null : this.reasonCode || null,
          free_text: isSkip ? null : this.freeText || null,
        });
        this.pending = null;
        this.rating = null;
        this.reasonCode = "";
        this.freeText = "";
        this.rateExpanded = false;
        this.showToast(
          res.enqueued_for_review
            ? this.copy[this.uiLanguage].toastEnqueued(res.enqueue_reason)
            : isSkip
            ? this.tr("toastSkipRecorded")
            : this.tr("toastRatingRecorded")
        );
        if (res.enqueued_for_review) this.loadQueue();
      } catch (err) {
        this.showToast(this.tr("toastFeedbackFailed"));
      } finally {
        this.sending = false;
      }
    },

    /* ---------------- HITL: tầng 2 ---------------- */
    async loadQueue() {
      try {
        this.queue = (await getJSON(`${API.feedback}/review/queue?limit=20`)).map((i) => ({
          ...i, _note: "", _expected: "", _sources: "", _promoting: false,
        }));
        this.$nextTick(() => window.lucide?.createIcons());
      } catch {
        this.queue = [];
      }
    },

    async promoteRegression(item) {
      if (item._promoting) return;
      // Cùng lý do như decideRegression: guard im lặng = "bấm không ăn".
      // expected_answer có min_length=8 ở Pydantic (RegressionCandidateRequest),
      // nên nói luôn ngưỡng đó thay vì để người dùng đoán vì sao nút chết.
      if (!item._expected.trim()) {
        this.showToast(this.tr("toastExpectedRequired"));
        return;
      }
      if (item._expected.trim().length < 8) {
        this.showToast(this.tr("toastExpectedTooShort"));
        return;
      }
      item._promoting = true;
      try {
        await postJSON(`${API.feedback}/review/promote`, {
          review_id: item.review_id,
          expected_answer: item._expected.trim(),
          expected_sources: item._sources.trim() || null,
          reviewer_note: item._note.trim() || null,
        });
        this.queue = this.queue.filter((x) => x.review_id !== item.review_id);
        this.showToast(this.tr("toastRegressionCreated"));
        this.loadRegressionCandidates();
      } catch (err) {
        console.error("promoteRegression failed:", err);
        this.showToast(`${this.tr("toastRegressionCreateFailed")} — ${this._errText(err)}`);
      } finally {
        item._promoting = false;
      }
    },

    async resolveItem(item, status) {
      try {
        await postJSON(`${API.feedback}/review/action`, {
          review_id: item.review_id,
          status,
          resolution_note: item._note || null,
        });
        this.queue = this.queue.filter((x) => x.review_id !== item.review_id);
        this.showToast(this.tr("toastReviewUpdated"));
      } catch {
        this.showToast(this.tr("toastReviewUpdateFailed"));
      }
    },

    /* ---------------- regression candidates ---------------- */
    /* Rút câu giải thích THẬT ra khỏi một lỗi fetch.
     * getJSON/postJSON ném Error("<status> <body>") kèm .status - toàn bộ
     * thông tin cần thiết đã có sẵn, các catch cũ chỉ đơn giản là vứt đi. */
    _errText(err) {
      const raw = String(err?.message || err || "");
      try {
        // Body của FastAPI là {"detail": "..."} - lấy đúng phần detail.
        const body = raw.slice(raw.indexOf(" ") + 1);
        const parsed = JSON.parse(body);
        if (parsed?.detail) return `${err.status || ""} ${parsed.detail}`.trim();
      } catch {}
      return raw.slice(0, 220) || this.tr("errUnknown");
    },

    /* Cảnh báo số liệu không khớp nguồn (2026-09-25).
     *
     * rag_pipeline.verify_numeric_claims() so từng con số trong câu trả lời
     * với các con số thật trong đoạn trích đã dẫn. Trước đây kết quả đó
     * không bao giờ tới được đây - nên một câu trả lời ghi 1,597,803,538,910
     * trong khi nguồn ghi 159,780,353,891 (lệch 10 lần) hiện ra như một câu
     * trả lời bình thường. Giờ mỗi con số lệch hiện kèm con số gần nhất
     * trong nguồn, để người đọc tự kiểm được, thay vì phải tin.
     *
     * Chỉ hiện khi có số LỆCH. "verified" và "no_numeric_claim" thì im lặng:
     * một nhãn xanh trên mọi câu trả lời sẽ thành nhiễu, và ngưỡng khớp 2%
     * trong verify_numeric_claims() vẫn là giả định mặc định chưa được xác
     * nhận (xem docstring của hàm đó) - nên nó đủ tin để CẢNH BÁO, chưa đủ
     * tin để CHỨNG NHẬN. */
    numericWarnings(t) {
      const nv = t && t.signals && t.signals.numeric_verification;
      if (!nv || nv.label !== "unverified_claim_mismatch") return [];
      const loc = this.uiLanguage === "vi" ? "vi-VN" : "en-US";
      const fmt = (v, pct) => (v === null || v === undefined)
        ? null
        : Number(v).toLocaleString(loc, { maximumFractionDigits: 2 }) + (pct ? "%" : "");
      return (nv.claims || []).filter((c) => !c.matched).map((c) => {
        const said = fmt(c.value, c.is_percent);
        const src = fmt(c.closest_source_value, c.is_percent);
        if (src === null) {
          return this.tr("numericNoSource").replace("{said}", said);
        }
        const off = c.is_percent
          ? (c.diff_points !== null ? `${c.diff_points} ${this.tr("pctPoints")}` : "")
          : (c.diff_pct !== null ? `${Number(c.diff_pct).toLocaleString(loc, { maximumFractionDigits: 1 })}%` : "");
        return this.tr("numericMismatch")
          .replace("{said}", said).replace("{src}", src).replace("{off}", off);
      });
    },

    /* Một dòng "ai viết / ai duyệt" cho panel Regression (2026-09-22).
     * Dòng 'accepted' sẽ thành ground truth trong fixture hồi quy, nên
     * người sắp bấm Accept phải thấy đáp án này do ai soạn - kể cả (nhất
     * là) khi người đó chính là mình. NULL hiện thành "không rõ" chứ không
     * giấu đi: một dòng không truy được nguồn là một dòng đáng nghi hơn,
     * không phải ít hơn. */
    provenanceLine(item) {
      const unknown = this.tr("provenanceUnknown");
      const who = (name, persona) =>
        name ? (persona ? `${name} (${persona})` : name) : unknown;
      let line = `${this.tr("provenanceWrittenBy")} ${who(item.promoted_by, item.promoted_persona)}`;
      if (item.status !== "candidate") {
        line += ` · ${this.tr("provenanceDecidedBy")} ${who(item.decided_by, item.decided_persona)}`;
      }
      return line;
    },

    async loadRegressionCandidates() {
      try {
        this.regressionCandidates = await getJSON(`${API.feedback}/review/regression?limit=20`);
      } catch (err) {
        // 2026-09-22: trước đây lỗi nạp danh sách bị nuốt và danh sách rỗng
        // đi, trông y hệt "chưa có candidate nào" - hai tình huống hoàn toàn
        // khác nhau mà giao diện hiển thị giống hệt.
        console.error("loadRegressionCandidates failed:", err);
        this.regressionCandidates = [];
        this.showToast(`${this.tr("toastRegressionLoadFailed")} — ${this._errText(err)}`);
      }
    },

    async decideRegression(item, status) {
      // 2026-09-22 - CHẨN ĐOÁN ĐƯỢC. Bản cũ có hai đường im lặng tuyệt đối,
      // và đúng cả hai đều biểu hiện y hệt nhau với người dùng: "bấm nút mà
      // không có gì xảy ra".
      //   1) guard dưới đây `return` trắng - status không phải 'candidate'
      //      (đã accept/reject trước đó, hoặc list đang cũ so với DB) thì cú
      //      bấm rơi vào hư không, không một dòng phản hồi.
      //   2) `catch {}` nuốt SẠCH lỗi - 409/502/mất phiên đăng nhập đều hiện
      //      đúng một câu "cập nhật thất bại" chung chung, vứt luôn phần
      //      detail mà chính backend đã cất công soạn (vd "Regression
      //      candidate đã được quyết định (accepted)").
      // Giờ cả hai đều nói rõ chuyện gì xảy ra. Quan trọng với một panel mà
      // mỗi hành động là một quyết định của CON NGƯỜI được ghi vào dấu vết
      // đánh giá - im lặng ở đây là mất chính thứ "giải trình được" mà tầng
      // regression sinh ra để bảo vệ.
      if (item._deciding) return;
      if (item.status !== "candidate") {
        this.showToast(this.tr("toastRegressionAlreadyDecided").replace("{status}", item.status));
        return;
      }
      await this._postDecision(item, status, { note: item.reviewer_note || null });
    },

    /* Lật lại một quyết định đã có (2026-09-22).
     *
     * Chỉ hiện khi item đã rời trạng thái 'candidate'. Luôn lật sang trạng
     * thái NGƯỢC LẠI - "lật lại" chỉ có đúng một nghĩa, không bắt người
     * dùng chọn lại accepted/rejected rồi vô tình bấm trúng cái cũ.
     *
     * Lý do là BẮT BUỘC, chặn ngay ở đây chứ không đợi backend trả 409:
     * người dùng biết mình thiếu gì trước khi mất một vòng mạng. Backend
     * vẫn kiểm tra lại (feedback_store) - đây chỉ là lớp tiện lợi, không
     * phải lớp bảo vệ. */
    async reverseRegression(item) {
      if (item._deciding) return;
      if (item.status === "candidate") {
        this.showToast(this.tr("toastNothingToReverse"));
        return;
      }
      const note = (item._reverseNote || "").trim();
      if (!note) {
        this.showToast(this.tr("toastReverseNeedsReason"));
        return;
      }
      const flipped = item.status === "accepted" ? "rejected" : "accepted";
      const ok = await this._postDecision(item, flipped, { note, reversal: true });
      if (ok) {
        item._reversing = false;
        item._reverseNote = "";
      }
    },

    /* Thân chung của duyệt và lật lại - một chỗ gọi mạng, một chỗ xử lý
     * lỗi, nên hai đường không thể lệch nhau về cách báo lỗi. */
    async _postDecision(item, status, { note = null, reversal = false } = {}) {
      item._deciding = true;
      try {
        const updated = await postJSON(
          `${API.feedback}/review/regression/${item.id}/decision`,
          { status, reviewer_note: note, allow_reversal: reversal },
        );
        Object.assign(item, updated);
        this.showToast(
          reversal ? this.tr("toastRegressionReversed").replace("{status}", status)
          : status === "accepted" ? this.tr("toastRegressionAccepted")
          : this.tr("toastRegressionRejected")
        );
        return true;
      } catch (err) {
        // Giữ nguyên văn detail của backend - đó là thứ nói được VÌ SAO
        // (vd "Tách vai: 'auditor' chính là người đã ra quyết định...").
        console.error("regression decision failed:", err);
        this.showToast(`${this.tr("toastRegressionUpdateFailed")} — ${this._errText(err)}`);
        // List có thể đang lệch với DB (vd ai đó đã quyết định ở tab khác) -
        // nạp lại để người dùng thấy trạng thái THẬT thay vì bấm lại vô ích.
        this.loadRegressionCandidates();
        return false;
      } finally {
        item._deciding = false;
      }
    },

    /* ---------------- dashboard Admin (kiểm tra database) ---------------- */
    async loadAdmin() {
      this.adminLoading = true;
      try {
        const [summary, coverage, models, tables, vietstockSummary, vietstockRuns, vietstockTickers] = await Promise.all([
          getJSON(`${API.admin}/pipeline/summary`),
          getJSON(`${API.admin}/coverage?limit=200`),
          getJSON(`${API.admin}/models`),
          getJSON(`${API.admin}/tables`),
          getJSON(`${API.admin}/vietstock/summary`),
          getJSON(`${API.admin}/vietstock/runs?limit=20`),
          getJSON(`${API.admin}/vietstock/tickers?limit=250`),
        ]);
        this.pipelineSummary = summary;
        this.coverage = coverage;
        this.models = models;
        this.tables = tables;
        this.vietstockSummary = vietstockSummary;
        this.vietstockRuns = vietstockRuns;
        this.vietstockTickers = vietstockTickers;
      } catch {
        this.showToast(this.tr("toastAdminLoadFailed"));
      } finally {
        this.adminLoading = false;
      }
      // Tài liệu theo mã tải riêng (có thể nhiều dòng, và người dùng lọc
      // theo mã) - không chặn phần trên nếu bảng pdf_documents rỗng/lỗi.
      await this.loadFiles();
    },

    /* ---------------- tab Biến động giá (company_price_snapshot) -------- */
    async loadPriceTickers() {
      try {
        this.priceTickers = await getJSON(`${API.admin}/prices/tickers`);
        if (!this.priceTickerSelected && this.priceTickers.length) {
          this.priceTickerSelected = this.priceTickers[0].ticker;
        }
        if (this.priceTickerSelected) await this.loadPriceSeries();
      } catch {
        this.showToast(this.tr("toastPriceTickersFailed"));
      }
    },

    async loadPriceSeries() {
      if (!this.priceTickerSelected) return;
      this.pricesLoading = true;
      try {
        this.priceSeries = await getJSON(
          `${API.admin}/prices?ticker=${encodeURIComponent(this.priceTickerSelected)}&days=${this.priceWindowDays}`
        );
        this.$nextTick(() => this.renderPriceChart());
      } catch {
        this.showToast(this.tr("toastPriceSeriesFailed"));
        this.priceSeries = [];
      } finally {
        this.pricesLoading = false;
      }
    },

    priceTickerMeta() {
      return this.priceTickers.find((t) => t.ticker === this.priceTickerSelected) || null;
    },

    // Biểu đồ ĐẦU TIÊN thật sự dùng Chart.js trong app này (script đã có
    // sẵn trong index.html từ trước nhưng chưa nơi nào gọi `new Chart(...)`
    // - phần HITL dùng CSS thuần cho "sao đánh giá", không phải Chart.js).
    renderPriceChart() {
      if (typeof Chart === "undefined" || !this.$refs.priceChartCanvas) return;
      const labels = this.priceSeries.map((r) => r.trade_date);
      const closes = this.priceSeries.map((r) => r.close_price);
      const pct = this.priceSeries.map((r) => r.pct_change);

      if (this._priceChart) this._priceChart.destroy();
      this._priceChart = new Chart(this.$refs.priceChartCanvas.getContext("2d"), {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: `${this.priceTickerSelected} - ${this.uiLanguage === "vi" ? "Giá đóng cửa (VNĐ)" : "Close price (VND)"}`,
              data: closes,
              borderColor: "#4f9dfb",
              backgroundColor: "rgba(79,157,251,0.12)",
              yAxisID: "y",
              tension: 0.15,
              pointRadius: 0,
              borderWidth: 1.6,
              fill: true,
            },
            {
              label: this.uiLanguage === "vi" ? "% thay đổi so với phiên trước" : "% change vs previous session",
              data: pct,
              borderColor: "#f5a623",
              yAxisID: "y1",
              tension: 0,
              pointRadius: 0,
              borderWidth: 1,
              borderDash: [3, 3],
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          scales: {
            y: { position: "left", title: { display: true, text: this.uiLanguage === "vi" ? "VNĐ" : "VND" } },
            y1: {
              position: "right",
              title: { display: true, text: this.uiLanguage === "vi" ? "% / phiên" : "% / session" },
              grid: { drawOnChartArea: false },
            },
          },
          plugins: {
            legend: { labels: { color: "#c9d1d9" } },
            tooltip: {
              callbacks: {
                label: (ctx) =>
                  ctx.dataset.yAxisID === "y1"
                    ? `${ctx.dataset.label}: ${ctx.parsed.y === null ? "—" : ctx.parsed.y.toFixed(2) + "%"}`
                    : `${ctx.dataset.label}: ${ctx.parsed.y === null ? "—" : Number(ctx.parsed.y).toLocaleString("vi-VN")}`,
              },
            },
          },
        },
      });
    },

    async loadFiles() {
      this.filesLoading = true;
      try {
        const q = this.fileTickerFilter.trim().toUpperCase();
        const url = q
          ? `${API.admin}/files?ticker=${encodeURIComponent(q)}&limit=500`
          : `${API.admin}/files?limit=500`;
        this.files = await getJSON(url);
      } catch {
        this.showToast(this.tr("toastFilesLoadFailed"));
      } finally {
        this.filesLoading = false;
      }
    },

    adminStatCards() {
      // Nhãn cho từng cột trả về từ /admin/pipeline/summary - giữ đúng thứ
      // tự khai báo trong api/services/admin/main.py. Song ngữ (2026-09-22,
      // trước đây hardcode tiếng Việt only) - khoá theo pipelineSummary's
      // key, không đưa vào copy.vi/en vì đây là danh sách CỐ ĐỊNH gắn với
      // 1 endpoint cụ thể, tách riêng cho dễ đối chiếu với main.py.
      const labelsVi = {
        n_companies: "Công ty",
        n_line_items: "Dòng số liệu BCTC",
        n_pdf_documents: "File PDF (đang hoạt động)",
        n_chunks: "Chunk RAG",
        n_chunks_embedded: "Chunk đã embed",
        n_web_chunks: "Chunk web/báo chí",
        n_answer_runs: "Lượt trả lời",
        n_feedback: "Đánh giá",
        n_review_pending: "Chờ review",
        n_vietstock_documents: "Vietstock document manifest",
        n_vietstock_downloaded: "Vietstock PDF đã tải",
        n_vietstock_news_events: "Vietstock tin/sự kiện",
      };
      const labelsEn = {
        n_companies: "Companies",
        n_line_items: "Financial line items",
        n_pdf_documents: "PDF files (active)",
        n_chunks: "RAG chunks",
        n_chunks_embedded: "Embedded chunks",
        n_web_chunks: "Web/news chunks",
        n_answer_runs: "Answer runs",
        n_feedback: "Ratings",
        n_review_pending: "Pending review",
        n_vietstock_documents: "Vietstock document manifest",
        n_vietstock_downloaded: "Vietstock PDFs downloaded",
        n_vietstock_news_events: "Vietstock news/events",
      };
      const labels = this.uiLanguage === "vi" ? labelsVi : labelsEn;
      const s = this.pipelineSummary || {};
      return Object.entries(labels).map(([key, label]) => ({
        key,
        label,
        value: s[key] === null || s[key] === undefined
          ? "—"
          : Number(s[key]).toLocaleString(this.uiLanguage === "vi" ? "vi-VN" : "en-US"),
      }));
    },

    formatDate(iso) {
      try {
        return new Date(iso).toLocaleString(this.uiLanguage === "vi" ? "vi-VN" : "en-AU", { dateStyle: "medium", timeStyle: "short" });
      } catch {
        return iso;
      }
    },

    formatMetrics(m) {
      if (!m || !Object.keys(m).length) return "—";
      return Object.entries(m)
        .slice(0, 3)
        .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(3) : v}`)
        .join(", ");
    },

    showToast(msg) {
      this.toast = msg;
      this.$nextTick(() => window.lucide?.createIcons());
      clearTimeout(this._toastTimer);
      this._toastTimer = setTimeout(() => (this.toast = ""), 3200);
    },
  };
}
