// API client - 2 service mặt người dùng (qa + feedback).
// Đổi host/port qua env Vite (VITE_QA_URL / VITE_FEEDBACK_URL) khi deploy;
// mặc định trỏ localhost đúng port trong backend/shared/bootstrap.py.
const QA_URL = import.meta.env.VITE_QA_URL || "http://localhost:8101";
const FEEDBACK_URL = import.meta.env.VITE_FEEDBACK_URL || "http://localhost:8102";

async function post(base, path, body) {
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

async function get(base, path) {
  const res = await fetch(`${base}${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

export const api = {
  ask: (question, sessionId, history) =>
    post(QA_URL, "/ask", { question, session_id: sessionId, history }),
  rate: (payload) => post(FEEDBACK_URL, "/rate", payload),
  reviewQueue: () => get(FEEDBACK_URL, "/admin/review-queue"),
  reviewAction: (id, status, note) =>
    post(FEEDBACK_URL, `/admin/review-queue/${id}`, { status, resolution_note: note || null }),
  hitlStats: () => get(FEEDBACK_URL, "/admin/hitl-stats"),
  health: async () => {
    const out = { qa: false, feedback: false };
    try { await get(QA_URL, "/health"); out.qa = true; } catch {}
    try { await get(FEEDBACK_URL, "/health"); out.feedback = true; } catch {}
    return out;
  },
};

export const REASON_LABELS = {
  good: "Đúng, hài lòng",
  wrong_number: "Sai số liệu",
  wrong_branch: "Sai nhánh routing",
  missing_evidence: "Thiếu nguồn/bằng chứng",
  hallucination: "Bịa đặt (hallucination)",
  outdated: "Dữ liệu cũ",
  unclear: "Trả lời không rõ",
  too_slow: "Quá chậm",
  other: "Khác",
};

export const BRANCH_STYLE = {
  dag: { label: "DAG · tra cứu số liệu", cls: "bg-emerald-500/15 text-emerald-500 border-emerald-500/30" },
  rag: { label: "RAG · tài liệu công bố", cls: "bg-sky-500/15 text-sky-500 border-sky-500/30" },
  whatif: { label: "What-if · kịch bản", cls: "bg-amber-500/15 text-amber-600 border-amber-500/30" },
  causal: { label: "Causal · nhân quả", cls: "bg-purple-500/15 text-purple-500 border-purple-500/30" },
  compare: { label: "Compare · so sánh giá", cls: "bg-pink-500/15 text-pink-500 border-pink-500/30" },
  absa: { label: "ABSA · cảm xúc", cls: "bg-cyan-500/15 text-cyan-600 border-cyan-500/30" },
};

export function branchStyle(branch) {
  if (!branch) return { label: "?", cls: "bg-slate-500/15 text-slate-500 border-slate-500/30" };
  const base = branch.replace(/_unsupported$/, "");
  const s = BRANCH_STYLE[base] || { label: branch, cls: "bg-slate-500/15 text-slate-500 border-slate-500/30" };
  return branch.endsWith("_unsupported")
    ? { label: `${s.label} (ngoài phạm vi)`, cls: "bg-slate-500/15 text-slate-400 border-slate-500/30" }
    : s;
}
