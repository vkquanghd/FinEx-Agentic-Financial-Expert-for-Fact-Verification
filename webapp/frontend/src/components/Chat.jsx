import React, { useEffect, useRef, useState } from "react";
import { Send, User, Bot, ChevronDown, ChevronUp, FileText, Clock3 } from "lucide-react";
import { branchStyle } from "../api.js";

const SUGGESTIONS = [
  "Doanh thu của VNZ quý 1 năm 2026?",
  "TTS của VNZ năm 2025?",
  "Công ty kiểm toán của VNZ năm 2025 là ai?",
  "VIC nêu rủi ro gì trong báo cáo thường niên?",
  "Nếu COGS của HPG tăng 5%, biên lợi nhuận gộp thay đổi ra sao?",
  "So sánh HPG và GAS",
];

function ts(d) {
  return d.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function Message({ m }) {
  const [showReason, setShowReason] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const isUser = m.role === "user";
  const b = branchStyle(m.branch);

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div className={`h-8 w-8 shrink-0 rounded-xl flex items-center justify-center shadow
        ${isUser ? "bg-gradient-to-br from-slate-500 to-slate-700" : "bg-gradient-to-br from-indigo-500 to-sky-500"}`}>
        {isUser ? <User size={15} className="text-white" /> : <Bot size={15} className="text-white" />}
      </div>

      <div className={`max-w-[85%] ${isUser ? "text-right" : ""}`}>
        <div className={`inline-block text-left rounded-2xl px-4 py-3 shadow-sm border
          ${isUser
            ? "bg-indigo-600 text-white border-indigo-500 rounded-tr-sm"
            : "bg-white dark:bg-slate-900/80 text-slate-800 dark:text-slate-100 border-slate-200 dark:border-slate-800 rounded-tl-sm"}`}>
          <div className="answer-body text-[13.5px] leading-relaxed">{m.content}</div>

          {!isUser && m.branch && (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span className={`text-[10.5px] px-2 py-0.5 rounded-full border font-medium ${b.cls}`}>
                {b.label}{m.fallback_from ? ` · fallback từ ${m.fallback_from}` : ""}
              </span>
              {m.latency_ms != null && (
                <span className="text-[10.5px] text-slate-400 inline-flex items-center gap-1">
                  <Clock3 size={11} /> {(m.latency_ms / 1000).toFixed(1)}s
                </span>
              )}
              {m.reason && (
                <button onClick={() => setShowReason(!showReason)}
                        className="text-[10.5px] text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 inline-flex items-center gap-0.5">
                  vì sao nhánh này {showReason ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                </button>
              )}
              {m.sources?.length > 0 && (
                <button onClick={() => setShowSources(!showSources)}
                        className="text-[10.5px] text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 inline-flex items-center gap-0.5">
                  <FileText size={11} /> {m.sources.length} nguồn {showSources ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                </button>
              )}
            </div>
          )}

          {showReason && (
            <div className="mt-2 text-[11px] text-slate-500 dark:text-slate-400 border-t border-slate-200 dark:border-slate-800 pt-2">
              {m.resolved_question && m.resolved_question !== m.original_question && (
                <div className="mb-1">Câu đã giải quyết ngữ cảnh: <i>{m.resolved_question}</i></div>
              )}
              {m.reason}
            </div>
          )}
          {showSources && (
            <div className="mt-2 text-[11px] text-slate-500 dark:text-slate-400 border-t border-slate-200 dark:border-slate-800 pt-2 space-y-0.5">
              {m.sources.map((s, i) => (
                <div key={i}>[{i + 1}] {s.ticker} · {s.doctype} · {s.source_file} · trang {s.page}
                  {s.year ? ` · năm ${s.year}` : ""} {s.similarity != null ? `· sim=${s.similarity}` : ""}</div>
              ))}
            </div>
          )}
        </div>
        <div className={`mt-1 text-[10px] text-slate-400 dark:text-slate-500 ${isUser ? "pr-1" : "pl-1"}`}>
          {ts(m.at)}{!isUser && m.awaitingFeedback ? " · ⭐ chưa đánh giá (panel HITL bên phải)" : ""}
        </div>
      </div>
    </div>
  );
}

function Thinking() {
  // Trạng thái "đang chạy" trực quan: pipeline từng bước + chấm nhảy.
  const steps = ["Khai triển viết tắt", "Giải quyết ngữ cảnh", "Router chọn nhánh", "Truy xuất & tổng hợp"];
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setI((v) => Math.min(v + 1, steps.length - 1)), 1800);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="flex gap-3">
      <div className="h-8 w-8 shrink-0 rounded-xl bg-gradient-to-br from-indigo-500 to-sky-500 flex items-center justify-center animate-pulse-fast">
        <Bot size={15} className="text-white" />
      </div>
      <div className="rounded-2xl rounded-tl-sm px-4 py-3 bg-white dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800">
        <div className="flex items-center gap-2 text-[12px] text-slate-500 dark:text-slate-300">
          <span className="flex gap-1">
            {[0, 1, 2].map((d) => (
              <span key={d} className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: `${d * 0.15}s` }} />
            ))}
          </span>
          {steps[i]}…
        </div>
        <div className="mt-2 flex gap-1">
          {steps.map((s, idx) => (
            <div key={s} className={`h-1 flex-1 rounded-full ${idx <= i ? "bg-indigo-500" : "bg-slate-200 dark:bg-slate-700"}`} />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function Chat({ messages, busy, awaitingFeedback, onSend }) {
  const [input, setInput] = useState("");
  const endRef = useRef(null);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages, busy]);

  const send = () => {
    const q = input.trim();
    if (!q || busy || awaitingFeedback) return;
    setInput("");
    onSend(q);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">
        {messages.length === 0 && (
          <div className="text-center mt-12">
            <div className="text-slate-400 dark:text-slate-500 text-sm mb-4">
              Hỏi về BCTC, tài liệu công bố, kịch bản what-if, nhân quả, so sánh giá…
            </div>
            <div className="flex flex-wrap justify-center gap-2 max-w-xl mx-auto">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => onSend(s)}
                        disabled={busy || awaitingFeedback}
                        className="text-[12px] px-3 py-1.5 rounded-full border border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-indigo-400 hover:text-indigo-500 transition disabled:opacity-40">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => <Message key={i} m={m} />)}
        {busy && <Thinking />}
        <div ref={endRef} />
      </div>

      <div className="border-t border-slate-200 dark:border-slate-800 p-3">
        {awaitingFeedback && (
          <div className="mb-2 text-[12px] text-amber-600 dark:text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-1.5">
            ⭐ Đánh giá (hoặc bỏ qua) câu trả lời trong panel HITL bên phải trước khi hỏi tiếp — mỗi
            đánh giá là một tín hiệu huấn luyện.
          </div>
        )}
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            disabled={busy || awaitingFeedback}
            placeholder={busy ? "Đang xử lý…" : "Nhập câu hỏi (VD: LNST của VNZ quý 1 năm 2026?)"}
            className="flex-1 rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-4 py-2.5 text-[13.5px] text-slate-800 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 disabled:opacity-50"
          />
          <button onClick={send} disabled={busy || awaitingFeedback || !input.trim()}
                  className="rounded-xl bg-gradient-to-br from-indigo-500 to-sky-500 text-white px-4 flex items-center gap-1.5 text-sm font-semibold shadow-lg shadow-indigo-500/25 hover:opacity-90 transition disabled:opacity-40">
            <Send size={15} /> Hỏi
          </button>
        </div>
      </div>
    </div>
  );
}
