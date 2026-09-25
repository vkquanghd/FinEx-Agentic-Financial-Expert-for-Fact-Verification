import React, { useEffect, useMemo, useState } from "react";
import TopBar from "./components/TopBar.jsx";
import Chat from "./components/Chat.jsx";
import Hitl from "./components/Hitl.jsx";
import { api } from "./api.js";

export default function App() {
  const [dark, setDark] = useState(() => (localStorage.getItem("theme") || "dark") === "dark");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState({ qa: false, feedback: false });
  const sessionId = useMemo(() => `web_${Math.random().toString(36).slice(2, 10)}`, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    api.health().then(setHealth);
    const t = setInterval(() => api.health().then(setHealth), 30000);
    return () => clearInterval(t);
  }, []);

  // câu trả lời mới nhất chưa được rate -> gate input + đẩy sang panel HITL
  const pendingMsg = [...messages].reverse().find((m) => m.role === "assistant" && m.awaitingFeedback);
  const pending = pendingMsg
    ? { answer_run_id: pendingMsg.answer_run_id, question: pendingMsg.original_question }
    : null;

  const onSend = async (q) => {
    setMessages((ms) => [...ms, { role: "user", content: q, at: new Date() }]);
    setBusy(true);
    try {
      const history = messages.filter((m) => m.role === "user").map((m) => m.content);
      const res = await api.ask(q, sessionId, history);
      setMessages((ms) => [...ms, {
        role: "assistant",
        content: res.answer,
        at: new Date(),
        branch: res.branch,
        fallback_from: res.fallback_from,
        reason: res.intent_reason,
        resolved_question: res.resolved_question,
        original_question: res.original_question,
        sources: res.sources,
        latency_ms: res.latency_ms,
        answer_run_id: res.answer_run_id,
        // chỉ gate khi answer_run log thành công (có id để rate)
        awaitingFeedback: res.answer_run_id != null,
      }]);
    } catch (e) {
      setMessages((ms) => [...ms, {
        role: "assistant",
        content: `Lỗi gọi QA service: ${e}. Kiểm tra backend đã chạy chưa (bash webapp/backend/run_dev.sh).`,
        at: new Date(),
      }]);
    } finally {
      setBusy(false);
    }
  };

  const onRated = () => {
    setMessages((ms) => ms.map((m) => (m.awaitingFeedback ? { ...m, awaitingFeedback: false } : m)));
  };

  return (
    <div className="h-screen flex flex-col">
      <TopBar dark={dark} setDark={setDark} health={health} sessionId={sessionId} />
      <main className="flex-1 min-h-0 max-w-7xl w-full mx-auto grid grid-cols-1 lg:grid-cols-[1fr_390px]">
        <section className="min-h-0 border-x border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-[#0d1428]">
          <Chat messages={messages} busy={busy} awaitingFeedback={!!pending} onSend={onSend} />
        </section>
        <aside className="hidden lg:block min-h-0 bg-slate-100/70 dark:bg-[#0b1020] border-r border-slate-200 dark:border-slate-800">
          <Hitl pending={pending} onRated={onRated} />
        </aside>
      </main>
    </div>
  );
}
