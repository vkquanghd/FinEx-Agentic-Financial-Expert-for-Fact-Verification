import React, { useEffect, useState } from "react";
import { Moon, Sun, Activity, CircleDot } from "lucide-react";

// TopBar: tiêu đề + đồng hồ thời gian thực (yêu cầu "chỉnh thời gian rõ
// ràng") + toggle dark mode (persist localStorage) + trạng thái 2 service.
export default function TopBar({ dark, setDark, health, sessionId }) {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const dot = (ok) => (
    <span className={`inline-flex items-center gap-1 text-xs ${ok ? "text-emerald-500" : "text-rose-500"}`}>
      <CircleDot size={12} className={ok ? "" : "animate-pulse"} />
    </span>
  );

  return (
    <header className="sticky top-0 z-20 backdrop-blur-md bg-white/70 dark:bg-[#0b1020]/70 border-b border-slate-200 dark:border-slate-800">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-indigo-500 via-sky-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-indigo-500/25">
            <Activity size={18} className="text-white" />
          </div>
          <div>
            <div className="font-extrabold text-slate-900 dark:text-white leading-tight">
              Financial QA <span className="text-indigo-500">· VN Disclosures</span>
            </div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">
              Adaptive Retrieval Routing · local-only · dữ liệu không rời máy
            </div>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-4">
          <div className="hidden sm:flex flex-col items-end">
            <div className="font-mono text-sm font-semibold text-slate-800 dark:text-slate-100 tabular-nums">
              {now.toLocaleTimeString("vi-VN")}
            </div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">
              {now.toLocaleDateString("vi-VN", { weekday: "long", day: "2-digit", month: "2-digit", year: "numeric" })}
            </div>
          </div>

          <div className="hidden md:flex items-center gap-2 px-2 py-1 rounded-lg bg-slate-200/60 dark:bg-slate-800/60 text-[11px] text-slate-600 dark:text-slate-300">
            {dot(health.qa)} QA {dot(health.feedback)} HITL
          </div>

          <button
            onClick={() => setDark(!dark)}
            title={dark ? "Chuyển sáng" : "Chuyển tối"}
            className="h-9 w-9 rounded-xl border border-slate-300 dark:border-slate-700 flex items-center justify-center text-slate-600 dark:text-amber-300 hover:bg-slate-200/70 dark:hover:bg-slate-800 transition"
          >
            {dark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>
      {sessionId && (
        <div className="max-w-7xl mx-auto px-4 pb-1 text-[10px] text-slate-400 dark:text-slate-500 font-mono">
          session: {sessionId}
        </div>
      )}
    </header>
  );
}
