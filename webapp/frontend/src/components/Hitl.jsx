import React, { useEffect, useState } from "react";
import { Star, SkipForward, Inbox, GaugeCircle, CheckCircle2, Eye, XCircle, Info } from "lucide-react";
import { api, REASON_LABELS } from "../api.js";

/* Panel HITL - TÁCH BIỆT khỏi chat theo đúng bản chất 3 lớp (DSA v6 §7):
   L1 thu tín hiệu (rate/skip) -> L2 uncertainty sampling (review queue) ->
   L3 batch gate (N>=50 & mean<3.5). Panel này cho thấy CẢ vòng đời của một
   tín hiệu feedback, không chỉ cái nút sao. */

function Stars({ value, onChange }) {
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((n) => (
        <button key={n} onClick={() => onChange(n)} title={`${n} sao`}
                className="transition hover:scale-110">
          <Star size={22}
                className={n <= value ? "fill-amber-400 text-amber-400" : "text-slate-300 dark:text-slate-600"} />
        </button>
      ))}
    </div>
  );
}

function RatingCard({ pending, onDone }) {
  const [rating, setRating] = useState(0);
  const [reason, setReason] = useState("");
  const [freeText, setFreeText] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => { setRating(0); setReason(""); setFreeText(""); setErr(null); }, [pending?.answer_run_id]);

  if (!pending) {
    return (
      <div className="text-[12px] text-slate-400 dark:text-slate-500 italic">
        Chưa có câu trả lời nào chờ đánh giá. Hỏi một câu bên trái để bắt đầu vòng feedback.
      </div>
    );
  }

  const submit = async (skip) => {
    setBusy(true); setErr(null);
    try {
      const res = await api.rate({
        answer_run_id: pending.answer_run_id,
        rating: skip ? null : rating,
        is_skip: skip,
        reason_code: reason || null,
        free_text: freeText || null,
      });
      onDone(res);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-3">
      <div className="text-[12px] text-slate-500 dark:text-slate-400 line-clamp-2">
        <span className="font-semibold text-slate-700 dark:text-slate-200">Q:</span> {pending.question}
      </div>
      <Stars value={rating} onChange={setRating} />
      <select value={reason} onChange={(e) => setReason(e.target.value)}
              className="w-full text-[12px] rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1.5 text-slate-700 dark:text-slate-200">
        <option value="">— Lý do (tùy chọn) —</option>
        {Object.entries(REASON_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
      </select>
      <input value={freeText} onChange={(e) => setFreeText(e.target.value)} maxLength={1000}
             placeholder="Ghi chú thêm (tùy chọn)…"
             className="w-full text-[12px] rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1.5 text-slate-700 dark:text-slate-200 placeholder:text-slate-400" />
      {err && <div className="text-[11px] text-rose-500">{err}</div>}
      <div className="flex gap-2">
        <button disabled={busy || rating === 0} onClick={() => submit(false)}
                className="flex-1 rounded-lg bg-indigo-600 text-white text-[12.5px] font-semibold py-1.5 hover:opacity-90 disabled:opacity-40 transition">
          Gửi đánh giá
        </button>
        <button disabled={busy} onClick={() => submit(true)}
                title="Skip cũng là dữ liệu - hệ thống ghi nhận bạn đã thấy nhưng bỏ qua"
                className="rounded-lg border border-slate-300 dark:border-slate-700 text-slate-500 dark:text-slate-300 text-[12.5px] px-3 py-1.5 hover:bg-slate-100 dark:hover:bg-slate-800 transition inline-flex items-center gap-1">
          <SkipForward size={13} /> Bỏ qua
        </button>
      </div>
      {rating > 0 && rating <= 2 && (
        <div className="text-[11px] text-amber-600 dark:text-amber-400">
          Rating ≤ 2 sẽ tự vào hàng đợi review (uncertainty sampling — Settles 2009).
        </div>
      )}
    </div>
  );
}

function GateStats({ stats }) {
  if (!stats) return <div className="text-[12px] text-slate-400 italic">Đang tải…</div>;
  const g = stats.gate || {};
  const pct = Math.min(100, Math.round(((g.n_progress || 0) / (g.n_required || 50)) * 100));
  return (
    <div className="space-y-2.5">
      <div className="flex justify-between text-[12px] text-slate-600 dark:text-slate-300">
        <span>Tiến độ batch gate</span>
        <span className="font-mono">{g.n_progress ?? 0}/{g.n_required ?? 50} feedback</span>
      </div>
      <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
        <div className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all" style={{ width: `${pct}%` }} />
      </div>
      <div className="grid grid-cols-3 gap-2 text-center">
        {[
          ["Mean rating", stats.mean_rating != null ? stats.mean_rating.toFixed(2) : "—"],
          ["Skip", stats.n_skip ?? 0],
          ["Chờ review", stats.pending_reviews ?? 0],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg bg-slate-100 dark:bg-slate-800/70 py-1.5">
            <div className="text-[13px] font-bold text-slate-800 dark:text-slate-100">{v}</div>
            <div className="text-[10px] text-slate-500">{k}</div>
          </div>
        ))}
      </div>
      {stats.per_branch?.length > 0 && (
        <div className="text-[11px] text-slate-500 dark:text-slate-400 space-y-0.5">
          {stats.per_branch.map((b) => (
            <div key={b.branch} className="flex justify-between">
              <span className="font-mono">{b.branch}</span>
              <span>{b.n_rated} rated · mean {b.mean_rating != null ? Number(b.mean_rating).toFixed(2) : "—"}</span>
            </div>
          ))}
        </div>
      )}
      <div className={`text-[11px] rounded-lg px-2 py-1.5 border ${g.would_trigger
        ? "text-rose-500 border-rose-500/30 bg-rose-500/10"
        : "text-emerald-600 dark:text-emerald-400 border-emerald-500/30 bg-emerald-500/10"}`}>
        {g.would_trigger
          ? "⚠ Gate ĐỦ điều kiện mở — cân nhắc retrain theo batch."
          : "Gate chưa mở — đúng thiết kế: không chỉnh model theo từng label (Christiano 2017)."}
      </div>
    </div>
  );
}

function ReviewQueue({ items, onAction }) {
  const [noteById, setNoteById] = useState({});
  if (!items?.length) {
    return <div className="text-[12px] text-slate-400 italic">Hàng đợi trống — chưa có ca bất định nào cần người xem.</div>;
  }
  return (
    <div className="space-y-3">
      {items.map((it) => (
        <div key={it.review_id} className="rounded-xl border border-slate-200 dark:border-slate-800 p-3 bg-slate-50 dark:bg-slate-900/60">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10.5px] px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-600 border border-amber-500/30">
              nguồn: {it.source}
            </span>
            <span className="text-[10px] text-slate-400 font-mono">#{it.review_id}</span>
          </div>
          <div className="text-[12px] text-slate-700 dark:text-slate-200 font-medium line-clamp-2">{it.question_text}</div>
          <div className="text-[11.5px] text-slate-500 dark:text-slate-400 line-clamp-2 mt-0.5">{it.answer_text}</div>
          <input
            value={noteById[it.review_id] || ""}
            onChange={(e) => setNoteById({ ...noteById, [it.review_id]: e.target.value })}
            placeholder="Ghi chú adjudication…"
            className="mt-2 w-full text-[11.5px] rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1 text-slate-700 dark:text-slate-200 placeholder:text-slate-400"
          />
          <div className="mt-2 flex gap-1.5">
            <button onClick={() => onAction(it.review_id, "resolved", noteById[it.review_id])}
                    className="flex-1 text-[11px] rounded-lg bg-emerald-600/90 text-white py-1 inline-flex items-center justify-center gap-1 hover:opacity-90">
              <CheckCircle2 size={12} /> Đã xử lý
            </button>
            <button onClick={() => onAction(it.review_id, "reviewed", noteById[it.review_id])}
                    className="flex-1 text-[11px] rounded-lg bg-sky-600/90 text-white py-1 inline-flex items-center justify-center gap-1 hover:opacity-90">
              <Eye size={12} /> Đã xem
            </button>
            <button onClick={() => onAction(it.review_id, "dismissed", noteById[it.review_id])}
                    className="flex-1 text-[11px] rounded-lg bg-slate-500/80 text-white py-1 inline-flex items-center justify-center gap-1 hover:opacity-90">
              <XCircle size={12} /> Bỏ
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function Hitl({ pending, onRated }) {
  const [stats, setStats] = useState(null);
  const [queue, setQueue] = useState([]);

  const refresh = async () => {
    try { setStats(await api.hitlStats()); } catch {}
    try { setQueue((await api.reviewQueue()).items || []); } catch {}
  };
  useEffect(() => { refresh(); }, [pending?.answer_run_id]);

  const onAction = async (id, status, note) => {
    try { await api.reviewAction(id, status, note); refresh(); } catch {}
  };

  const Section = ({ icon: Icon, title, children, accent }) => (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/70 p-4 shadow-sm">
      <div className={`flex items-center gap-2 mb-3 text-[13px] font-bold ${accent}`}>
        <Icon size={15} /> {title}
      </div>
      {children}
    </div>
  );

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <Section icon={Star} title="L1 · Đánh giá câu trả lời" accent="text-amber-500">
        <RatingCard pending={pending} onDone={(res) => { onRated(res); refresh(); }} />
      </Section>

      <Section icon={Inbox} title="L2 · Hàng đợi review (uncertainty sampling)" accent="text-sky-500">
        <ReviewQueue items={queue} onAction={onAction} />
      </Section>

      <Section icon={GaugeCircle} title="L3 · Batch retrain gate" accent="text-emerald-500">
        <GateStats stats={stats} />
      </Section>

      <div className="rounded-2xl border border-dashed border-slate-300 dark:border-slate-700 p-3 text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
        <div className="flex items-center gap-1.5 font-semibold text-slate-600 dark:text-slate-300 mb-1">
          <Info size={12} /> Vì sao HITL tách khỏi chat?
        </div>
        Dữ liệu tài chính không phải lúc nào cũng được confirm chắc chắn — nên hệ này dùng
        <b> batch-feedback HITL</b>: rating của bạn là <i>tín hiệu</i> (không phải ground truth),
        ca bất định được dồn cho người xem (Settles 2009), và model chỉ được cân nhắc chỉnh
        theo <i>lô</i> khi đủ bằng chứng (Christiano 2017) — không bao giờ theo từng câu.
      </div>
    </div>
  );
}
