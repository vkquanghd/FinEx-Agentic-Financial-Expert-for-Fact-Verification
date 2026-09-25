"""Evaluation layer - src/evaluation/. Implements the metrics fixed in
docs/Thesis_Direction_Redesign.pdf section 7 ("Baseline đối chứng và độ đo
đánh giá"):
  - Routing Accuracy + Macro F1 (RQ1, RQ2, RQ6)
  - Exact Match cho câu hỏi số liệu (RQ2)
  - Faithfulness/Relevance theo RAGAS (RQ1, RQ4)
  - Tỷ lệ nhầm kỳ báo cáo trước/sau temporal anchor (RQ4)
  - (RQ6, not yet built here) sai số ATE so với giá trị mô phỏng đã biết

Deliberately does NOT implement: SQL hallucination rate (RQ2 - needs Nhánh B
Text-to-SQL Agent, not built yet), before/after regime-signal comparison
(RQ3 - needs Trend-Aligned Dynamic Reranking, not built yet), ABSA
qualitative eval (RQ5 - ABSA not built yet). See docs/PROGRESS.md for the
current build-out order - this module is built incrementally as the branches
it measures get built, per the 10-week plan's own sequencing (benchmark +
eval come in weeks 7-8, after the branches being measured).
"""
