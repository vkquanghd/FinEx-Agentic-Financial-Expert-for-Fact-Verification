"""RQ7 - Conversational Context Resolver (see docs/Ghi_chu_thao_luan_phan_bien_2026-07-26.md
section 8/8.10 for the full design discussion and rationale). Sits BEFORE
src/router/rules.py in the pipeline: turns a context-dependent follow-up
question ("Còn năm trước thì sao?") into a standalone question the existing
router/branches can handle unchanged, using per-session conversation memory
(NOT baked into any fine-tuned weights - see context_resolver.py's module
docstring for why that distinction matters).
"""
