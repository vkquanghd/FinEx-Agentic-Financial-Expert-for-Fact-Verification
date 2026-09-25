# FinEx — Agentic Financial Expert for Fact Verification

RMIT Minor Thesis project (Master of AI). FinEx answers financial questions
about Vietnamese listed companies by routing each question to the retrieval
strategy best suited to it, instead of relying on a single retrieval-augmented
generation (RAG) pipeline for everything.

The thesis compares three systems on the same question set:

| System | What it does |
|---|---|
| **S1** | RAG-only — every question goes through document retrieval + LLM generation |
| **S2** | Structured-lookup-only — every question goes through a bounded SQL lookup over parsed financial statements |
| **S3** | **The router** — classifies each question and sends it to whichever of 12 branches actually fits (structured lookup, ratio lookup, what-if scenario, causal estimate, multi-company compare, aspect-based sentiment, auditor lookup, price snapshot, or RAG as the fallback) |

This repository is the implementation behind that comparison, plus the
conversational-memory layer (RQ3) that lets S3 resolve follow-up questions
("What about its total liabilities?") using conversation history.

## Architecture

<img src="assets/diagrams/system-architecture.png" alt="FinEx system architecture" width="800">

Three data paths feed into the router: a structured ETL pipeline (Vietstock
exports + OCR'd disclosure PDFs → parsed financial statements), a RAG
pipeline (disclosure chunks → embeddings → pgvector), and a set of
analytical modules (what-if calculator, causal treatment-effect estimator,
ABSA sentiment classifier). The router decides per-question which of these
to use.

## The router (S3)

<img src="assets/diagrams/query-routing.png" alt="Query routing flow" width="800">

Regex/keyword-marker classification (`src/router/rules.py`) picks one of 12
branches per question — no LLM call needed for routing itself, only for
generation once a branch is selected. See `assets/diagrams/` for the
per-branch pipeline diagrams (RAG, ingestion, what-if, causal,
context-resolution for RQ3).

## Research questions

- **RQ1** — Does adaptive routing improve answer quality/relevance over a
  single fixed strategy (S1 or S2 alone)?
- **RQ2** — How much latency and quantitative-correctness cost/benefit does
  routing add compared to the two baselines?
- **RQ3** — Can the router resolve conversational follow-up questions
  (pronoun reference, ellipsis, topic switches) using history, without
  breaking questions that are already self-contained?

## Quick start

```bash
git clone <this-repo-url>
cd finex
cp .env.example .env            # fill in DB credentials, etc.
docker compose up -d --scale ollama=0   # if using a native Ollama install
# or: docker compose up -d              # if letting Docker run Ollama too
```

Full setup (native `dev.sh` mode vs Docker, first-time data loading, Ollama
GPU notes, troubleshooting) is in **[RUNNING.md](RUNNING.md)**.

## Project layout

```
src/core/        config, DB connection, shared constants
src/ingestion/   Vietstock ETL (company + macro data), PDF OCR
src/structured/  bounded structured lookup + what-if calculator (System S2)
src/causal/      cross-sectional treatment-effect analysis, macro/gold lookup
src/retrieval/   RAG path - chunking, embeddings, pgvector, query pipeline (System S1)
src/router/      the adaptive router across 12 branches (System S3)
src/absa/        aspect-based sentiment analysis (zero-shot LLM prompting)
src/dialogue/    conversational context resolution (RQ3 memory)
src/models/      local LLM generation via Ollama (Qwen3)
api/              FastAPI services (gateway/qa/feedback/admin) + web frontend
scripts/          batch-test and maintenance scripts, incl. scripts/test_router_batch.py
infra/postgres/  PostgreSQL + pgvector schema (auto-applied by Docker)
data/             NOT included in this repo — see RUNNING.md for how to populate it
```

## Requirements

Two separate dependency sets, matching the two things you might run:

- `requirements.txt` (repo root) — the research/ETL/notebook environment
  (pandas, scikit-learn, sentence-transformers, streamlit, etc.)
- `api/requirements.txt` — the lighter API-service environment (FastAPI +
  the subset of `src/` the services import)

See RUNNING.md for which one to install for which workflow.

## License

See [LICENSE](LICENSE). *(Currently unset — choose a license before making
this repository public.)*
