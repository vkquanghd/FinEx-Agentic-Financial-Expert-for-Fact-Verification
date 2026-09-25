"""
PhoBERT / Sentence-BERT-style embeddings.

Model: BAAI/bge-m3 (1024-dim dense embeddings), chosen after checking
VN-MTEB (Pham et al. 2025, arXiv:2507.21500) -- scored well on Vietnamese
retrieval and supports up to 8192-token input. NOT vietnamese-bi-encoder
(768-dim, floated earlier in docs/data_plan.md) -- that model scored
notably lower on the same benchmark's retrieval task (25.37 vs bge-m3's
much higher retrieval score). See docs/Thesis_Direction_Redesign.pdf
section 4 for the full comparison and why this beats fine-tuning an
embedding model within the 10-week scope (no query-passage pair data to
fine-tune on yet).

Uses sentence-transformers (simpler API than BAAI's own FlagEmbedding
package for dense-only output, which is all this project needs -- bge-m3
also supports sparse/multi-vector retrieval via FlagEmbedding, not used
here to keep pgvector storage/query simple: one 1024-dim column, cosine
similarity, no hybrid fusion logic to maintain).

First call downloads ~2.2GB of model weights from Hugging Face and caches
them locally (~/.cache/huggingface) -- do this on a machine with a real
internet connection, not an offline/sandboxed one. CPU works (just slower,
seconds not milliseconds per batch); GPU auto-used if available via
CUDA/MPS (sentence-transformers picks this up automatically, no code
change needed either way).

2026-07-21: "auto-used if available" turned out to not be verifiable in
practice -- SentenceTransformer() with no explicit `device` doesn't log
what it picked, so a genuinely-slow run (656,918 chunks from the 218-ticker
TiniX batch) and a silent CPU fallback look identical from the console
output. Made the device selection explicit + logged so this is never
ambiguous again -- confirm the printed line says "mps"/"cuda", not "cpu",
on the very first call.
"""
import os
from functools import lru_cache

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # must match the VECTOR(1024) column in infra/pgvector/migrations/001_disclosure_chunks.sql

# bge-m3 was NOT trained with e5-style "query: " / "passage: " input
# prefixes, so none are added here -- unlike e5/gte-family models, raw text
# is the expected input for both queries and documents.


def _pick_device() -> str:
    """Select the embedding runtime, with an explicit demo override.

    MPS is normally preferred on Apple Silicon, but the transformer stack can
    stall on some local PyTorch/MPS combinations.  ``FINEX_EMBEDDING_DEVICE``
    lets the operator choose ``cpu``, ``mps`` or ``cuda`` without changing
    corpus vectors or source code.  ``auto`` retains the previous policy.
    """
    requested = os.getenv("FINEX_EMBEDDING_DEVICE", "auto").strip().lower()
    if requested in {"cpu", "mps", "cuda"}:
        return requested
    if requested not in {"", "auto"}:
        print(f"[embeddings] unknown FINEX_EMBEDDING_DEVICE={requested!r}; using auto")
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    device = _pick_device()
    print(f"[embeddings] loading {EMBEDDING_MODEL_NAME} on device='{device}'")
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


def embed_texts(texts, batch_size: int = 16, show_progress: bool = False):
    """Embed a list of strings, return a list of 1024-dim float lists
    (plain Python lists, ready to format into a pgvector literal)."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress,
        # Cosine similarity assumes unit-normalized vectors -- matches the
        # `vector_cosine_ops` index in the pgvector migration. If this is
        # ever changed to False, the index type must change too.
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embed_query(text: str):
    """Same model/normalization as embed_texts, single-string convenience
    wrapper for query-time use (src/retrieval/rag_pipeline.py, not built yet)."""
    return embed_texts([text])[0]


if __name__ == "__main__":
    # Sanity check: confirms the model downloads/loads and produces vectors
    # of the expected dimension, without touching pgvector at all.
    #   python3 -m src.retrieval.embeddings
    sample = ["Tổng tài sản của công ty tăng 12% so với cùng kỳ năm trước."]
    vecs = embed_texts(sample)
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"Vectors: {len(vecs)}, dim: {len(vecs[0])} (expected {EMBEDDING_DIM})")
    assert len(vecs[0]) == EMBEDDING_DIM, "Dimension mismatch -- check EMBEDDING_MODEL_NAME vs pgvector schema"
    print("OK")
