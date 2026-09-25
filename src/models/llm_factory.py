"""LLM client - local Qwen3 via Ollama (https://ollama.com), no API key,
nothing leaves the machine. Single place every other module calls into, so
swapping providers later (e.g. a paid API for comparison) only touches this
file - see docs/Thesis_Direction_Redesign.pdf section 9 for why fine-tuning
and cloud APIs were deliberately ruled out for the current 10-week scope.

One-time setup (see README.md for the full step):
    brew install ollama
    ollama serve &              # or just open the Ollama.app
    ollama pull qwen3:8b        # ~5.2GB (q4_K_M quant) - or qwen3:4b (~2.5GB) if slower/lighter needed

Usage:
    from src.models.llm_factory import generate
    text = generate("Tóm tắt đoạn văn sau...", system="Bạn là trợ lý tài chính.")
"""
import os
import time

import requests

from src.core.config import OLLAMA_HOST, OLLAMA_MODEL

DEFAULT_TIMEOUT = int(os.getenv("FINEX_LLM_TIMEOUT_S", "180"))
                       # seconds - local CPU/MPS inference on a long context can be slow, not a network call.
                       # Raised from 120 (2026-07-22): a real run_eval.py pass over 70 RAG items hit a
                       # ReadTimeout on item ~51/70 after ~25 min of sustained generation (likely thermal
                       # throttling on a MacBook running continuous local inference, not a hung server -
                       # every prior call in that same run succeeded within 10-50s). 180s + the retry below
                       # absorb a transient slow-down without needing to guess a "correct" fixed timeout.
DEFAULT_RETRIES = int(os.getenv("FINEX_LLM_RETRIES", "2"))
                       # retry only on timeout/connection errors (transient), not on a real 4xx/5xx from Ollama itself


class OllamaNotReachable(RuntimeError):
    """Raised when Ollama isn't running or the model isn't pulled yet -
    distinct from a generic request exception so callers (e.g. the router)
    can catch this specifically and degrade gracefully instead of crashing."""


def check_ollama() -> bool:
    """Quick sanity check, same pattern as src.core.db.check_connection()."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(m.startswith(OLLAMA_MODEL.split(":")[0]) for m in models):
            print(f"WARNING - Ollama is running but '{OLLAMA_MODEL}' isn't pulled yet. "
                  f"Run: ollama pull {OLLAMA_MODEL}")
            print(f"Models currently available: {models or '(none)'}")
            return False
        print(f"OK - Ollama reachable at {OLLAMA_HOST}, '{OLLAMA_MODEL}' available")
        return True
    except requests.exceptions.RequestException as e:
        print(f"FAILED to reach Ollama at {OLLAMA_HOST}: {e}")
        print("Is it running? Try: ollama serve  (or open the Ollama.app)")
        return False


def _generate_raw(prompt: str, system: str, temperature: float, timeout: int, retries: int,
                   think: bool = False) -> dict:
    """Shared retry/request logic for generate()/generate_with_stats() below
    (2026-09-08 refactor - was inlined in generate() alone; pulled out so
    generate_with_stats() doesn't duplicate the retry/backoff logic, and so
    generate()'s public signature/behavior is completely unchanged for every
    existing caller). Returns Ollama's FULL response JSON (not just
    ["response"]) - the extra fields (prompt_eval_count, eval_count,
    total_duration, etc., all present when stream=False) are what
    generate_with_stats() surfaces for the bilingual latency/token benchmark
    (scripts/bilingual_latency_benchmark.py) - see Ollama's /api/generate
    docs for field meanings; durations are nanoseconds on the wire.

    Retries on timeout/connection errors, AND on a 5xx HTTP error status from
    Ollama itself (added 2026-07-22 for timeout/connection; extended
    2026-09-08 to cover 5xx after a real bilingual test run hit "500 Server
    Error" from Ollama mid-batch, following several 90-120s generations in a
    row - same underlying cause as the ReadTimeout incident below, just
    surfacing as a 500 this time instead of a hung connection). A 4xx (e.g.
    404 model not found) is still raised immediately with NO retry - that's
    a real configuration error, not a transient stall.

    Original 2026-07-22 incident: a real long run_eval.py pass over 70
    sequential local generations hit exactly one ReadTimeout around item 51,
    after ~25 minutes of continuous local inference (every other call that
    run succeeded in 10-50s) - most likely thermal throttling on sustained
    laptop inference, not a hung server. A short exponential backoff (2s,
    4s, ...) gives it a chance to recover instead of aborting the whole
    batch on one transient blip.

    Raises OllamaNotReachable with a clear, actionable message once retries
    are exhausted - callers decide whether to catch it (e.g. rag_pipeline.py
    lets it propagate as a clear error; a future router might catch it to
    fall back to a "LLM unavailable, here's the raw retrieved context"
    response).
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
        # 2026-09-13: qwen3:8b is a reasoning ("thinking") model - by
        # default it generates an extended chain-of-thought trace (Ollama's
        # response includes this separately in the "thinking" field) BEFORE
        # the actual answer, even for trivial prompts. Diagnosed live: a
        # bare "2+2=?" generated 581 tokens (a full paragraph of internal
        # reasoning) and took 147s end-to-end (load_duration was only
        # 0.16s - the model was already warm; nearly all 147s was
        # eval_duration, i.e. token generation) - this alone explains why
        # every LLM-backed branch in the 2026-09-12/13 audit run (rag_core16,
        # absa, nhanh_b, qualitative, bilingual_en) hit the 180s x 3 retry
        # timeout: a multi-call branch like ABSA (5 aspects) or nhanh_b
        # (multi-agent Text-to-SQL) compounds this per call, easily exceeding
        # 180s even before any real slowness/overload. Ollama's
        # /api/generate officially supports a "think" field to disable this
        # (https://docs.ollama.com/capabilities/thinking) - set False here
        # since this project's prompts want a fast, literal, grounded answer
        # (temperature=0.2 above reflects the same "no creative wandering"
        # intent), not an exposed reasoning trace. Pass think=True at the
        # call site for the rare case where the extra reasoning step is
        # actually wanted.
        "think": think,
    }
    if system:
        payload["system"] = system

    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
        except requests.exceptions.HTTPError as e:
            # 2026-09-08: a real live run hit "500 Server Error" from Ollama
            # itself mid-batch (after several 90-120s generations - same
            # "sustained local inference stresses the process" pattern as
            # the ReadTimeout incident documented above, just manifesting as
            # a 500 instead of a hung connection this time). A 5xx means
            # Ollama's OWN process choked on this request (crash/OOM/model
            # reload) - plausibly transient, same retry/backoff as
            # Timeout/ConnectionError above. A 4xx (e.g. 404 model not
            # found) is a real, non-transient configuration error - still
            # raised immediately, no retry, same as before this change.
            status = e.response.status_code if e.response is not None else None
            if status is not None and 500 <= status < 600:
                last_error = e
                if attempt < retries:
                    time.sleep(2 ** (attempt + 1))
                    continue
            raise OllamaNotReachable(
                f"Could not reach Ollama at {OLLAMA_HOST} (model={OLLAMA_MODEL}): {e}. "
                f"Is Ollama running (`ollama serve`) and is the model pulled "
                f"(`ollama pull {OLLAMA_MODEL}`)?"
            ) from e
        except requests.exceptions.RequestException as e:
            # any other real error (DNS failure, malformed URL, etc.) - not
            # transient, don't retry
            raise OllamaNotReachable(
                f"Could not reach Ollama at {OLLAMA_HOST} (model={OLLAMA_MODEL}): {e}. "
                f"Is Ollama running (`ollama serve`) and is the model pulled "
                f"(`ollama pull {OLLAMA_MODEL}`)?"
            ) from e

    raise OllamaNotReachable(
        f"Could not reach Ollama at {OLLAMA_HOST} (model={OLLAMA_MODEL}) after "
        f"{retries + 1} attempt(s): {last_error}. Is Ollama running (`ollama serve`) "
        f"and is the model pulled (`ollama pull {OLLAMA_MODEL}`)?"
    ) from last_error


def generate(prompt: str, system: str = None, temperature: float = 0.2,
             timeout: int = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES,
             think: bool = False) -> str:
    """Single-turn generation via Ollama's /api/generate endpoint (stream=False -
    simpler to work with than the streaming default, fine for this project's
    scale - not serving concurrent users). temperature=0.2 default (not 0.7+)
    because financial QA wants consistent, literal answers grounded in the
    retrieved context, not creative variation.

    think=False by default (2026-09-13) - see _generate_raw()'s docstring/
    comment for the diagnosed root cause: qwen3:8b's default "thinking" mode
    was directly responsible for the 12 OllamaNotReachable timeouts in the
    2026-09-12/13 audit run (a bare "2+2=?" took 147s / 581 tokens with
    thinking on). Pass think=True at a specific call site if that call
    genuinely benefits from an exposed reasoning trace.

    Unchanged public behavior after the 2026-09-08 _generate_raw() refactor -
    see that function's docstring for the retry/backoff rationale."""
    return _generate_raw(prompt, system, temperature, timeout, retries, think)["response"].strip()


def generate_with_stats(prompt: str, system: str = None, temperature: float = 0.2,
                         timeout: int = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES,
                         think: bool = False) -> dict:
    """Same call as generate(), but returns real Ollama-reported token counts
    and phase durations instead of just the answer text - added 2026-09-08
    for the VI/EN latency benchmark (the user's thesis defense is in English,
    the corpus/system prompts are Vietnamese-first, so "how many tokens does
    an English question cost vs the same question in Vietnamese" needed a
    real measurement, not a char-count proxy like callbacks.timed_generation
    uses elsewhere in this codebase).

    Returns:
        response            str   - the generated text (already .strip()'d)
        prompt_tokens       int|None - Ollama's prompt_eval_count (input, incl. system+context)
        completion_tokens   int|None - Ollama's eval_count (output tokens generated)
        total_duration_s    float - wall time for the whole call (load+prompt+eval)
        prompt_eval_s       float - time spent processing the prompt
        eval_s              float - time spent generating the completion
        load_duration_s     float - time spent loading the model into memory (0 if already warm)

    All duration fields default to 0.0 if Ollama's response omits them (older
    Ollama versions) - None would force every caller to null-check before
    arithmetic; 0.0 is a safe "couldn't measure this phase" default here
    since these are report-quality diagnostics, not values fed back into any
    correctness check."""
    data = _generate_raw(prompt, system, temperature, timeout, retries, think)
    ns = 1_000_000_000
    return {
        "response": data["response"].strip(),
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
        "total_duration_s": data.get("total_duration", 0) / ns,
        "prompt_eval_s": data.get("prompt_eval_duration", 0) / ns,
        "eval_s": data.get("eval_duration", 0) / ns,
        "load_duration_s": data.get("load_duration", 0) / ns,
    }


if __name__ == "__main__":
    if check_ollama():
        print("\n--- test generation ---")
        print(generate("Trả lời ngắn gọn bằng tiếng Việt: 2+2 bằng mấy?"))
