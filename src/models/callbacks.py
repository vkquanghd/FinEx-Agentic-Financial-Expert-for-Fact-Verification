"""Lightweight generation logging - not a full observability stack (out of
scope), just enough to report latency numbers in the thesis's evaluation
section without hand-timing things manually later.

Usage:
    from src.models.callbacks import timed_generation
    text, elapsed_s = timed_generation(llm_factory.generate, prompt, system=sys_prompt)
"""
import time


def timed_generation(generate_fn, *args, **kwargs):
    """Wraps any generate_fn(...) -> str call, returns (result, elapsed_seconds).
    Prints a one-line log so long local-CPU generations don't look hung -
    same "make slow things visible instead of silent" pattern used in
    text_splitter.py's per-file read-time warnings."""
    start = time.monotonic()
    result = generate_fn(*args, **kwargs)
    elapsed = time.monotonic() - start
    print(f"[llm] generated {len(result)} chars in {elapsed:.1f}s "
          f"({len(result) / elapsed:.0f} chars/s)" if elapsed > 0 else "[llm] generated instantly")
    return result, elapsed
