"""Schema + validation + I/O helpers for `data/eval/absa_annotations.jsonl` -
the labeled ABSA dataset Task #6 of the 2026-07-29 architecture review asked
for ("absa_training_dataset"). See docs/PROGRESS.md for the full rationale.

WHY a JSONL file, not a DB table: src/absa/absa_pipeline.py's own module
docstring documents a LOCKED design decision (Thesis_Direction_Redesign.pdf
section 9) that RQ5 uses zero-shot LLM prompting specifically because "no
annotated corpus" exists, deferring PhoBERT+LoRA fine-tuning to future work.
This file is infrastructure FOR that future work, not a change to RQ5's
already-locked zero-shot scope. Following this project's own convention for
small, versionable labeled/eval corpora (benchmark.jsonl,
rq7_conversational_benchmark.jsonl) rather than the operational Postgres
tables (which hold large, continuously-growing ETL'd data) - a labeled
dataset a human annotates by hand is exactly that kind of small, git-
trackable artefact, not a live database table.

WHY no labels are pre-populated here: fabricating ABSA labels would defeat
the entire purpose of a ground-truth annotated corpus - if this project
could reliably auto-generate correct sentiment labels, RQ5 would not need to
exist as "zero-shot vs fine-tuned" question in the first place. This module
is PURELY the schema + validation + append/load infrastructure; real labels
must come from a human via scripts/absa_annotate_cli.py (or manual editing)
run on the user's own machine. As of 2026-07-29, data/eval/absa_annotations.jsonl
does not exist / is empty - that is correct and expected, not a bug.

Reuses src/absa/absa_pipeline.py's ASPECTS/VALID_SENTIMENTS taxonomy
directly (imported, not redefined) so a labeled example and a live RQ5
zero-shot prediction are always directly comparable - the whole point of
building this corpus is to eventually evaluate (or fine-tune) against
these exact same 5 aspects/5 sentiment values (now including "mixed", see
absa_pipeline.py's 2026-07-29 evening update), not a parallel taxonomy
that would need its own mapping layer.

**Always-5-aspects schema (revised 2026-07-29 evening, Task #12 from the
round-2 architecture review - docs/PROGRESS.md has the full writeup).**
This is a deliberate REVERSAL of this file's original design: earlier the
same day, `validate_annotation()` let a human label only the aspects a
chunk actually discussed, treating an omitted aspect as an implicit
"not covered by this annotation". The review correctly pointed out the
real cost of that: a model trained on this corpus would then see almost
no NEGATIVE `not_mentioned` examples for a given aspect (since annotators
naturally only write down what stood out), even though "this chunk does
not discuss governance at all" is exactly as real and informative a label
as "this chunk praises governance" - the corpus would be systematically
missing the single most common real-world case per aspect. Every record
now MUST carry all 5 ASPECTS keys exactly once, with `not_mentioned` used
as an explicit, first-class value rather than an omission - the annotator
CLI (scripts/absa_annotate_cli.py) was reworked alongside this file to
prompt for all 5 aspects per chunk instead of a subset-selection menu.

**New per-record metadata fields** (also Task #12): `annotator_id`,
`annotation_guideline_version`, `annotation_round`, `split`, `created_at`,
`chunk_hash` are now REQUIRED; `adjudicated_sentiment` (per-aspect,
optional) and `evidence_start`/`evidence_end` (per-aspect, optional
character offsets into `text`) round out the schema. See each field's
inline comment below for what it's for and why.

**llm_unreviewed rows are never valid eval data** (also Task #12, per the
review's explicit point): a `label_source="llm_unreviewed"` record is a
raw, un-reviewed LLM proposal - not ground truth - and validate_annotation()
now REFUSES it if `split` is "dev" or "test". Its only legitimate use is
`split="train"` as a pseudo-label / weak-supervision signal for a future
fine-tuning experiment, never as something a reported accuracy number is
computed against.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.absa.absa_pipeline import ASPECTS, VALID_SENTIMENTS

ANNOTATIONS_PATH = Path("data/eval/absa_annotations.jsonl")

LABEL_SOURCES = {
    "human",                 # a person read the text and assigned every one of the 5 aspects themselves
    "llm_assisted_reviewed", # an LLM (e.g. absa_pipeline.analyze_text()) proposed labels, a human reviewed/corrected EVERY aspect
    "llm_unreviewed",        # an LLM proposed labels, NOT yet human-reviewed - NOT valid ground truth, see module
                             # docstring: split MUST be "train" (weak supervision only), never "dev"/"test".
}

# Same "dev"/"test" naming as scripts/split_router_benchmark.py's router
# split (router_development.jsonl / router_test.jsonl), plus "train" for
# rows meant to feed a future fine-tuning experiment rather than be
# reported as an eval number at all - one shared split vocabulary across
# both labeled corpora in this project, not two parallel conventions.
VALID_SPLITS = {"train", "dev", "test"}

REQUIRED_FIELDS = {
    "id", "ticker", "text", "aspects", "label_source",
    "annotator_id",                # who (or "llm" for llm_unreviewed) produced this record - free-text, not a curated user table
    "annotation_guideline_version", # e.g. "v1" - bumped whenever the labeling instructions/taxonomy change, so a
                                     # later reader can tell which guideline version a given row was labeled under
    "annotation_round",             # int, starting at 1 - bumped on a deliberate RE-annotation pass (e.g. adjudication)
    "split",                         # one of VALID_SPLITS - see llm_unreviewed restriction above
    "created_at",                    # ISO-8601 UTC timestamp string, when this record was written
    "chunk_hash",                    # SHA-256 of `text` (see compute_chunk_hash()) - lineage/dedup, same "detect drift
                                       # without silently trusting stale content" purpose as predictive_features.feature_hash
}


def compute_chunk_hash(text: str) -> str:
    """SHA-256 hex digest of the annotated text - lets a later reader
    confirm a labeled record still matches the disclosure_chunks row it
    was labeled from (chunk text is immutable per CHUNKING_VERSION, but a
    hash check costs nothing and catches any accidental mismatch)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    """UTC ISO-8601 timestamp, used to populate created_at - factored out
    so tests can monkeypatch a fixed value instead of asserting on
    wall-clock time."""
    return datetime.now(timezone.utc).isoformat()


def validate_annotation(record: dict) -> list:
    """Returns a list of problem strings (empty = valid) - same "report,
    don't guess-fix" convention as absa_pipeline.py's _validate_result().

    UNLIKE this file's original version, this DOES now require all 5
    ASPECTS keys to be present, each exactly once - see this module's
    docstring for why the earlier "only label what's discussed" design was
    reversed. An annotator who genuinely sees no signal for an aspect must
    write sentiment="not_mentioned" for it explicitly, not omit it."""
    problems = []
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        problems.append(f"Thiếu trường bắt buộc: {sorted(missing)}")
        return problems  # other checks below assume these exist

    if not isinstance(record["text"], str) or not record["text"].strip():
        problems.append("'text' phải là chuỗi khác rỗng.")

    if record["label_source"] not in LABEL_SOURCES:
        problems.append(f"'label_source' không hợp lệ: '{record['label_source']}' - phải là 1 trong {sorted(LABEL_SOURCES)}")

    if record["split"] not in VALID_SPLITS:
        problems.append(f"'split' không hợp lệ: '{record['split']}' - phải là 1 trong {sorted(VALID_SPLITS)}")
    elif record["label_source"] == "llm_unreviewed" and record["split"] != "train":
        problems.append(
            f"label_source='llm_unreviewed' nhưng split='{record['split']}' - nhãn LLM CHƯA được người review "
            f"KHÔNG được dùng làm dev/test (không phải ground truth thật), chỉ được split='train' "
            f"(pseudo-label/weak supervision)."
        )

    if not isinstance(record.get("annotation_round"), int) or record["annotation_round"] < 1:
        problems.append(f"'annotation_round' phải là số nguyên >= 1, nhận được: {record.get('annotation_round')!r}")

    if record.get("text") and record.get("chunk_hash") != compute_chunk_hash(record["text"]):
        problems.append("'chunk_hash' không khớp với SHA-256 thực tế của 'text' (dùng compute_chunk_hash()).")

    aspects = record["aspects"]
    if not isinstance(aspects, list):
        problems.append("'aspects' phải là danh sách.")
        return problems

    seen_aspects = set()
    for i, a in enumerate(aspects):
        if not isinstance(a, dict):
            problems.append(f"aspects[{i}] không phải object.")
            continue
        aspect_name = a.get("aspect")
        sentiment = a.get("sentiment")
        if aspect_name not in ASPECTS:
            problems.append(f"aspects[{i}]: khía cạnh không hợp lệ '{aspect_name}' - phải là 1 trong {sorted(ASPECTS)}")
        elif aspect_name in seen_aspects:
            problems.append(f"aspects[{i}]: khía cạnh '{aspect_name}' bị lặp lại trong cùng 1 dòng.")
        else:
            seen_aspects.add(aspect_name)
        if sentiment not in VALID_SENTIMENTS:
            problems.append(f"aspects[{i}]: sentiment không hợp lệ '{sentiment}' - phải là 1 trong {sorted(VALID_SENTIMENTS)}")
        if sentiment not in ("not_mentioned", None) and not a.get("evidence_span"):
            problems.append(f"aspects[{i}]: sentiment='{sentiment}' nhưng thiếu 'evidence_span' "
                             f"(bắt buộc trừ khi sentiment='not_mentioned').")
        adjudicated = a.get("adjudicated_sentiment")
        if adjudicated is not None and adjudicated not in VALID_SENTIMENTS:
            problems.append(f"aspects[{i}]: adjudicated_sentiment không hợp lệ '{adjudicated}' - "
                             f"phải là 1 trong {sorted(VALID_SENTIMENTS)} hoặc vắng mặt/null.")
        e_start, e_end = a.get("evidence_start"), a.get("evidence_end")
        if e_start is not None or e_end is not None:
            if not (isinstance(e_start, int) and isinstance(e_end, int) and 0 <= e_start < e_end <= len(record["text"])):
                problems.append(f"aspects[{i}]: evidence_start/evidence_end không hợp lệ "
                                 f"({e_start}, {e_end}) cho text dài {len(record['text'])} ký tự.")

    missing_aspects = set(ASPECTS) - seen_aspects
    if missing_aspects:
        problems.append(f"Thiếu khía cạnh bắt buộc (phải liệt kê đủ 5, kể cả not_mentioned): {sorted(missing_aspects)}")

    return problems


def load_annotations(path: Path = ANNOTATIONS_PATH) -> list:
    """Returns [] if the file doesn't exist yet (expected state as of
    2026-07-29 - see this module's docstring) rather than raising, so
    callers (e.g. the annotation CLI's resume logic) don't need their own
    file-existence check."""
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def already_annotated_keys(path: Path = ANNOTATIONS_PATH, annotator_id: str = None) -> set:
    """(ticker, source_file, page, chunk_index) keys already present in the
    annotation file - lets scripts/absa_annotate_cli.py skip chunks a human
    has already labeled on a prior, possibly-interrupted run, same
    resume-safety spirit as src/retrieval/vector_store.py's
    _existing_chunk_state(). Rows with no source_file/page/chunk_index
    (e.g. hand-written examples not tied to a real disclosure_chunks row)
    are skipped by key, not counted.

    annotator_id (added 2026-07-30, Task #22 - the ABSA pilot's inter-
    annotator agreement step needs TWO people to independently label the
    SAME chunks): when given, only counts a key as "already annotated" if
    THIS annotator_id specifically has a row for it - a key another
    annotator already labeled is NOT considered done for a different
    annotator_id. Without this, the second person to run
    scripts/absa_annotate_cli.py against the double-annotation subset would
    see every chunk the first person already labeled as "already done" and
    silently skip it, making double-annotation impossible by construction.
    Passing annotator_id=None (the default) preserves the original
    global-across-everyone behavior for any other caller that genuinely
    wants it."""
    keys = set()
    for r in load_annotations(path):
        if annotator_id is not None and r.get("annotator_id") != annotator_id:
            continue
        if r.get("source_file") is not None and r.get("page") is not None and r.get("chunk_index") is not None:
            keys.add((r.get("ticker"), r["source_file"], r["page"], r["chunk_index"]))
    return keys


def append_annotation(record: dict, path: Path = ANNOTATIONS_PATH) -> None:
    """Validates then appends ONE record as a new line - append-only, same
    crash-safety reasoning as every other incrementally-written JSONL file
    in this project (a labeling session interrupted mid-way keeps every
    row written so far, nothing is lost or needs re-doing). Raises
    ValueError (does NOT silently skip) if the record fails
    validate_annotation() - a human annotator should be told immediately
    their last entry was malformed, not find out much later when someone
    tries to load the file for training."""
    problems = validate_annotation(record)
    if problems:
        raise ValueError(f"Bản ghi annotation không hợp lệ, KHÔNG được ghi: {problems}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
