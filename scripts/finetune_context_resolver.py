"""LoRA fine-tuning for the RQ7 Context Resolver (src/dialogue/context_resolver.py's
FineTunedContextResolver). See docs/Ghi_chu_thao_luan_phan_bien_2026-07-26.md
section 8.10 for the design decision this implements (chốt 2026-07-28):
mT5-small (300M, default) or mT5-base (580M, stretch goal) + LoRA via
HuggingFace `peft` (rank r=8/16, alpha=16/32, dropout=0.05, applied to the
encoder-decoder's Q/V attention matrices only, ~0.5-1% of params updated).

CANNOT be run inside the project sandbox: downloading google/mt5-small
needs network access the sandbox doesn't have, and training needs a real
GPU/MPS device the sandbox doesn't have either. Run this on your own
machine once data/eval/rq7_conversational_benchmark.jsonl exists (built by
scripts/build_rq7_dataset.py, which HAS already been run and committed - see
that file).

One-time setup (on your machine, NOT in this sandbox):
    pip install transformers peft torch sentencepiece accelerate

Run:
    python3 scripts/finetune_context_resolver.py
    python3 scripts/finetune_context_resolver.py --model google/mt5-base --epochs 5

Output: a checkpoint directory at models/context_resolver_lora/ (adapter
weights + tokenizer), loadable directly by
src/dialogue/context_resolver.py's FineTunedContextResolver.
"""
import argparse
import json
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "rq7_conversational_benchmark.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "context_resolver_lora"


def load_training_pairs(dataset_path: Path = DATASET_PATH) -> list[dict]:
    """Turns each conversation into one (input_text, target_text) training
    pair: input = history turns joined + the final (possibly reduced)
    utterance; target = the real gold_standalone_question. Pure function,
    no torch/transformers dependency - independently unit-testable (see
    tests/test_dialogue/test_finetune_context_resolver.py)."""
    pairs = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            turns = conv["turns"]
            history_text = " | ".join(t["utterance"] for t in turns[:-1])
            current = turns[-1]["utterance"]
            input_text = f"resolve: {history_text} </s> {current}" if history_text else f"resolve: {current}"
            pairs.append({
                "conv_id": conv["conv_id"],
                "error_category": conv["error_category"],
                "input_text": input_text,
                "target_text": conv["gold_standalone_question"],
            })
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/mt5-small",
                         help="Base model - google/mt5-small (default) or google/mt5-base (stretch goal)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=10,
                         help="Small dataset (64 conversations as of 2026-07-28) needs more epochs than a "
                              "large-corpus fine-tune would - watch validation loss for overfitting.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoTokenizer, MT5ForConditionalGeneration,
                                   Trainer, TrainingArguments)
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}. Run on your own machine (not the sandbox) after "
            f"`pip install transformers peft torch sentencepiece accelerate`."
        )

    pairs = load_training_pairs()
    if len(pairs) < 10:
        raise SystemExit(
            f"Only {len(pairs)} training pairs found in {DATASET_PATH} - re-run "
            f"scripts/build_rq7_dataset.py first, or check it produced items."
        )
    print(f"Loaded {len(pairs)} training pairs from {DATASET_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = MT5ForConditionalGeneration.from_pretrained(args.model)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q", "v"],  # Q/V attention matrices only, per the 2026-07-28 design decision
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    def _tokenize(batch):
        model_inputs = tokenizer(batch["input_text"], max_length=256, truncation=True, padding="max_length")
        labels = tokenizer(text_target=batch["target_text"], max_length=64, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    from datasets import Dataset
    ds = Dataset.from_list(pairs).map(_tokenize, batched=True)
    split = ds.train_test_split(test_size=0.15, seed=42)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "_training_logs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args,
                       train_dataset=split["train"], eval_dataset=split["test"])
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
