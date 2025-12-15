
###############################################################
# train_intelligence.py
# Trains a corporate anomaly signal classifier using spaCy v3
###############################################################

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import spacy
from spacy.training.example import Example
from spacy.util import minibatch

# -----------------------
# Config
# -----------------------
TRAIN_FILE = Path("training_data.jsonl")
OUTPUT_DIR = Path("model_corporate_signals")
SEED = 42
EPOCHS = 10
BATCH_SIZE = 32

LABELS = [
    "WORKFLOW_DEVIATION",
    "EVIDENCE_CONFIDENCE",
    "ESCALATION_RISK",
    "DOCUMENTATION_INTEGRITY",
    "INTERPRETATION_BIAS",
    "CONFLICTING_INPUTS",
]


def _is_valid_entry(entry: Dict) -> bool:
    """Basic schema validation for a JSONL entry."""
    if not isinstance(entry, dict):
        return False
    if "text" not in entry or "cats" not in entry:
        return False
    if not isinstance(entry["text"], str):
        return False
    if not isinstance(entry["cats"], dict):
        return False
    # Ensure all expected labels exist and are numeric (0/1 floats or ints)
    cats = entry["cats"]
    for label in LABELS:
        if label not in cats:
            return False
        if not isinstance(cats[label], (int, float)):
            return False
    return True


def _dedupe_entries(entries: List[Dict]) -> List[Dict]:
    """
    Deduplicate entries in two ways:
    1) Exact JSON object duplicate removal (hashable via string dump).
    2) Optional: text-level dedupe (same 'text' + same 'cats').
    """
    seen_json = set()
    unique = []

    for e in entries:
        key = json.dumps(e, sort_keys=True, ensure_ascii=False)
        if key in seen_json:
            continue
        seen_json.add(key)
        unique.append(e)

    # Optional text-level dedupe:
    seen_text_cats = set()
    deduped = []
    for e in unique:
        text_key = (e["text"], tuple(sorted(e["cats"].items())))
        if text_key in seen_text_cats:
            continue
        seen_text_cats.add(text_key)
        deduped.append(e)

    return deduped


def load_training_data(path: Path) -> Tuple[spacy.Language, "spacy.pipeline.TextCategorizer", List[Example]]:
    """Loads .jsonl training cases into spaCy Examples with robust cleaning."""

    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat_multilabel", last=True)

    # Add labels
    for label in LABELS:
        textcat.add_label(label)

    # Read file robustly
    raw_entries: List[Dict] = []
    invalid_count = 0
    total_lines = 0

    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                invalid_count += 1
                continue
            if _is_valid_entry(entry):
                raw_entries.append(entry)
            else:
                invalid_count += 1

    # Deduplicate
    entries = _dedupe_entries(raw_entries)

    # Log stats
    print(f"📄 Read lines: {total_lines}")
    print(f"✅ Valid entries: {len(raw_entries)}")
    print(f"🧹 Invalid/Skipped: {invalid_count}")
    print(f"🔁 After dedupe: {len(entries)}")

    if len(entries) == 0:
        raise ValueError("No valid training examples after cleaning/deduplication.")

    # Build Examples
    examples: List[Example] = []
    for entry in entries:
        doc = nlp.make_doc(entry["text"])
        cats = entry["cats"]
        examples.append(Example.from_dict(doc, {"cats": cats}))

    # Shuffle for training
    random.Random(SEED).shuffle(examples)
    return nlp, textcat, examples


def train():
    if not TRAIN_FILE.exists():
        print("❌ No training_data.jsonl found!")
        return

    print("📘 Loading training data...")
    nlp, textcat, examples = load_training_data(TRAIN_FILE)

    # Initialize model
    print("🚀 Initializing model...")
    optimizer = nlp.initialize(get_examples=lambda: examples)

    # Train epochs with minibatching
    print("🧪 Starting training...")
    for epoch in range(1, EPOCHS + 1):
        losses = {}
        # Create shuffled minibatches each epoch
        random.Random(SEED + epoch).shuffle(examples)
        batches = minibatch(examples, size=BATCH_SIZE)
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)
        print(f"Epoch {epoch}/{EPOCHS} — Loss: {losses}")

    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("💾 Saving model to:", OUTPUT_DIR)
    nlp.to_disk(OUTPUT_DIR)
    print("✅ Training complete!")


if __name__ == "__main__":
    train()
