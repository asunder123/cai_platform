###############################################################
# train_intelligence.py
# Trains a corporate anomaly signal classifier using spaCy v3
###############################################################

import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
import json
from pathlib import Path

TRAIN_FILE = Path("training_data.jsonl")
OUTPUT_DIR = Path("model_corporate_signals")


def load_training_data(path):
    """Loads .jsonl training cases into spaCy Examples."""
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat_multilabel", last=True)

    labels = [
        "WORKFLOW_DEVIATION",
        "EVIDENCE_CONFIDENCE",
        "ESCALATION_RISK",
        "DOCUMENTATION_INTEGRITY",
        "INTERPRETATION_BIAS",
        "CONFLICTING_INPUTS"
    ]

    for label in labels:
        textcat.add_label(label)

    examples = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            entry = json.loads(line)
            doc = nlp.make_doc(entry["text"])
            cats = entry["cats"]
            examples.append(Example.from_dict(doc, {"cats": cats}))

    return nlp, textcat, examples


def train():
    if not TRAIN_FILE.exists():
        print("❌ No training_data.jsonl found!")
        return

    print("📘 Loading training data...")
    nlp, textcat, examples = load_training_data(TRAIN_FILE)

    print("🚀 Starting training...")
    optimizer = nlp.initialize()

    for epoch in range(10):
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
        print(f"Epoch {epoch+1}/10 — Loss: {losses}")

    print("💾 Saving model to:", OUTPUT_DIR)
    nlp.to_disk(OUTPUT_DIR)
    print("✅ Training complete!")


if __name__ == "__main__":
    train()
