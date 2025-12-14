###############################################################
# augment_corporate_cases.py
# Generates paraphrased & noisy training variants
###############################################################

import json
from pathlib import Path
import random

INPUT = Path("training_data.jsonl")
OUTPUT = Path("training_data_augmented.jsonl")

PARAPHRASE_MAP = {
    "delay": ["pushback", "slippage", "postponement"],
    "missing": ["not attached", "unavailable", "absent"],
    "conflict": ["disagreement", "misalignment", "clash"],
    "urgent": ["high priority", "time-sensitive", "critical"],
}

def paraphrase(text):
    for k, alternatives in PARAPHRASE_MAP.items():
        if k in text:
            text = text.replace(k, random.choice(alternatives))
    return text

def augment():
    if not INPUT.exists():
        print("❌ No training_data.jsonl found!")
        return

    with open(INPUT, "r", encoding="utf8") as fin, \
         open(OUTPUT, "w", encoding="utf8") as fout:

        for line in fin:
            entry = json.loads(line)
            fout.write(json.dumps(entry) + "\n")

            # add augmentation
            augmented = entry.copy()
            augmented["text"] = paraphrase(entry["text"].lower())
            fout.write(json.dumps(augmented) + "\n")

    print("✨ Augmented dataset written to:", OUTPUT)


if __name__ == "__main__":
    augment()
