###############################################################
# build_jsonl_dataset.py
# Combines multiple JSONL files into one training dataset
###############################################################

import json
from pathlib import Path

INPUT_DIR = Path(".")
OUTPUT = Path("training_data_combined.jsonl")

def build():
    files = sorted(INPUT_DIR.glob("*.jsonl"))
    print("📦 JSONL files found:", files)

    with open(OUTPUT, "w", encoding="utf8") as fout:
        for fp in files:
            if fp.name == OUTPUT.name:
                continue
            print("Adding:", fp)
            for line in open(fp, "r", encoding="utf8"):
                fout.write(line)

    print("✨ Combined dataset written to:", OUTPUT)


if __name__ == "__main__":
    build()
