###############################################################
# validate_training_dataset.py
# Ensures labels, text, and structure are correct in JSONL
###############################################################

import json
from pathlib import Path

FILE = Path("training_data.jsonl")

REQUIRED_KEYS = [
    "WORKFLOW_DEVIATION",
    "EVIDENCE_CONFIDENCE",
    "ESCALATION_RISK",
    "DOCUMENTATION_INTEGRITY",
    "INTERPRETATION_BIAS",
    "CONFLICTING_INPUTS"
]


def validate():
    if not FILE.exists():
        print("❌ training_data.jsonl missing")
        return

    print("🔍 Validating training data…")

    errors = 0

    with open(FILE, "r", encoding="utf8") as f:
        for i, line in enumerate(f, 1):
            try:
                item = json.loads(line)

                if "text" not in item:
                    print(f"❌ Missing text at line {i}")
                    errors += 1

                if "cats" not in item:
                    print(f"❌ Missing cats at line {i}")
                    errors += 1
                    continue

                for k in REQUIRED_KEYS:
                    if k not in item["cats"]:
                        print(f"❌ Missing label {k} at line {i}")
                        errors += 1

            except Exception as e:
                print(f"❌ JSON parse error at line {i}:", e)
                errors += 1

    if errors == 0:
        print("✅ Dataset is valid!")
    else:
        print(f"⚠️ Completed with {errors} errors.")


if __name__ == "__main__":
    validate()
