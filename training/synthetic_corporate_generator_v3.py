###############################################################
# synthetic_corporate_generator_v3.py
# Generates labeled synthetic corporate anomaly cases
###############################################################

import json
from random import choice, random
from pathlib import Path

OUTPUT = Path("synthetic_v3_corporate.jsonl")

WORKFLOW_ISSUES = [
    "The delivery sequence became blocked due to unclear upstream handover.",
    "Milestones shifted multiple times leading to a scheduling loop.",
    "A nonstandard approval flow introduced bottlenecks."
]

DOCUMENTATION_ISSUES = [
    "Multiple documents showed inconsistent metadata.",
    "Version conflict created confusion in requirement interpretation.",
    "Important evidence was missing from the attachment."
]

ESCALATIONS = [
    "Client initiated early escalation due to perceived delays.",
    "High pressure from leadership created urgency around decisions.",
    "Conflicting expectations triggered a governance escalation."
]

CONFLICTS = [
    "Stakeholders provided contradictory directions.",
    "Internal teams disagreed on interpretation of requirements.",
    "There was conflict over ownership of deliverables."
]


def generate_case():
    text = choice(WORKFLOW_ISSUES) + " " + \
           choice(DOCUMENTATION_ISSUES) + " " + \
           choice(ESCALATIONS) + " " + \
           choice(CONFLICTS)

    cats = {
        "WORKFLOW_DEVIATION": 1,
        "EVIDENCE_CONFIDENCE": 0 if "missing" in text else 1,
        "ESCALATION_RISK": 1,
        "DOCUMENTATION_INTEGRITY": 0 if "inconsistent" in text or "conflict" in text else 1,
        "INTERPRETATION_BIAS": 1 if "interpretation" in text else 0,
        "CONFLICTING_INPUTS": 1,
    }

    return {"text": text, "cats": cats}


def generate_dataset(n=200):
    print("💼 Generating synthetic corporate dataset…")

    with open(OUTPUT, "w", encoding="utf8") as f:
        for _ in range(n):
            json.dump(generate_case(), f)
            f.write("\n")

    print("📄 Dataset written to:", OUTPUT)


if __name__ == "__main__":
    generate_dataset(200)
