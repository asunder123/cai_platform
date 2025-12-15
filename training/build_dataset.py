
# ------------------------------------------------------------------
# Synthesis helpers
# ------------------------------------------------------------------
def make_entry(text: str, cats: dict) -> dict:
    return {"text": text, "cats": cats}

def cats_all_zero() -> dict:
    return {lbl: 0 for lbl in LABELS}

def cats_one_hot(lbl: str) -> dict:
    d = cats_all_zero()
    d[lbl] = 1
    return d

def cats_multi(*labels: str) -> dict:
    d = cats_all_zero()
    for l in labels:
        d[l] = 1
    return d

# ------------------------------------------------------------------
# Build dataset: positives, hard negatives, combos
# ------------------------------------------------------------------
examples = []

# 1) Positives: 10 per label × 6 = 60
for lbl, texts in POS.items():
    for t in texts:
        examples.append(make_entry(t, cats_one_hot(lbl)))

# 2) Hard negatives: 10 items (all zeros)
for t in NEUTRAL:
    examples.append(make_entry(t, cats_all_zero()))

# 3) Combos (multi‑label) – to teach co‑occurrence
COMBOS = [
    ("WORKFLOW_DEVIATION", "ESCALATION_RISK"),
    ("EVIDENCE_CONFIDENCE", "ESCALATION_RISK"),
    ("DOCUMENTATION_INTEGRITY", "EVIDENCE_CONFIDENCE"),
    ("INTERPRETATION_BIAS", "EVIDENCE_CONFIDENCE"),
    ("CONFLICTING_INPUTS", "ESCALATION_RISK"),
    ("WORKFLOW_DEVIATION", "DOCUMENTATION_INTEGRITY"),
    ("CONFLICTING_INPUTS", "DOCUMENTATION_INTEGRITY"),
    ("INTERPRETATION_BIAS", "ESCALATION_RISK"),
    ("WORKFLOW_DEVIATION", "EVIDENCE_CONFIDENCE", "ESCALATION_RISK"),
    ("CONFLICTING_INPUTS", "INTERPRETATION_BIAS"),
]

COMBO_TEXTS = [
    "Urgent deployment skipped approvals; conflicting team updates raised escalation.",
    "Evidence pack lacked artifacts while escalation was opened on missed deliverables.",
    "API docs were incomplete; validation proofs missing across the review.",
    "Analysts drew conclusions without raw metrics; confidence remains low.",
    "Two reports disagreed on capacity; client initiated escalation protocol.",
    "Change advanced out of sequence; configuration documents were contradictory.",
    "Stakeholder timelines conflicted; policy references were out of date.",
    "Narrative overstated results; escalation was considered by sponsors.",
    "Deployment bypassed gates; no validation logs attached and escalation filed.",
    "Audit findings diverged; interpretation leaned on partial data.",
]

for i, combo in enumerate(COMBOS):
    # ✅ Use the proper text for the combo, not "COMBOS and ..."
    examples.append(make_entry(COMBO_TEXTS[i], cats_multi(*combo)))

# 4) Augmentations (paraphrases with token variation) – create >600
systems = ["SAP", "Salesforce", "Kafka", "Snowflake", "Databricks", "Jenkins", "GitHub", "ArgoCD"]
verbs = ["deployed", "rolled out", "promoted", "applied", "executed", "enabled"]
gov = ["CAB", "Change Advisory Board", "governance gate", "approval chain"]
proofs = ["logs", "screenshots", "run IDs", "metrics", "dashboards", "artifacts"]
risks = ["SLA breach", "missed milestone", "open blocker", "contract risk", "high‑severity defect"]
conflicts = ["reports", "dashboards", "audits", "timelines", "versions", "alerts"]

# generate ~100 per label via templating (6 × 100 = 600)
for lbl, base_texts in POS.items():
    for _ in range(100):
        sys = random.choice(systems)
        v = random.choice(verbs)
        g = random.choice(gov)
        p = random.choice(proofs)
        r = random.choice(risks)
        c = random.choice(conflicts)

        if lbl == "WORKFLOW_DEVIATION":
            txt = f"{sys} change was {v} without {g}, causing out‑of‑sequence execution."
        elif lbl == "EVIDENCE_CONFIDENCE":
            txt = f"Validation cited success but attached no {p} or baseline data for {sys}."
        elif lbl == "ESCALATION_RISK":
            txt = f"Escalation risk increased after {r} related to {sys} remained unresolved."
        elif lbl == "DOCUMENTATION_INTEGRITY":
            txt = f"Documentation for {sys} omitted encryption steps and conflicted versions."
        elif lbl == "INTERPRETATION_BIAS":
            txt = f"Conclusion about {sys} performance relied on partial {p} and optimistic assumptions."
        elif lbl == "CONFLICTING_INPUTS":
            txt = f"{sys} {c} presented mismatched outcomes, leaving sign‑off unclear."
        else:
            txt = f"{sys} update proceeded as planned."

        examples.append(make_entry(txt, cats_one_hot(lbl)))

# Deduplicate and shuffle
def dedupe(jsonl: list) -> list:
    """Remove exact duplicates (by JSON string key) with consistent indentation."""
    seen = set()
    out = []
    for e in jsonl:
        key = json.dumps(e, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out

examples = dedupe(examples)
random.shuffle(examples)

# Write JSONL
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

# Summary
cnt = Counter()
for e in examples:
    for lbl, v in e["cats"].items():
        if v == 1:
            cnt[lbl] += 1

print(f"✅ Wrote {len(examples)} examples to {OUT_PATH}")
for lbl in LABELS:
    print(f"{lbl:24s} {cnt[lbl]:5d}")

