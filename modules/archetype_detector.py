ARCHETYPES = {
    "Documentation Drift": [
        "version conflict", "mismatch", "missing", "metadata", "inconsistent document"
    ],
    "Ownership Misalignment": [
        "unclear ownership", "handover", "responsible", "ownership gap"
    ],
    "Ambiguity Spiral": [
        "contradiction", "fragmented", "ambiguous", "multiple interpretations"
    ],
    "Scope Mutation": [
        "scope change", "rework", "requirements change", "scope shift"
    ],
    "Escalation Loop": [
        "escalation", "critical", "pressure", "urgent"
    ]
}


def detect_archetypes(text):
    t = text.lower()
    matched = []

    for name, triggers in ARCHETYPES.items():
        for term in triggers:
            if term in t:
                matched.append(name)
                break

    return matched or ["No dominant archetypal pattern detected."]
