def extract_signals(text):
    text = text.lower()

    return {
        "WORKFLOW_DEVIATION": 1.0 if "delay" in text or "bottleneck" in text else 0.3,
        "EVIDENCE_CONFIDENCE": 0.2 if "no proof" in text or "uncertain" in text else 0.7,
        "ESCALATION_RISK": 0.9 if "escalate" in text or "urgent" in text else 0.4,
        "DOCUMENTATION_INTEGRITY": 0.3 if "missing document" in text else 0.8,
        "INTERPRETATION_BIAS": 0.7 if "assumption" in text or "interpretation" in text else 0.4,
        "CONFLICTING_INPUTS": 0.85 if "conflict" in text or "contradiction" in text else 0.3
    }
