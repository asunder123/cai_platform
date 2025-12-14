def classify_process_buckets(text):
    text = text.lower()

    return {
        "Requirements": 0.8 if "scope" in text or "requirements" in text else 0.2,
        "Delivery": 0.7 if "delay" in text or "milestone" in text else 0.3,
        "Stakeholder": 0.9 if "client" in text or "conflict" in text else 0.2,
        "Governance": 0.6 if "escalation" in text or "approval" in text else 0.3
    }
