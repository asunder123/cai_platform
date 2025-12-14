def predict_risks(signals):

    risk_score = (
        signals["WORKFLOW_DEVIATION"] * 0.3 +
        signals["ESCALATION_RISK"] * 0.4 +
        signals["CONFLICTING_INPUTS"] * 0.3
    )

    return {
        "schedule_risk": round(min(1.0, risk_score), 2),
        "client_risk": round(min(1.0, risk_score * 1.1), 2),
        "documentation_risk": round(1 - signals["DOCUMENTATION_INTEGRITY"], 2)
    }
