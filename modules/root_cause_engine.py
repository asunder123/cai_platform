
def infer_root_causes(signals: dict, prompt_keywords: list = None) -> list:
    """
    Infers likely root causes based on signal strengths and optional prompt keywords.
    Adds contextual phrasing and severity-based logic.
    
    Parameters
    ----------
    signals : dict
        Dictionary of signal scores (0.0–1.0) keyed by category name.
    prompt_keywords : list, optional
        Extracted keywords from the original prompt for contextual anchoring.
    
    Returns
    -------
    list of str
        Root cause hypotheses or a stability message if no major anomalies detected.
    """

    rc = []

    # Documentation integrity issues
    if signals.get("DOCUMENTATION_INTEGRITY", 0) < 0.4:
        rc.append("Critical documentation gaps undermining traceability")
    elif 0.4 <= signals.get("DOCUMENTATION_INTEGRITY", 0) <= 0.6:
        rc.append("Moderate documentation inconsistencies affecting clarity")

    # Conflicting inputs
    if signals.get("CONFLICTING_INPUTS", 0) > 0.7:
        rc.append("Severe contradictions in stakeholder inputs")
    elif 0.5 <= signals.get("CONFLICTING_INPUTS", 0) <= 0.7:
        rc.append("Partial misalignment among contributors")

    # Escalation risk
    if signals.get("ESCALATION_RISK", 0) > 0.7:
        rc.append("High escalation pressure influencing decision-making")
    elif 0.5 <= signals.get("ESCALATION_RISK", 0) <= 0.7:
        rc.append("Emerging escalation signals requiring proactive mitigation")

    # Optional keyword anchoring for context
    if prompt_keywords:
        rc.append(f"Contextual factors linked to: {', '.join(prompt_keywords[:3])}")

    return rc or ["Stable conditions; minimal anomalies detected"]
