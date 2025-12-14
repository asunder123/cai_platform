def infer_root_causes(signals):
    rc = []

    if signals["DOCUMENTATION_INTEGRITY"] < 0.4:
        rc.append("Weak documentation integrity")
    if signals["CONFLICTING_INPUTS"] > 0.6:
        rc.append("Contradictory stakeholder inputs")
    if signals["ESCALATION_RISK"] > 0.6:
        rc.append("High escalation pressure influencing decisions")

    return rc or ["Stable root-causes; minimal anomalies"]
