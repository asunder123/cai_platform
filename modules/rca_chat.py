###############################################################
# modules/rca_chat.py — Contextual Paraphrasing Engine
###############################################################

def contextual_paraphrase(text, signals, root_causes, archetypes):
    """
    Generates a contextual, paraphrased RCA summary.
    Driven by:
    - Themes detected from input text
    - Signal strengths (ML + rule-based hybrid)
    - Root cause indicators
    - Archetype matches
    """

    lower = text.lower()
    themes = []

    # Theme extraction from text
    if "workflow" in lower or "handover" in lower:
        themes.append("workflow disruption")
    if "document" in lower or "version" in lower or "spec" in lower:
        themes.append("documentation drift")
    if "api" in lower or "interface" in lower:
        themes.append("integration misalignment")
    if "client" in lower or "escalation" in lower:
        themes.append("client pressure and rising escalation risk")
    if "requirement" in lower or "interpret" in lower:
        themes.append("ambiguous requirement interpretation")
    if "qa" in lower or "testing" in lower:
        themes.append("testing misalignment and validation blockage")

    # Build narrative elements
    theme_part = ""
    if themes:
        theme_part = (
            "The scenario reflects " +
            ", ".join(themes) +
            ". "
        )

    high = [k for k, v in signals.items() if v > 0.65]
    moderate = [k for k, v in signals.items() if 0.4 < v <= 0.65]

    signal_part = ""
    if high:
        signal_part += (
            "Strong contributing indicators include " +
            ", ".join(high) + ". "
        )
    if moderate:
        signal_part += (
            "Additional moderate signals include " +
            ", ".join(moderate) + ". "
        )

    rc_part = ""
    if root_causes:
        rc_part = (
            "Underlying factors likely involve " +
            ", ".join(root_causes) +
            ". "
        )

    arch_part = ""
    if archetypes:
        arch_part = (
            "The pattern resembles the following archetype(s): " +
            ", ".join(archetypes) +
            ". "
        )

    # Final blend
    summary = (
        theme_part +
        signal_part +
        rc_part +
        arch_part +
        "Overall, these dynamics suggest fragmented situational awareness, "
        "misaligned expectations, and a breakdown of cross-team coherence."
    )

    return summary.strip()
