
"""
Minimal RCA Chat Module
Provides contextual paraphrase and normalization utilities for RCA assistant.
"""

from typing import Any, Dict, List, Sequence, Optional
import json

# ---------------------------------------------------------------------
# Contextual paraphrase logic
# ---------------------------------------------------------------------
def contextual_paraphrase(
    canonical_text: str,
    merged_signals: Dict[str, Any],
    inferred_rca: Dict[str, Any],
    archetypes: Sequence[str],
    *,
    facets: Optional[Dict[str, Any]] = None,
    evidence: Optional[Sequence[Dict[str, Any]]] = None,
    return_dict: bool = True,
) -> Dict[str, Any]:
    """
    Generates a richer paraphrase using available context:
    - canonical_text: user query interpreted by the system
    - facets: environment, service, endpoint, region, etc.
    - merged_signals: top signals, stats, notes
    - inferred_rca: root cause hints
    - archetypes: known RCA patterns
    - evidence: optional supporting data
    """
    env = facets.get("environment") or "unknown"
    svc = facets.get("service") or "unspecified service"
    region = facets.get("region") or "unknown region"
    endpoint = facets.get("endpoint")
    time_window = facets.get("time_window") or "last_1h"

    # Build header
    header = f"Contextual Analysis for {svc} in {env} ({region})"
    if endpoint:
        header += f" – endpoint {endpoint}"
    header += f" – window {time_window}"

    # Signals and RCA hints
    signals_text = ", ".join(merged_signals.get("top", [])) or "No dominant signals detected"
    rca_text = ", ".join(map(str, inferred_rca.get("root_causes", []))) or "No RCA hints available"
    archetype_text = ", ".join(archetypes) if archetypes else "None"

    # Evidence summary
    evidence_summary = f"{len(evidence)} evidence items" if evidence else "No evidence provided"

    synthesis = (
        f"{header}\n\n"
        f"Intent: {canonical_text}\n"
        f"Signals observed: {signals_text}\n"
        f"Root cause hints: {rca_text}\n"
        f"Archetypes: {archetype_text}\n"
        f"Evidence summary: {evidence_summary}\n"
    )

    # Suggested next steps
    next_steps = [
        "Validate RCA hints against recent deployments",
        "Compare latency/error metrics for anomalies",
        "Review logs for correlated failures",
    ]

    return {
        "synthesis": synthesis,
        "sections": {
            "problem": header,
            "signals": signals_text,
            "rca": rca_text,
            "risk": "Potential service degradation if unresolved",
            "tests": "\n".join(f"- {step}" for step in next_steps),
            "actions": "\n".join([
                f"Attach logs for {svc}",
                f"List deployment changes in {env}",
                f"Provide p95/p99 latency comparison"
            ]),
        },
        "followups": [
            "Which environment is most impacted?",
            "Any recent config changes?",
            "Do we have correlated alerts?"
        ],
        "next_prompts": [
            "Show error logs",
            "Compare p95 latency",
            "List recent deployments"
        ],
        "state": {
            "facets": facets,
            "top_signals": merged_signals.get("top", []),
            "evidence_ids": [],
            "confidence": 0.7
        }
    }

# ---------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------
def normalize_rca_dict(rca: Any) -> Dict[str, Any]:
    if rca is None:
        return {"root_causes": [], "confidence_overall": None, "next_best_tests": []}
    if isinstance(rca, list):
        return {"root_causes": rca, "confidence_overall": None, "next_best_tests": []}
    if isinstance(rca, dict):
        return {
            "root_causes": rca.get("root_causes", []),
            "confidence_overall": rca.get("confidence_overall"),
            "next_best_tests": rca.get("next_best_tests", []),
        }
    return {"root_causes": [], "confidence_overall": None, "next_best_tests": []}

def normalize_archetypes_list(arch: Any) -> List[str]:
    if arch is None:
        return []
    if isinstance(arch, str):
        return [arch]
    try:
        return list(arch)
    except Exception:
        return []

def normalize_signals_dict(merged_signals: Any) -> Dict[str, Any]:
    if merged_signals is None:
        return {"top": [], "stats": {}, "melt": [], "notes": None}
    if isinstance(merged_signals, dict):
        return {
            "top": merged_signals.get("top", []),
            "stats": merged_signals.get("stats", {}),
            "melt": merged_signals.get("melt", []),
            "notes": merged_signals.get("notes"),
        }
    return {"top": [], "stats": {}, "melt": [], "notes": None}

# ---------------------------------------------------------------------
# Semantic tokenization (basic)
# ---------------------------------------------------------------------
def semantic_tokens(text: str) -> List[str]:
    """
    Tokenize text for semantic matching.
    Retains '/', '-', '_' for endpoints and versions.
    """
    import re
    stopwords = {"the", "a", "an", "and", "or", "to", "of", "for", "on", "in", "at", "by", "with"}
    s = re.sub(r"[^a-zA-Z0-9/_\-. ]+", " ", (text or "")).lower().strip()
    toks = [t for t in re.split(r"\s+", s) if t and t not in stopwords]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
