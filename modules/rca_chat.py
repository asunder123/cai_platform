
###############################################################
# modules/rca_chat.py — Contextual Paraphrasing Engine (v2.1)
###############################################################

from typing import Dict, List, Tuple
import re
from collections import Counter

def contextual_paraphrase(
    text: str,
    signals: Dict[str, float],
    root_causes: List[str],
    archetypes: List[str],
    *,
    max_quote_len: int = 160,
    include_sections: bool = True,
    return_dict: bool = False,
    # --- NEW keyword parameters ---
    max_keywords: int = 6,
    min_token_len: int = 3,
    boost_terms: Tuple[str, ...] = (
        # domain-anchoring boosters
        "handover", "workflow", "runbook", "cab", "version", "spec",
        "jira", "xray", "api", "schema", "client", "escalation",
        "sev", "severity", "rca", "traceability", "release", "cutover"
    ),
    decorate: bool = True,      # decorate echoed keywords (bold **term**)
    attach_in_synthesis: bool = True  # include echoed keywords in synthesis guidance
) -> str | Dict[str, str]:
    """
    Generates a contextual, paraphrased RCA summary tied to the input prompt.
    Now extracts keywords/keyphrases from the prompt and echoes them in the narrative.

    Returns: string summary or sectioned dict when return_dict=True.
    """

    # -------------------------
    # 0) Preprocess prompt
    # -------------------------
    clean_text = re.sub(r"\s+", " ", text).strip()
    quote = (clean_text[:max_quote_len] + ("…" if len(clean_text) > max_quote_len else "")) or "—"
    lower = clean_text.lower()

    # -------------------------
    # 1) Keyword & keyphrase extraction (no external libs)
    # -------------------------
    # Basic tokenization (keep alphanum & hyphen)
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]+", lower)

    # Minimal, extensible stopwords (add more if needed)
    stop = {
        "the","and","or","to","of","for","in","on","with","by","at","as","a","an","is","are",
        "was","were","it","this","that","those","these","from","into","over","under","be","been",
        "than","then","but","so","not","no","do","did","does","can","could","should","would",
        "we","you","they","he","she","i","our","their","his","her","your","my"
    }

    # Filter tokens: length and stopwords
    core_tokens = [t for t in tokens if len(t) >= min_token_len and t not in stop]

    # Build n-grams (bigrams + trigrams) from filtered tokens
    def ngrams(seq: List[str], n: int) -> List[str]:
        return [" ".join(seq[i:i+n]) for i in range(len(seq)-n+1)]

    bigrams = ngrams(core_tokens, 2)
    trigrams = ngrams(core_tokens, 3)

    # Frequency scores (favor trigrams, then bigrams, then unigrams)
    uni_counts = Counter(core_tokens)
    bi_counts = Counter(bigrams)
    tri_counts = Counter(trigrams)

    # Boost domain terms slightly if they appear
    def boosted_score(term: str, base: int) -> float:
        bonus = 0.4 if any(bt in term for bt in boost_terms) else 0.0
        return base + bonus

    scored = []
    for t, c in tri_counts.items():
        scored.append((t, boosted_score(t, c * 3)))        # trigrams weight 3
    for t, c in bi_counts.items():
        scored.append((t, boosted_score(t, c * 2)))        # bigrams weight 2
    for t, c in uni_counts.items():
        scored.append((t, boosted_score(t, c * 1)))        # unigrams weight 1

    # Sort by score desc, then length desc to prefer longer phrases
    scored.sort(key=lambda x: (-x[1], -len(x[0])))

    # Deduplicate while preserving order
    seen_k = set()
    top_keywords = []
    for term, _score in scored:
        if term not in seen_k:
            seen_k.add(term)
            top_keywords.append(term)
        if len(top_keywords) >= max_keywords:
            break

    # Optional decoration for visible anchoring
    def decorate_kw(kw: str) -> str:
        return f"**{kw}**" if decorate else kw

    anchored_keywords = [decorate_kw(k) for k in top_keywords]

    # -------------------------
    # 2) Theme extraction (same as v2, slightly expanded)
    # -------------------------
    lexicon = {
        "workflow disruption": [
            r"\bworkflow\b", r"\bhandover\b", r"\bgate\b", r"\bcutover\b",
            r"\bapproval\b", r"\bprocess\b", r"\bcab\b"
        ],
        "documentation drift": [
            r"\bdocument\b", r"\bversion\b", r"\bspec\b", r"\brunbook\b",
            r"\brelease notes\b", r"\bmetadata\b", r"\btraceability\b"
        ],
        "integration misalignment": [
            r"\bapi\b", r"\binterface\b", r"\bintegration\b", r"\bcontract\b",
            r"\bschema\b"
        ],
        "client pressure and rising escalation risk": [
            r"\bclient\b", r"\bescalation\b", r"\bcustomer\b", r"\bsev-?1\b",
            r"\bseverity\b", r"\bcomplaint\b"
        ],
        "ambiguous requirement interpretation": [
            r"\brequirement\b", r"\binterpret\b", r"\bacceptance criteria\b",
            r"\bscope\b", r"\bambigu(?:ous|ity)\b"
        ],
        "testing misalignment and validation blockage": [
            r"\bqa\b", r"\btesting\b", r"\bxray\b", r"\bvalidation\b",
            r"\bcoverage\b", r"\brepro(duce|ducible)\b"
        ],
    }

    themes: List[str] = []
    for theme, patterns in lexicon.items():
        if any(re.search(p, lower) for p in patterns):
            themes.append(theme)
    # Deduplicate
    seen_t = set()
    themes = [t for t in themes if not (t in seen_t or seen_t.add(t))]

    # -------------------------
    # 3) Signal ordering & phrasing
    # -------------------------
    ordered_signals: List[Tuple[str, float]] = sorted(
        ((k, float(v)) for k, v in signals.items()),
        key=lambda kv: (-kv[1], kv[0])
    )
    high = [k for k, v in ordered_signals if v > 0.65]
    moderate = [k for k, v in ordered_signals if 0.4 < v <= 0.65]
    low = [k for k, v in ordered_signals if v <= 0.4]

    def phrase_signals():
        parts = []
        if high:
            parts.append(f"Strong indicators: {', '.join(high)}.")
        if moderate:
            parts.append(f"Moderate indicators: {', '.join(moderate)}.")
        if not high and not moderate and low:
            parts.append(f"Weak indicators present: {', '.join(low)}.")
        return " ".join(parts)

    indicators_text = phrase_signals()

    # -------------------------
    # 4) Causes & archetypes phrasing
    # -------------------------
    def join_list(label: str, items: List[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return f"{label}: {items[0]}."
        return f"{label}: " + ", ".join(items[:-1]) + f", and {items[-1]}."

    causes_text = join_list("Likely underlying factors", root_causes)
    archetypes_text = join_list("Pattern mirrors archetype(s)", archetypes)

    # -------------------------
    # 5) Thematic narrative (with keyword anchor)
    # -------------------------
    if themes:
        theme_part = "This scenario reflects " + ", ".join(themes) + "."
    else:
        theme_part = "This scenario reflects general delivery and coordination concerns."

    if anchored_keywords:
        theme_part += f" Key prompt terms: {', '.join(anchored_keywords)}."

    # -------------------------
    # 6) Synthesis with prompt tie-back and keyword echo
    # -------------------------
    synthesis_core = (
        "Overall, the dynamics point to fragmented situational awareness, "
        "misaligned expectations, and challenges in cross-team coherence."
    )

    if "ESCALATION_RISK" in signals and signals["ESCALATION_RISK"] > 0.65:
        synthesis_core += " Given elevated escalation risk, prioritize ownership clarity, unified communication, and audit-ready evidence."
    if "EVIDENCE_CONFIDENCE" in signals and signals["EVIDENCE_CONFIDENCE"] <= 0.4:
        synthesis_core += " Close evidence gaps by attaching logs, timestamps, validation results, and traceable approvals."

    if attach_in_synthesis and anchored_keywords:
        synthesis_core += " Actions should directly address: " + ", ".join(anchored_keywords) + "."

    # -------------------------
    # 7) Sections & output
    # -------------------------
    context_text = f"Prompt context: “{quote}”"
    if include_sections or return_dict:
        sections = {
            "context": context_text,
            "indicators": indicators_text or "No signals provided.",
            "causes": causes_text or "No root causes provided.",
            "archetypes": archetypes_text or "No archetypes provided.",
            "synthesis": f"{theme_part} {synthesis_core}".strip()
        }
        if return_dict:
            return sections
        summary = (
            f"{sections['context']}\n\n"
            f"{sections['indicators']}\n"
            f"{sections['causes']}\n"
            f"{sections['archetypes']}\n\n"
            f"{sections['synthesis']}"
        )
    else:
        summary = (
            f"{context_text} {indicators_text} {causes_text} {archetypes_text} "
            f"{theme_part} {synthesis_core}"
        ).strip()

    return summary