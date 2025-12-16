
# modules/conv_assistant.py
# =====================================================================
# Conversational Assistant for RCA – Pure FAISS Retrieval (Dynamic, Generic)
# - Scenario ingestion → chunking → vector indexing (FAISS/NumPy)
# - Query-time retrieval over scenario chunks + prior prompts
# - Dynamic facets from case+prompt (environment/service/endpoint/region/time_window)
# - Intent detection (compare/show/what/which/when/why/who/where)
# - Intent-aware paraphrasing; injects prompt + top-chunk keywords (similarity-qualified)
# - Dedup helpers to avoid repeated phrases
# - Adaptive neighbor gating (overlap, optional vector sim, optional intent gate)
# Python 3.12+
# =====================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import os
import re
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------
# Configuration (all configurable via environment variables)
# ---------------------------------------------------------------------
BASE_DIR = Path(os.getenv("CHAOS_BASE_DIR", str(Path(__file__).resolve().parent)))
VDB_DIR = Path(os.getenv("CHAOS_VDB_DIR", str(Path(BASE_DIR) / "faiss_index")))
VDB_DIR.mkdir(parents=True, exist_ok=True)
VDB_INDEX_PATH = VDB_DIR / "index.faiss"
VDB_META_PATH = VDB_DIR / "meta.json"

DEFAULT_EMBED_DIM = int(os.getenv("CHAOS_EMBED_DIM", "1024"))
TOPK_NEIGHBORS = int(os.getenv("CHAOS_TOPK", "3"))

# Overlap threshold (raise to 0.60–0.70 for stricter matching)
NEIGHBOR_OVERLAP_THRESHOLD = float(os.getenv("CHAOS_OVERLAP_THRESHOLD", "0.65"))

# Optional: minimum vector similarity cutoff (on normalized vectors; IP ~ cosine)
MIN_VECTOR_SIM = float(os.getenv("CHAOS_MIN_VECTOR_SIM", "0.55"))

# Optional: intent gate to keep neighbors whose intent is compatible with the user's
ENABLE_INTENT_GATE = os.getenv("CHAOS_INTENT_GATE", "true").strip().lower() in ("1", "true", "yes")

# Toggle intent-aware paraphrasing (on by default)
CHAOS_ENABLE_PARAPHRASE = os.getenv("CHAOS_ENABLE_PARAPHRASE", "true").strip().lower() in ("1", "true", "yes")

# Optional: force single-sentence answers
CHAOS_ONE_SENTENCE = os.getenv("CHAOS_ONE_SENTENCE", "false").strip().lower() in ("1", "true", "yes")

# Scenario chunking config
CHUNK_MAX_CHARS = int(os.getenv("CHAOS_CHUNK_MAX_CHARS", "600"))
CHUNK_MIN_CHARS = int(os.getenv("CHAOS_CHUNK_MIN_CHARS", "120"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHAOS_CHUNK_OVERLAP_CHARS", "80"))
TOPK_CHUNKS_FOR_ANSWER = int(os.getenv("CHAOS_TOPK_CHUNKS", "3"))
RESET_INDEX_ON_SET_CASE = os.getenv("CHAOS_RESET_ON_SET", "true").strip().lower() in ("1", "true", "yes")

# Stopwords (optional override via env CSV)
_env_stopwords = os.getenv("CHAOS_STOPWORDS", "")
STOPWORDS: set[str] = (
    {s.strip().lower() for s in _env_stopwords.split(",") if s.strip()}
    or {
        "the", "a", "an", "and", "or", "to", "of", "for", "on", "in", "at", "by", "with",
        "is", "are", "be", "was", "were", "it", "this", "that", "as", "from", "into",
        "since", "about", "over", "last", "next", "vs", "per", "than", "then", "but",
        "so", "we", "you", "they", "i"
    }
)

# Optional environment words & known services (CSV from env; domain-agnostic)
_ENV_WORDS = {
    w.strip().lower()
    for w in os.getenv("CHAOS_ENV_WORDS", "prod,production,staging,preprod,dev,uat,test").split(",")
    if w.strip()
}
_KNOWN_SERVICES = {
    w.strip().lower()
    for w in os.getenv("CHAOS_SERVICES", "").split(",")
    if w.strip()
}

# ---------------------------------------------------------------------
# Optional FAISS support (fallback to NumPy linear search if unavailable)
# ---------------------------------------------------------------------
_HAVE_FAISS = False
try:
    import faiss  # faiss-cpu
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False

# ---------------------------------------------------------------------
# Tokenizer (robust, never returns None)
# ---------------------------------------------------------------------
def semantic_tokens(text: str) -> List[str]:
    """
    Tokenize text for semantic matching.
    Retains '/', '-', '_' for endpoints and versions. Always returns a list.
    """
    s = re.sub(r"[^a-zA-Z0-9/_\-. ]+", " ", (text or "")).lower().strip()
    toks = [t for t in re.split(r"\s+", s) if t and t not in STOPWORDS]
    # keep unique order-preserving
    seen: set[str] = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

# ---------------------------------------------------------------------
# Utility: token overlap (robust)
# ---------------------------------------------------------------------
def _string_overlap(a: str, b: str) -> float:
    at = set(semantic_tokens(a))
    bt = set(semantic_tokens(b))
    if not at:
        return 0.0
    return len(at & bt) / max(1, len(at))

# ---------------------------------------------------------------------
# Deterministic prompt embedder (no external models)
# ---------------------------------------------------------------------
class PromptEmbedder:
    """
    Produces a fixed-dim normalized vector using token-hash bag-of-words
    over semantic_tokens. No external models required.
    """
    def __init__(self, dim: Optional[int] = None):
        self.dim = int(dim or DEFAULT_EMBED_DIM)

    def embed(self, text: str) -> np.ndarray:
        toks = semantic_tokens(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        for t in toks:
            h = abs(hash(t)) % self.dim
            vec[h] += 1.0
        # L2 normalize (so IP ~ cosine)
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            vec /= nrm
        return vec

# ---------------------------------------------------------------------
# Vector DB (FAISS if available, otherwise NumPy cosine)
# ---------------------------------------------------------------------
class VectorDB:
    def __init__(self, dim: int, index_path: Optional[Path] = None, meta_path: Optional[Path] = None, metric: str = "cosine"):
        self.dim = dim
        self.index_path = Path(index_path or VDB_INDEX_PATH)
        self.meta_path = Path(meta_path or VDB_META_PATH)
        self.metric = metric  # "cosine" → inner product on normalized vectors
        self.faiss_index = None
        self._vectors: List[np.ndarray] = []   # fallback storage
        self._meta: List[Dict[str, Any]] = []  # parallel metadata entries

        # Load meta first
        self._meta = self._load_meta()

        # Try FAISS
        if _HAVE_FAISS:
            try:
                if self.metric == "cosine":
                    self.faiss_index = faiss.IndexFlatIP(dim)
                else:
                    self.faiss_index = faiss.IndexFlatL2(dim)
                if self.index_path.is_file():
                    self.faiss_index = faiss.read_index(str(self.index_path))
            except Exception:
                self.faiss_index = None

        # Fallback vectors for NumPy search
        if not self.faiss_index:
            self._load_vectors_np()

    def _load_meta(self) -> List[Dict[str, Any]]:
        if self.meta_path.is_file():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _persist_meta(self):
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_vectors_np(self):
        np_path = str(self.index_path) + ".npy"
        if Path(np_path).is_file():
            try:
                arr = np.load(np_path)
                self._vectors = [v for v in arr]
            except Exception:
                self._vectors = []

    def _persist_vectors_np(self):
        np_path = str(self.index_path) + ".npy"
        try:
            if self._vectors:
                np.save(np_path, np.array(self._vectors, dtype=np.float32))
        except Exception:
            pass

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]):
        """
        vectors: [N, dim] float32, normalized for cosine metric
        metas: list of dicts with parallel length N
        """
        if vectors is None or len(vectors) == 0:
            return
        n = vectors.shape[0]
        now = time.time()
        for i, m in enumerate(metas[:n]):
            m.setdefault("ts", now)
            m.setdefault("id", f"vdb_{int(now*1000)}_{len(self._meta)+i}")
            self._meta.append(m)

        if self.faiss_index is not None:
            try:
                self.faiss_index.add(vectors)
                faiss.write_index(self.faiss_index, str(self.index_path))
            except Exception:
                # fall back to numpy if FAISS write fails
                self.faiss_index = None
                self._vectors.extend(list(vectors))
                self._persist_vectors_np()
        else:
            self._vectors.extend(list(vectors))
            self._persist_vectors_np()

        self._persist_meta()

    def search(self, query_vec: np.ndarray, k: int = TOPK_NEIGHBORS) -> List[Tuple[float, Dict[str, Any]]]:
        if query_vec is None or query_vec.shape[0] != self.dim:
            return []
        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            D, I = self.faiss_index.search(query_vec.reshape(1, -1), k)
            results: List[Tuple[float, Dict[str, Any]]] = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self._meta):
                    continue
                results.append((float(score), self._meta[idx]))
            return results

        # Fallback: linear cosine similarity over normalized vectors
        if not self._vectors:
            return []
        mat = np.stack(self._vectors, axis=0)  # [M, dim]
        sims = mat @ query_vec.reshape(-1, 1)   # [M, 1], cosine if normalized
        sims = sims.squeeze(1)
        idxs = np.argsort(-sims)[:k]
        return [(float(sims[i]), self._meta[i]) for i in idxs]

    def size(self) -> int:
        if self.faiss_index is not None:
            return self.faiss_index.ntotal
        return len(self._vectors)

# ---------------------------------------------------------------------
# Dynamic facet extraction (case + prompt; domain-agnostic)
# ---------------------------------------------------------------------
_REGION_RE = re.compile(r"\b[a-z]{2}-[a-z]+-\d\b", re.I)
_ENDPOINT_RE = re.compile(r"(?:get|post|put|patch|delete)\s+(/[^\s]+)|\b(/[a-z0-9/_\-.]+)", re.I)

def _extract_env(text: str) -> Optional[str]:
    toks = set(semantic_tokens(text))
    if not toks:
        return None
    for w in _ENV_WORDS:
        if w in toks:
            return "prod" if w in ("prod", "production") else w
    return None

def _extract_region(text: str) -> Optional[str]:
    m = _REGION_RE.search(text or "")
    return m.group(0).lower() if m else None

def _extract_endpoints(text: str) -> List[str]:
    out: List[str] = []
    for p in _ENDPOINT_RE.findall((text or "")):
        ep = p[0] or p[1]
        if ep and ep not in out:
            out.append(ep)
    return out

def _extract_service(text: str) -> Optional[str]:
    raw = (text or "").lower()
    # preceding word before 'service'
    m = re.search(r"\b([a-z0-9\-_]+)\s+service\b", raw)
    if m:
        return m.group(1)
    # known services boost
    toks = semantic_tokens(text)
    for t in toks:
        if t in _KNOWN_SERVICES:
            return t
    # proximity heuristic around endpoints
    eps = _extract_endpoints(text)
    if eps:
        base = eps[0].strip("/")
        parts = base.split("/")
        if parts:
            cand = parts[0]
            if cand and cand in toks:
                return cand
    return None

def _extract_facets_from_text(case_text: str, user_prompt: str) -> Dict[str, Any]:
    """
    Derive facets opportunistically from raw text (case + prompt).
    Uses env vars and regex patterns; no hardcoded domain-specific lists.
    """
    combined = f"{case_text or ''}\n{user_prompt or ''}"
    facets: Dict[str, Any] = {}
    env = _extract_env(combined)
    rg = _extract_region(combined)
    svc = _extract_service(combined)
    eps = _extract_endpoints(combined)

    if env: facets["environment"] = env
    if rg: facets["region"] = rg
    if svc: facets["service"] = svc
    if eps: facets["endpoint"] = eps[0]  # prefer first to keep answer concise

    # infer time window if prompt mentions common terms
    prompt_toks = set(semantic_tokens(user_prompt or ""))
    if "last_15m" in prompt_toks:
        facets["time_window"] = "last_15m"
    elif "last_1h" in prompt_toks or ("last" in prompt_toks and "1h" in prompt_toks):
        facets["time_window"] = "last_1h"
    elif "last_24h" in prompt_toks:
        facets["time_window"] = "last_24h"
    elif "last_release" in prompt_toks:
        facets["time_window"] = "last_release"

    return facets

# ---------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------
def _detect_intent(prompt: str) -> str:
    """
    Returns one of:
      - "compare", "show", "what", "which", "when", "why", "who", "where", "action"
    """
    p = (prompt or "").lower().strip()

    if re.search(r"\bcompare\b|\bvs\b|versus\b|baseline\b", p):
        return "compare"
    if re.search(r"\bshow\b|\bdisplay\b|\blist\b|\bfetch\b|\bget\b", p):
        return "show"
    if p.startswith("what "):
        return "what"
    if p.startswith("which "):
        return "which"
    if p.startswith("when "):
        return "when"
    if p.startswith("who "):
        return "who"
    if p.startswith("where "):
        return "where"
    if re.search(r"\bwhy\b|root cause|rca", p):
        return "why"
    return "action"

# ---------------------------------------------------------------------
# Neighbor gating: adaptive overlap + optional vector sim + optional intent gate
# ---------------------------------------------------------------------
_FACET_KEYS = ("environment", "service", "endpoint", "region", "time_window")

def _adaptive_overlap_threshold(base: float, prompt: str) -> float:
    """
    Tighten the token-overlap threshold for richer prompts.
    - Short (<=5 tokens): base
    - Medium (6–15): base + 0.05
    - Long (>15): base + 0.10
    Cap at 0.85 to avoid over-filtering.
    """
    ntoks = len(semantic_tokens(prompt or ""))
    if ntoks <= 5:
        return min(0.85, base)
    if ntoks <= 15:
        return min(0.85, base + 0.05)
    return min(0.85, base + 0.10)

def _aggregate_facets_from_neighbors(
    neighbors: Sequence[Tuple[float, Dict[str, Any]]],
    case_text: str,
    user_prompt: str,
    threshold: float = NEIGHBOR_OVERLAP_THRESHOLD,
) -> Dict[str, Any]:
    """
    Majority-vote facets from *high-quality* neighbors only:
    - require overlap with both case_text and user_prompt >= adaptive threshold
    - optionally require vector similarity >= MIN_VECTOR_SIM
    - optionally require intent consistency with user_prompt
    """
    if not neighbors:
        return {}

    thr = _adaptive_overlap_threshold(threshold, user_prompt)
    user_intent = _detect_intent(user_prompt)

    aggregated: Dict[str, List[str]] = {k: [] for k in _FACET_KEYS}
    for score, meta in neighbors:
        # Gate: vector similarity cutoff (if enabled via env)
        if MIN_VECTOR_SIM > 0.0 and score < MIN_VECTOR_SIM:
            continue

        src_text = (meta.get("canonical") or meta.get("text") or "")
        if not src_text:
            continue

        ov_user = _string_overlap(src_text, user_prompt)
        ov_case = _string_overlap(src_text, case_text)
        if ov_user < thr or ov_case < thr:
            continue

        if ENABLE_INTENT_GATE:
            neighbor_intent = _detect_intent(src_text)
            compatible = {
                ("show", "what"), ("what", "show"),
                ("which", "show"), ("show", "which"),
                ("compare", "action"), ("action", "compare"),
            }
            if not (neighbor_intent == user_intent or (neighbor_intent, user_intent) in compatible):
                continue

        mf = (meta or {}).get("facets", {})
        for k in _FACET_KEYS:
            v = mf.get(k)
            if isinstance(v, str) and v.strip():
                aggregated[k].append(v.strip())

    # majority vote
    result: Dict[str, Any] = {}
    for k, votes in aggregated.items():
        if votes:
            counts: Dict[str, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            winner = max(counts.items(), key=lambda x: x[1])[0]
            result[k] = winner
    return result

# ---------------------------------------------------------------------
# Direct Answer generation (diagnostic/guidance)
# ---------------------------------------------------------------------
def _generate_direct_answer(canonical: str, facets: Dict[str, Any]) -> str:
    intent = _detect_intent(canonical)
    svc = facets.get("service")
    env = facets.get("environment")
    region = facets.get("region")
    win = facets.get("time_window")
    ep = facets.get("endpoint")

    ctx_parts = []
    if svc: ctx_parts.append(svc)
    if env: ctx_parts.append(env)
    if region: ctx_parts.append(region)
    ctx = " / ".join(ctx_parts)
    win_str = f" during {win}" if win else ""

    if intent == "compare":
        base = f"Compare key metrics (p95 latency, error rate){win_str}"
        if ctx: base += f" for {ctx}"
        if ep: base += f" at endpoint {ep}"
        return base + "."
    if intent == "show":
        base = f"Show relevant data (logs, metrics){win_str}"
        if ctx: base += f" for {ctx}"
        if ep: base += f" focusing on {ep}"
        return base + "."
    if intent in ("what", "which"):
        base = "Identify what spiked"
        if ctx: base += f" in {ctx}"
        base += f"{win_str}"
        tips = " Check p95/p99 latency and error rate; correlate with recent changes."
        if ep: tips += f" Include traces for {ep}."
        return base + "." + " " + tips
    if intent == "when":
        base = "Determine when the anomaly started"
        if ctx: base += f" in {ctx}"
        base += f"{win_str}"
        return base + ". Inspect timeline around releases and alerts."
    if intent == "why":
        base = "Investigate why the issue occurred"
        if ctx: base += f" in {ctx}"
        base += f"{win_str}"
        add = " Review recent deployments, config changes, and error signatures."
        if ep: add += f" Analyze traces for {ep}."
        return base + ". " + add
    if intent == "who":
        base = "Identify ownership/approvers for relevant changes"
        if ctx: base += f" in {ctx}"
        base += f"{win_str}"
        return base + ". Check change IDs and approver records."
    if intent == "where":
        base = "Identify where alerts or anomalies occurred"
        if ctx: base += f" in {ctx}"
        base += f"{win_str}"
        return base + ". Review monitoring sources and alert routing."

    base = f"{canonical.strip().capitalize()}"
    if ctx: base += f" for {ctx}"
    base += f"{win_str}"
    if ep: base += f"; include endpoint {ep}"
    return base + "."

# ---------------------------------------------------------------------
# Dedup helpers and intent-aware paraphraser (user-facing)
# ---------------------------------------------------------------------
def _norm_text(t: Optional[str]) -> str:
    """Normalize text for dedup: lowercase, collapse whitespace, strip trailing punctuation."""
    if not t:
        return ""
    return re.sub(r"[ \t\n\r]+", " ", t.strip().lower()).rstrip(".;:,")

def _unique_join(parts: List[str], sep: str = " ") -> str:
    """Join parts keeping only the first occurrence of each normalized element."""
    seen = set()
    out: List[str] = []
    for p in parts:
        np = _norm_text(p)
        if not np or np in seen:
            continue
        seen.add(np)
        out.append(p.strip())
    return sep.join(out).strip()

def _append_unique(acc: List[str], *new_parts: str) -> List[str]:
    """Append parts to an accumulator only if their normalized form isn't already present."""
    seen = { _norm_text(x) for x in acc }
    for npart in new_parts:
        n = _norm_text(npart)
        if n and n not in seen:
            acc.append(npart.strip())
            seen.add(n)
    return acc

def _fmt_ctx(facets: Dict[str, Any]) -> str:
    """Build a compact context string from facets without duplication (service / env / region)."""
    parts: List[str] = []
    svc = facets.get("service")
    env = facets.get("environment")
    region = facets.get("region")
    parts = _append_unique(parts, *(x for x in [svc, env, region] if x))
    return " / ".join(parts)

def _extract_keywords(s: str, exclude: set[str] = STOPWORDS, limit: int = 6) -> List[str]:
    """Extract up to 'limit' meaningful tokens from any text."""
    toks = semantic_tokens(s or "")
    toks = [t for t in toks if t not in exclude and len(t) > 1]
    return toks[:limit]

# ------------------------------ Scenario Chunking ---------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def _chunk_scenario(text: str, max_chars: int, min_chars: int, overlap: int) -> List[str]:
    """
    Create meaningful chunks by aggregating sentences up to max_chars with overlap.
    Ensures each chunk >= min_chars when possible.
    """
    paras = [p.strip() for p in (text or "").split("\n") if p.strip()]
    sentences: List[str] = []
    for pa in paras:
        sentences.extend(_split_sentences(pa))

    chunks: List[str] = []
    buf = ""
    for sent in sentences:
        next_len = (len(buf) + 1 + len(sent)) if buf else len(sent)
        if next_len <= max_chars:
            buf = (buf + " " + sent).strip() if buf else sent
        else:
            if buf:
                if len(buf) >= min_chars:
                    chunks.append(buf)
                else:
                    # try to attach the long sentence if buffer is too small
                    buf = (buf + " " + sent).strip()
                    chunks.append(buf[:max_chars])
                # build overlap tail
                tail = buf[-overlap:].split()
                buf = " ".join(tail) if tail else ""
            else:
                chunks.append(sent[:max_chars])
                tail = sent[-overlap:].split()
                buf = " ".join(tail) if tail else ""
    if buf and len(buf) >= max_chars // 3:
        chunks.append(buf.strip())

    # Ensure non-empty
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks

# ---------------------------------------------------------------------
# Paraphrase (inject prompt + top chunk keywords)
# ---------------------------------------------------------------------
def _paraphrase_for_intent(canonical: str, facets: Dict[str, Any], top_chunk_texts: List[str]) -> str:
    """
    Intent-aware paraphrase that injects:
      - prompt keywords
      - keywords from the top similar scenario chunks (similarity-qualified)
    """
    intent = _detect_intent(canonical)
    ctx = _fmt_ctx(facets)
    ep = facets.get("endpoint")
    win = facets.get("time_window")
    win_str = f" during {win}" if win else ""

    prompt_kw = _extract_keywords(canonical, limit=6)
    chunk_kw: List[str] = []
    for t in top_chunk_texts:
        chunk_kw.extend(_extract_keywords(t, limit=6))
    # dedup while preserving order, prefer prompt kw first
    combined_kw = list(dict.fromkeys(prompt_kw + chunk_kw))[:8]
    kw = " ".join(combined_kw) if combined_kw else ""

    base_prompt = canonical.strip().rstrip("?").capitalize()
    parts: List[str] = []

    if intent == "compare":
        parts = _append_unique(
            parts,
            "Compare current metrics against baseline",
            kw and f"for {kw}" or "",
            win and win_str or "",
            ctx and f"for {ctx}" or "",
            ep and f"at {ep}" or "",
            "(p95/p99 latency, error rate, throughput)."
        )
        return _unique_join(parts)

    if intent == "show":
        parts = _append_unique(
            parts,
            "Show logs and key metrics",
            kw and f"for {kw}" or "",
            win and win_str or "",
            ctx and f"for {ctx}" or "",
            ep and f"focusing on {ep}" or "",
            "; include errors, latency, and recent changes."
        )
        return _unique_join(parts)

    if intent == "what":
        parts = _append_unique(
            parts,
            "Identify what changed or spiked",
            kw and f"related to {kw}" or "",
            ctx and f"in {ctx}" or "",
            win and win_str or "",
            ep and f"at {ep}" or "",
            "; correlate metrics with deployments/config updates."
        )
        return _unique_join(parts)

    if intent == "which":
        parts = _append_unique(
            parts,
            "Determine which configuration or component changed",
            kw and f"affecting {kw}" or "",
            ctx and f"in {ctx}" or "",
            win and win_str or "",
            "; diff configs, deployments, and feature flags."
        )
        return _unique_join(parts)

    if intent == "when":
        parts = _append_unique(
            parts,
            "Find when the anomaly started",
            kw and f"around {kw}" or "",
            ctx and f"in {ctx}" or "",
            win and win_str or "",
            "; inspect alert timelines, deploy windows, and SLO breaches."
        )
        return _unique_join(parts)

    if intent == "why":
        parts = _append_unique(
            parts,
            "Investigate why the issue occurred",
            kw and f"for {kw}" or "",
            ctx and f"in {ctx}" or "",
            win and win_str or "",
            ep and f"at {ep}" or "",
            "; review recent deployments, config changes, and error signatures."
        )
        return _unique_join(parts)

    if intent == "who":
        parts = _append_unique(
            parts,
            "Identify ownership and approvers for the change",
            kw and f"linked to {kw}" or "",
            ctx and f"in {ctx}" or "",
            win and win_str or "",
            "; check change requests, incident records, and approver logs."
        )
        return _unique_join(parts)

    if intent == "where":
        parts = _append_unique(
            parts,
            "Identify where alerts or anomalies occurred",
            kw and f"for {kw}" or "",
            ctx and f"in {ctx}" or "",
            win and win_str or "",
            "; review monitoring sources and alert routing."
        )
        return _unique_join(parts)

    # Fallback: sensible imperative paraphrase
    parts = _append_unique(
        parts,
        base_prompt,
        kw and f"(context: {kw})" or "",
        ctx and f"for {ctx}" or "",
        win and win_str or "",
        ep and f"; include endpoint {ep}" or "",
        "."
    )
    return _unique_join(parts)

def _to_one_sentence(s: str) -> str:
    """Optionally condense to a single sentence."""
    s = s.strip()
    m = re.split(r"[.;]\s*", s)
    return (m[0].strip() + ".") if m and m[0].strip() else s

# ---------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------
@dataclass
class TurnInput:
    case_text: str
    user_prompt: str
    history: List[Dict[str, str]] = field(default_factory=list)  # optional conversational history

@dataclass
class TurnOutput:
    canonical: str            # Paraphrased user-facing text (when enabled)
    facets: Dict[str, Any]
    neighbors_used: int
    vdb_size: int
    direct_answer: str        # Diagnostic/guidance string
    state: Dict[str, Any]

def process_turn(
    inp: TurnInput,
    *,
    retriever: Optional[VectorDB] = None,
    embedder: Optional[PromptEmbedder] = None,
    neighbors_k: int = TOPK_NEIGHBORS,
) -> TurnOutput:
    """
    Pure retrieval flow:
      1) Embed user_prompt
      2) Search neighbors (over scenario chunks + prior prompts)
      3) Aggregate facets from neighbors (with adaptive gating)
      4) Derive facets from case+prompt (regex/env vocab)
      5) Merge (prefer derived where present)
      6) Produce direct, pointed answer (intent-aware)
      7) Paraphrase for user-facing output (inject prompt + top-chunk keywords)
    """
    embedder = embedder or PromptEmbedder()
    retriever = retriever or VectorDB(dim=embedder.dim)

    # Retrieval
    try:
        qvec = embedder.embed(inp.user_prompt)
        neighbors = retriever.search(qvec, k=neighbors_k)
    except Exception:
        neighbors = []

    # Collect top-matching scenario chunks for semantic qualification
    top_chunks: List[Tuple[float, str, str]] = []  # (score, id, text)
    for score, meta in neighbors:
        if MIN_VECTOR_SIM > 0.0 and score < MIN_VECTOR_SIM:
            continue
        if (meta or {}).get("source") == "case_chunk":
            text = meta.get("text") or meta.get("canonical") or ""
            top_chunks.append((float(score), meta.get("id", ""), text))
    # sort by score desc and trim
    top_chunks.sort(key=lambda x: -x[0])
    top_chunk_texts = [t for _, __, t in top_chunks[:TOPK_CHUNKS_FOR_ANSWER]]

    # Neighbor facets
    neighbor_facets = _aggregate_facets_from_neighbors(
        neighbors, inp.case_text, inp.user_prompt, threshold=NEIGHBOR_OVERLAP_THRESHOLD
    )

    # Derived facets from text
    derived_facets = _extract_facets_from_text(inp.case_text, inp.user_prompt)

    # Merge facets (prefer derived when present)
    facets = dict(neighbor_facets or {})
    for k, v in derived_facets.items():
        if v and not facets.get(k):
            facets[k] = v
    # minimal melt default (observability domain)
    if "melt" not in facets:
        facets["melt"] = ["logs", "metrics"]

    # Canonical raw prompt
    raw_canonical = inp.user_prompt.strip()

    # Direct answer (diagnostic)
    direct_answer = _generate_direct_answer(raw_canonical, facets)

    # Paraphrase for user-facing output (with similarity-qualified chunk keywords)
    if CHAOS_ENABLE_PARAPHRASE:
        final_text = _paraphrase_for_intent(raw_canonical, facets, top_chunk_texts)
    else:
        final_text = direct_answer

    if CHAOS_ONE_SENTENCE:
        final_text = _to_one_sentence(final_text)

    state = {
        "case_echo": inp.case_text,
        "canonical_raw": raw_canonical,
        "neighbors_used": len(neighbors),
        "vdb_size": retriever.size(),
        "overlap_threshold": NEIGHBOR_OVERLAP_THRESHOLD,
        "embed_dim": embedder.dim,
        "min_vector_sim": MIN_VECTOR_SIM,
        "intent_gate": ENABLE_INTENT_GATE,
        # include evidence for debugging
        "top_chunks": [
            {"score": sc, "id": cid, "text": (txt[:200] + ("..." if len(txt) > 200 else ""))}
            for sc, cid, txt in top_chunks[:TOPK_CHUNKS_FOR_ANSWER]
        ],
    }

    return TurnOutput(
        canonical=final_text,
        facets=facets,
        neighbors_used=len(neighbors),
        vdb_size=retriever.size(),
        direct_answer=direct_answer,
        state=state,
    )

# ---------------------------------------------------------------------
# Public API + simple converser (persistent VDB)
# ---------------------------------------------------------------------
__all__ = [
    "process_turn",
    "TurnInput",
    "TurnOutput",
    "CaseConverser",
    "PromptEmbedder",
    "VectorDB",
]

class CaseConverser:
    """Stores one case; each ask() is a retrieval turn. Persists scenario chunks and prompts in VDB."""
    def __init__(self, case_text: str):
        self.case_text = case_text
        self.history: List[Dict[str, str]] = []
        self._embedder = PromptEmbedder()
        self._vdb = VectorDB(dim=self._embedder.dim)
        # Index scenario chunks immediately
        self._index_case_chunks(reset=RESET_INDEX_ON_SET_CASE)

    def _reset_vdb_storage(self):
        # Clear persisted indices & meta if present
        for p in [VDB_INDEX_PATH, Path(str(VDB_INDEX_PATH) + ".npy"), VDB_META_PATH]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        # Recreate fresh VDB instance
        self._vdb = VectorDB(dim=self._embedder.dim)

    def _index_case_chunks(self, reset: bool = False):
        if reset:
            self._reset_vdb_storage()

        chunks = _chunk_scenario(self.case_text, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS)
        if not chunks:
            return
        vecs: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        now = time.time()
        for i, ch in enumerate(chunks):
            vecs.append(self._embedder.embed(ch))
            metas.append({
                "source": "case_chunk",
                "id": f"chunk_{int(now)}_{i}",
                "text": ch,
                "canonical": ch,
                "facets": _extract_facets_from_text(self.case_text, ch),
                "case": self.case_text,
            })
        try:
            self._vdb.add(np.stack(vecs, axis=0), metas)
        except Exception:
            pass  # if indexing fails, retrieval falls back gracefully

    def set_case(self, case_text: str) -> None:
        self.case_text = case_text
        self.history.clear()
        # Re-index scenario chunks
        self._index_case_chunks(reset=RESET_INDEX_ON_SET_CASE)

    def ask(self, user_prompt: str, *, track_history: bool = True) -> TurnOutput:
        inp = TurnInput(case_text=self.case_text, user_prompt=user_prompt, history=self.history)
        out = process_turn(inp, retriever=self._vdb, embedder=self._embedder)

        # Optional: capture to conversational history
        if track_history:
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": out.canonical})

        # Persist this prompt in the VDB (index original prompt only)
        try:
            vec = self._embedder.embed(user_prompt)
            vecs = np.stack([vec], axis=0)
            meta = [{
                "source": "prompt",
                "text": user_prompt,           # index the original prompt
                "canonical": out.canonical,    # paraphrased user-facing text
                "facets": out.facets,
                "case": self.case_text,
            }]
            self._vdb.add(vecs, meta)
        except Exception:
            pass

        return out

