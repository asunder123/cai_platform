
###############################################################
# app.py — Capgemini Corporate Anomaly Intelligence Platform
# FINAL VERSION — Stable, ML-enabled, RCA Paraphrasing Enabled
###############################################################

import os
import sys
import shutil
import json
from pathlib import Path
import importlib
import importlib.util

import streamlit as st

# Optional dependency: spaCy
try:
    import spacy
except Exception as e:
    spacy = None

# -----------------------------
# PATH SETUP
# -----------------------------
APP_ROOT = Path(__file__).resolve().parent
MODULE_ROOT = APP_ROOT / "modules"
DEFAULT_MODEL_PATH = APP_ROOT / "training" / "model_corporate_signals"

sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(MODULE_ROOT))


# -----------------------------
# UTILS — Module loader
# -----------------------------
def load_module(module_name: str, func_name: str):
    """
    Loads a function from a Python module.
    Tries regular import first, then falls back to direct file loading from modules/.
    """
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, func_name)
    except Exception:
        filename = module_name.split(".")[-1] + ".py"
        file_path = MODULE_ROOT / filename
        if not file_path.exists():
            raise ImportError(f"Module file not found: {file_path}")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader, f"Invalid import spec for {file_path}"
        spec.loader.exec_module(mod)
        return getattr(mod, func_name)


# -----------------------------
# MODEL DISCOVERY + ENSURE
# -----------------------------
def discover_model_paths():
    """
    Search for spaCy model directories anywhere under APP_ROOT.
    Prefers 'model_corporate_signals' (current), but supports 'corporate_signal_model' (legacy).
    Returns a list of actual directory Paths.
    """
    hits = []
    preferred_names = {"model_corporate_signals", "corporate_signal_model"}  # current + legacy

    for root, dirs, files in os.walk(APP_ROOT):
        for d in dirs:
            if d in preferred_names:
                hits.append(Path(root) / d)
    return hits


def ensure_model_available() -> Path | None:
    """
    Ensure the ML model exists at root-level training/model_corporate_signals/.
    - If DEFAULT_MODEL_PATH exists: return it.
    - Else: if we discover a model elsewhere, move it into DEFAULT_MODEL_PATH.
    - Returns Path or None.
    """
    # If already in the correct place, we’re done.
    if DEFAULT_MODEL_PATH.exists() and DEFAULT_MODEL_PATH.is_dir():
        return DEFAULT_MODEL_PATH

    found = discover_model_paths()
    if not found:
        return None

    # Prefer current naming
    def score(p: Path) -> int:
        return 1 if p.name == "model_corporate_signals" else 0

    found.sort(key=score, reverse=True)
    src = found[0]

    # If src is already the desired target, just return
    if src == DEFAULT_MODEL_PATH:
        return DEFAULT_MODEL_PATH

    # Ensure destination parent exists
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If a stale/partial destination exists, clean it
    if DEFAULT_MODEL_PATH.exists():
        shutil.rmtree(DEFAULT_MODEL_PATH)

    try:
        shutil.move(str(src), str(DEFAULT_MODEL_PATH))
        return DEFAULT_MODEL_PATH
    except Exception as e:
        # If cross-device move fails, attempt copytree
        try:
            shutil.copytree(src, DEFAULT_MODEL_PATH)
            # Optional: clean up source after successful copy
            # shutil.rmtree(src)
            return DEFAULT_MODEL_PATH
        except Exception as e2:
            print("❌ Failed to place model:", e, "| fallback copy error:", e2)
            return None


# -----------------------------
# IMPORT RULE-BASED TOOLS
# -----------------------------
extract_signals = load_module("modules.signal_engine", "extract_signals")
classify_process_buckets = load_module("modules.process_classifier", "classify_process_buckets")
infer_root_causes = load_module("modules.root_cause_engine", "infer_root_causes")
detect_archetypes = load_module("modules.archetype_detector", "detect_archetypes")
predict_risks = load_module("modules.risk_predictor", "predict_risks")
graph_signals = load_module("modules.graph_visualizer", "graph_signals")

# RCA paraphrasing assistant
try:
    from modules.rca_chat import contextual_paraphrase
except Exception:
    contextual_paraphrase = None

# Case Query Engine (conv_assistant.py)
try:
    from conv_assistant import build_case_index
except Exception:
    build_case_index = None


# -----------------------------
# ML MODEL LOAD + HYBRID LOGIC
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_trained_model():
    """
    Attempt to locate and load the spaCy model once per session.
    Returns the loaded model or None.
    """
    if spacy is None:
        return None

    model_path = ensure_model_available()
    if not model_path or not model_path.exists():
        return None

    try:
        model = spacy.load(model_path)
        return model
    except Exception as e:
        print("❌ Error loading ML model:", e)
        return None


def ml_predict(text: str, trained_model):
    """Return ML prediction dict or None."""
    if trained_model is None or spacy is None:
        return None
    doc = trained_model(text)
    return doc.cats


def hybrid_signals(rule_based: dict, ml: dict | None) -> dict:
    """Blend rule-based and ML outputs; ML overrides where present."""
    if not ml:
        return rule_based
    merged = {}
    for key in rule_based:
        merged[key] = round((rule_based[key] * 0.3) + (ml.get(key, 0) * 0.7), 3)
    # Include pure ML categories not in rule-based
    for key in ml:
        if key not in merged:
            merged[key] = round(ml[key], 3)
    return merged


# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Capgemini RCA Intelligence", layout="wide")
st.title("🏢 Capgemini RCA Intelligence Platform (Hybrid ML + Rule-Based)")
st.caption("Real-time Analysis · Process Intelligence · RCA Paraphrasing")

page = st.sidebar.radio(
    "Navigate",
    [
        "Analyze Case",
        "Case Query Engine",
        "Model Status",
        "Help",
    ],
    index=0,
)


# -----------------------------
# PAGE RENDERERS
# -----------------------------
def render_model_status():
    st.header("📡 Model Status")
    model_path = ensure_model_available()
    trained_model = load_trained_model()

    st.write("### Model Path:")
    st.code(str(model_path) if model_path else "None")

    st.write("### Exists:", bool(model_path and model_path.exists()))

    if trained_model:
        st.success("ML model successfully loaded.")
        # spaCy model meta
        try:
            st.json(trained_model.meta)
        except Exception:
            st.info("Model metadata not available.")
    else:
        st.error("ML model not loaded (spaCy missing or model not found).")

    st.write("### Root Directory Contents")
    try:
        st.json(os.listdir(APP_ROOT))
    except Exception as e:
        st.warning(f"Unable to list APP_ROOT: {e}")


def render_analyze_case():
    st.header("🔍 Case Intelligence Analysis")
    text = st.text_area("Enter case text", height=250)

    if st.button("Analyze") and text.strip():
        # Rule-based signals
        rb = extract_signals(text)

        # ML contribution
        trained_model = load_trained_model()
        ml = ml_predict(text, trained_model)

        # Merge signals
        merged = hybrid_signals(rb, ml)

        # Downstream analytics
        buckets = classify_process_buckets(text)
        rca = infer_root_causes(merged)
        arch = detect_archetypes(text)
        risks = predict_risks(merged)
        fig = graph_signals(merged)

        # Display
        st.subheader("🧠 Hybrid Signal Strengths")
        st.json(merged)

        if ml:
            st.subheader("🤖 ML Contribution")
            st.json(ml)
        else:
            st.info("ML contribution not available.")

        st.subheader("📌 Process Buckets")
        st.json(buckets)

        st.subheader("🧩 Root Cause Indicators")
        st.write(rca)

        st.subheader("📚 Archetype Pattern Match")
        st.write(arch)

        st.subheader("⚠ Predicted Risks")
        st.json(risks)

        st.subheader("📊 Signal Graph")
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Graph rendering failed. Ensure plotly is available and graph_signals returns a Figure.")

        st.subheader("📝 Contextual RCA Summary")
        if contextual_paraphrase:
            try:
                summary = contextual_paraphrase(text, merged, rca, arch)
                st.write(summary)
            except Exception as e:
                st.warning(f"RCA paraphrase failed: {e}")
        else:
            st.info("RCA paraphrasing module not available.")


def render_case_query_engine():
    st.header("📘 Case Paragraph Query Engine")
    st.caption(
        "Paste a case → it is chunked, indexed, and queried → "
        "precise sentence-level answers are returned with evidence."
    )

    if build_case_index is None:
        st.error("conv_assistant.build_case_index not available. Ensure conv_assistant.py is present.")
        return

    # Inputs
    case_text = st.text_area(
        "🧾 Case Description",
        height=300,
        placeholder="Paste the full case description here...",
    )

    query_text = st.text_area(
        "🔎 Query / Paragraph Question",
        height=150,
        placeholder="Ask a question grounded in the case (e.g., cause, fix, timeline)...",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        chunk_size = st.slider("Chunk size (words)", min_value=60, max_value=200, value=120)
    with col2:
        overlap = st.slider("Chunk overlap (words)", min_value=10, max_value=80, value=30)
    with col3:
        top_k = st.slider("Top chunks to search", min_value=1, max_value=5, value=3)

    sentences_per_chunk = st.slider("Sentences per chunk", min_value=1, max_value=3, value=1)

    # Execution
    if st.button("Analyze Case"):
        if not case_text.strip():
            st.warning("Please provide a case description.")
        elif not query_text.strip():
            st.warning("Please provide a query.")
        else:
            with st.spinner("Building case index..."):
                try:
                    case_index = build_case_index(
                        case_text=case_text,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                except Exception as e:
                    st.error(f"Index build failed: {e}")
                    return

            with st.spinner("Extracting precise answers..."):
                try:
                    results = case_index.query(
                        query_text=query_text,
                        top_k_chunks=top_k,
                        sentences_per_chunk=sentences_per_chunk,
                    )
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    return

            st.subheader("📌 Extracted Answers")
            if not results:
                st.error("No relevant answers found in the case.")
            else:
                for i, r in enumerate(results, 1):
                    with st.expander(
                        f"Answer {i} | sentence_score={r.get('sentence_score', 0):.3f} | "
                        f"chunk_score={r.get('chunk_score', 0):.3f}"
                    ):
                        st.markdown("**Answer:**")
                        st.write(r.get("answer", ""))

                        st.markdown("---")
                        st.markdown("**Source Context:**")
                        st.write(r.get("source_chunk", ""))

    with st.expander("🔍 How this works"):
        st.markdown(
            """
            **Pipeline**
            1. Case text is cleaned and chunked with overlap  
            2. TF-IDF index built over chunks  
            3. Relevant chunks retrieved using cosine similarity  
            4. Sentences inside chunks are ranked  
            5. Most relevant sentences are returned as answers  

            **Properties**
            - Deterministic (no hallucinations)
            - CPU-only
            - Fully explainable
            - Evidence-preserving
            """
        )


def render_help():
    st.header("ℹ Help & Instructions")
    st.markdown(
        """
        ### How to Train & Use the ML Model

        1. Create a **training_data.jsonl** containing labeled texts.
        2. Validate schema and category names used by downstream rule engines.
        3. Train your spaCy model (e.g., `spacy train`) and output to `training/model_corporate_signals/`.
        4. Launch this app; it will auto-detect and load the model.  
           - Legacy model dir `corporate_signal_model/` is also supported and will be moved/copied into place.

        ### Troubleshooting
        - If the model fails to load:
          - Verify spaCy is installed in the app environment.
          - Confirm `training/model_corporate_signals/` exists and contains a valid spaCy pipeline.
          - Check console/logs for file permission issues when moving/copying directories.
        - If graphs fail to render:
          - Ensure Plotly is installed and `graph_signals()` returns a valid Figure.

        ### Modules & Extensibility
        - **Rule-based** engines reside under `modules/`:
          - `signal_engine.py`, `process_classifier.py`, `root_cause_engine.py`,
            `archetype_detector.py`, `risk_predictor.py`, `graph_visualizer.py`
        - **RCA paraphrasing**: `modules/rca_chat.py` (optional)
        - **Case Query Engine**: `conv_assistant.py` with `build_case_index()`

        ### Notes
        - Hybrid signal blending uses 30% rule-based + 70% ML by default.
        - You can adjust the weights in `hybrid_signals()` to suit your domain.
        """
    )


# -----------------------------
# ROUTER
# -----------------------------
if page == "Model Status":
    render_model_status()
elif page == "Analyze Case":
    render_analyze_case()
elif page == "Case Query Engine":
       render_case_query_engine()

