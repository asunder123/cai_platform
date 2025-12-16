###############################################################
# app.py — Capgemini Corporate Anomaly Intelligence Platform
# FINAL VERSION — Stable, ML-enabled, RCA Paraphrasing Enabled
###############################################################

import streamlit as st
import sys, os, json, shutil
from pathlib import Path
import importlib, importlib.util
import spacy


#################################################################
# PATH SETUP
#################################################################

APP_ROOT = Path(__file__).resolve().parent
MODULE_ROOT = APP_ROOT / "modules"
DEFAULT_MODEL_PATH = APP_ROOT / "training/model_corporate_signals"

sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(MODULE_ROOT))


#################################################################
# AUTO-DETECT + REPAIR MODEL LOCATION
#################################################################

#################################################################
# AUTO-DETECT + REPAIR MODEL LOCATION
#################################################################

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


def ensure_model_available():
    """
    Ensure ML model exists at root-level training/model_corporate_signals/.
    - If DEFAULT_MODEL_PATH exists: return it.
    - Else: if we discover a model elsewhere, move it into DEFAULT_MODEL_PATH.
    - Returns Path or None.
    """
    # If already in the correct place, we’re done.
    if DEFAULT_MODEL_PATH.exists() and DEFAULT_MODEL_PATH.is_dir():
        print("✅ Model found in correct location:", DEFAULT_MODEL_PATH)
        return DEFAULT_MODEL_PATH

    found = discover_model_paths()
    if not found:
        print("❌ No model directory named 'model_corporate_signals' or 'corporate_signal_model' found anywhere")
        return None

    # Choose the best candidate (prefer current naming)
    def score(p: Path) -> int:
        return 1 if p.name == "model_corporate_signals" else 0
    found.sort(key=score, reverse=True)

    src = found[0]
    # If src is already the desired target, just return
    if src == DEFAULT_MODEL_PATH:
        print("✅ Model already located at:", DEFAULT_MODEL_PATH)
        return DEFAULT_MODEL_PATH

    # Ensure destination parent exists
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If a stale/partial destination exists, clean it
    if DEFAULT_MODEL_PATH.exists():
        shutil.rmtree(DEFAULT_MODEL_PATH)

    print(f"⚠ Model found at: {src} → moving to {DEFAULT_MODEL_PATH}")
    try:
        shutil.move(str(src), str(DEFAULT_MODEL_PATH))
        print("✅ Model moved to:", DEFAULT_MODEL_PATH)
        return DEFAULT_MODEL_PATH
    except Exception as e:
        print("❌ Failed to move model:", e)
       


def ensure_model_available():
    """Ensure ML model exists in root-level training/model_corporate_signals/."""
    found = discover_model_paths()

    if not found:
        print("❌ No corporate_signal_model found anywhere")
        return None

    # If already in correct place
    if DEFAULT_MODEL_PATH.exists():
        print("✅ Model found in correct location:", DEFAULT_MODEL_PATH)
        return DEFAULT_MODEL_PATH

    # Move first found into correct location
    src = found[0]
    print(f"⚠ Model found at incorrect location: {src}")
    if DEFAULT_MODEL_PATH.exists():
        shutil.rmtree(DEFAULT_MODEL_PATH)
    shutil.move(str(src), str(DEFAULT_MODEL_PATH))
    print("✅ Model moved to:", DEFAULT_MODEL_PATH)
    return DEFAULT_MODEL_PATH


#################################################################
# MODULE LOADER
#################################################################

def load_module(module_name, func_name):
    """Loads a module function from modules/ with fallback."""
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, func_name)
    except Exception:
        filename = module_name.split(".")[-1] + ".py"
        file_path = MODULE_ROOT / filename
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, func_name)


#################################################################
# IMPORT RULE-BASED TOOLS
#################################################################

extract_signals = load_module("modules.signal_engine", "extract_signals")
classify_process_buckets = load_module("modules.process_classifier", "classify_process_buckets")
infer_root_causes = load_module("modules.root_cause_engine", "infer_root_causes")
detect_archetypes = load_module("modules.archetype_detector", "detect_archetypes")
predict_risks = load_module("modules.risk_predictor", "predict_risks")
graph_signals = load_module("modules.graph_visualizer", "graph_signals")
from modules.rca_chat import contextual_paraphrase


#################################################################
# LOAD ML MODEL
#################################################################

MODEL_PATH = ensure_model_available()
trained_model = None

if MODEL_PATH and MODEL_PATH.exists():
    try:
        trained_model = spacy.load(MODEL_PATH)
        print("🎉 ML MODEL LOADED:", MODEL_PATH)
    except Exception as e:
        print("❌ Error loading ML model:", e)
else:
        print("⚠ ML model not found. Running rule-based only.")


def ml_predict(text):
    """Return ML prediction or None."""
    if trained_model is None:
        return None
    doc = trained_model(text)
    return doc.cats


def hybrid_signals(rule_based, ml):
    """ML overrides rule-based where present."""
    if not ml:
        return rule_based
    merged = {}
    for key in rule_based:
        merged[key] = round((rule_based[key] * 0.3) + (ml.get(key, 0) * 0.7), 3)
    return merged


#################################################################
# STREAMLIT UI
#################################################################

st.set_page_config(page_title="Capgemini RCA Intelligence", layout="wide")
st.title("🏢 Capgemini RCA Intelligence Platform (Hybrid ML + Rule-Based)")
st.caption("Real-time Analysis · Process Intelligence · RCA Paraphrasing")


page = st.sidebar.radio("Navigate", [
    "Analyze Case",
    "Conversational Assistant",
    "Model Status",
    "Help"
])


#################################################################
# PAGE — MODEL STATUS
#################################################################

if page == "Model Status":
    st.header("📡 Model Status")

    st.write("### Model Path:")
    st.code(str(MODEL_PATH))

    st.write("### Exists:", MODEL_PATH.exists() if MODEL_PATH else False)

    if trained_model:
        st.success("ML model successfully loaded.")
        st.json(trained_model.meta)
    else:
        st.error("ML model not loaded.")

    st.write("### Root Directory Contents")
    st.json(os.listdir(APP_ROOT))


#################################################################
# PAGE — ANALYZE CASE
#################################################################

if page == "Analyze Case":
    st.header("🔍 Case Intelligence Analysis")

    text = st.text_area("Enter case text", height=250)

    if st.button("Analyze") and text.strip():

        rb = extract_signals(text)
        ml = ml_predict(text)
        merged = hybrid_signals(rb, ml)

        buckets = classify_process_buckets(text)
        rca = infer_root_causes(merged)
        arch = detect_archetypes(text)
        risks = predict_risks(merged)
        fig = graph_signals(merged)

        st.subheader("🧠 Hybrid Signal Strengths")
        st.json(merged)

        if ml:
            st.subheader("🤖 ML Contribution")
            st.json(ml)

        st.subheader("📌 Process Buckets")
        st.json(buckets)

        st.subheader("🧩 Root Cause Indicators")
        st.write(rca)

        st.subheader("📚 Archetype Pattern Match")
        st.write(arch)

        st.subheader("⚠ Predicted Risks")
        st.json(risks)

        st.subheader("📊 Signal Graph")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📝 Contextual RCA Summary")
        summary = contextual_paraphrase(text, merged, rca, arch)
        st.write(summary)




###############################################################
# PAGE — Conversational Assistant (Pure FAISS Retrieval)
###############################################################
from modules.conv_assistant import CaseConverser
import streamlit as st
import json

if page == "Conversational Assistant":
    st.header("💬 RCA Conversational Assistant (Pure Retrieval)")

    # Persistent converser across interactions
    if "converser" not in st.session_state:
        st.session_state.converser = None

    # Case description input
    case_text = st.text_area(
        "Provide the case description (context for all queries)",
        height=140,
        placeholder="Example: Checkout service experienced elevated error rates in prod (ap-south-1) after the last deployment..."
    )

    # Initialize converser when case changes
    if case_text.strip():
        if not st.session_state.converser or st.session_state.converser.case_text != case_text.strip():
            st.session_state.converser = CaseConverser(case_text.strip())

    # User prompt input
    user_prompt = st.text_input(
        "Ask a question about the case",
        placeholder="e.g., What was spiking ?"
    )

    # Process button
    if st.button("Process Query", type="primary", use_container_width=True):
        if not case_text.strip():
            st.warning("Please provide a case description.")
        elif not user_prompt.strip():
            st.warning("Please provide a query.")
        else:
            try:
                turn = st.session_state.converser.ask(user_prompt.strip())
            except Exception as e:
                st.exception(e)
                st.stop()

            # Canonical query
            st.markdown("### ✅ Canonical Query")
            st.write(turn.canonical or "_(empty)_")

            # Direct Answer
            st.markdown("### 🎯 Direct Answer")
            st.write(turn.direct_answer or "_No direct answer generated._")

            # Diagnostics
            st.markdown("### ⚙️ Diagnostics")
            st.write(f"""
            - Neighbors Used: `{turn.neighbors_used}`
            - VDB Size: `{turn.vdb_size}`
            - Embed Dim: `{turn.state.get('embed_dim')}`
            - Overlap Threshold: `{turn.state.get('overlap_threshold')}`
            """)

            # Optional: Show facets
            with st.expander("Facets"):
                st.json(turn.facets)

            # Optional: Show state details
            with st.expander("State"):
               st.json(turn.state)



#################################################################
# PAGE — HELP
#################################################################

if page == "Help":
    st.header("ℹ Help & Instructions")

    st.markdown("""
### How to Train & Use the ML Model

1. Create a **training_data.jsonl**
2. Validate it:
""")