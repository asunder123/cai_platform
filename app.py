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



# --- Replace your existing ml_predict with this hardened version ---
def ml_predict(text: str):
    """
    Return ML prediction (doc.cats) or {} with informative logging.
    Ensures a text classification pipe exists and avoids returning None.
    """
    if trained_model is None:
        # No model loaded at all
        print("⚠ ml_predict: trained_model is None — ML will not contribute.")
        return {}

    # Try common spaCy textcat pipe names
    possible_pipes = ["textcat", "textcat_multilabel", "textcat_ensemble"]
    pipe_name = next((p for p in possible_pipes if trained_model.has_pipe(p)), None)

    if pipe_name is None:
        # Model loaded but has no text classification pipe
        print(f"⚠ ml_predict: No text classification pipe found in pipeline {trained_model.pipe_names}")
        try:
            doc = trained_model(text)  # still run; doc.cats likely empty
            return doc.cats or {}
        except Exception as e:
            print("❌ ml_predict: error running model without textcat:", e)
            return {}

    # Run the model and return category scores
    try:
        doc = trained_model(text)
        cats = doc.cats or {}
        # Optional: log to console for quick triage
        if not cats:
            print("ℹ ml_predict: doc.cats is empty (model ran, but produced no categories).")
        else:
            print(f"✅ ml_predict: categories from '{pipe_name}' → {cats}")
        return cats
    except Exception as e:
        print("❌ ml_predict: error running textcat:", e)
        return {}



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
    "Root Cause Explorer",
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

    # ========== On Analyze: compute and persist results ==========
    if st.button("Analyze") and text.strip():
        # --- Compute all outputs (unchanged from your logic) ---
        rb = extract_signals(text)
        ml = ml_predict(text)              # may be None if model not loaded
        merged = hybrid_signals(rb, ml)
        buckets = classify_process_buckets(text)
        rca = infer_root_causes(merged)
        arch = detect_archetypes(text)
        risks = predict_risks(merged)

        # --- Persist in session_state for safe re-use across reruns ---
        st.session_state["text"] = text
        st.session_state["rb"] = rb
        st.session_state["ml"] = ml
        st.session_state["merged"] = merged
        st.session_state["buckets"] = buckets
        st.session_state["rca"] = rca
        st.session_state["arch"] = arch
        st.session_state["risks"] = risks

        # --- Immediate display of JSON/sections (same as before) ---
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

    # ========== Visualizations & download — driven by session_state ==========
    # Render only if we have prior results in session_state
    if "merged" in st.session_state and st.session_state["merged"]:
        rb = st.session_state.get("rb", {})
        ml = st.session_state.get("ml", None)
        merged = st.session_state["merged"]
        buckets = st.session_state.get("buckets", [])
        rca = st.session_state.get("rca", {})
        arch = st.session_state.get("arch", {})
        risks = st.session_state.get("risks", {})
        text_cached = st.session_state.get("text", "")
        ml_available = ml is not None

        # -------------------- Plotly helpers (inline, self-contained) --------------------
        import plotly.graph_objects as go
        import plotly.express as px

        def _fig_bar_sorted(signals: dict):
            items = sorted(signals.items(), key=lambda kv: kv[1], reverse=True)
            names = [k for k, _ in items]
            values = [v for _, v in items]
            fig = go.Figure(data=[go.Bar(x=names, y=values, marker_color="#1F77B4")])
            fig.update_layout(
                title="Signal Strengths (sorted)",
                xaxis_title="Signals",
                yaxis_title="Strength",
                height=420,
                template="plotly_white"
            )
            return fig

        def _fig_radar(signals: dict):
            if not signals:
                return go.Figure()
            names = list(signals.keys())
            values = [signals[k] for k in names]
            names += [names[0]]  # close loop
            values += [values[0]]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values, theta=names, fill='toself', name='Signals',
                marker=dict(color="#17A2B8")
            ))
            fig.update_layout(
                title="Signal Pattern (Radar)",
                polar=dict(radialaxis=dict(visible=True, range=[0, max(1.0, max(values))])),
                showlegend=False,
                height=420,
                template="plotly_white"
            )
            return fig

        def _fig_donut_ml_vs_rule(rule_based: dict, ml_dict: dict | None):
            ml_sum = float(sum(ml_dict.values())) if ml_dict else 0.0
            rb_sum = float(sum(rule_based.values())) if rule_based else 0.0
            parts = [{"label": "Rule-based", "value": rb_sum},
                     {"label": "ML", "value": ml_sum}]
            fig = px.pie(
                parts, names="label", values="value", hole=0.55, color="label",
                color_discrete_map={"Rule-based": "#6C757D", "ML": "#20C997"}
            )
            fig.update_layout(
                title="Contribution Split (ML vs Rule-based)",
                height=320,
                template="plotly_white"
            )
            return fig

        def _fig_top_n_table(signals: dict, n: int = 5):
            items = sorted(signals.items(), key=lambda kv: kv[1], reverse=True)[:n]
            fig = go.Figure(data=[go.Table(
                header=dict(values=["Signal", "Strength"], fill_color="#343A40", font=dict(color="white")),
                cells=dict(values=[[k for k, _ in items], [round(v, 3) for _, v in items]])
            )])
            fig.update_layout(title=f"Top {n} Signals", height=320, template="plotly_white")
            return fig

        def _fig_heatmap_buckets(signals: dict, buckets_list: list[str]):
            """
            Simple keyword-based mapping for visualization. Replace with your canonical mapping if available.
            """
            mapping = {
                "Operations": ["capacity", "latency", "timeout", "cpu"],
                "Release & Change": ["change", "cab", "feature", "rollback", "flag"],
                "Data Governance": ["data", "schema", "mapping", "reconciliation", "id"],
                "Vendor Management": ["vendor", "third", "provider", "partner"],
                "Network/Security": ["waf", "firewall", "rate", "ingress", "ddos", "acl"]
            }
            
            if isinstance(buckets_data, dict):
             x_labels = list(buckets_data.keys())
            
            elif isinstance(buckets_data, list):
             x_labels = buckets_data
           
            else:
             x_labels = ["Bucket1", "Bucket2", "Bucket3"]


            y = list(signals.keys())
            z = []
            for sig in y:
                lower = sig.lower()
                row = []
                for b in x:
                    keywords = mapping.get(b, [])
                    score = signals[sig] if any(k in lower for k in keywords) else 0.0
                    row.append(score)
                z.append(row)

            fig = go.Figure(data=go.Heatmap(
                z=z, x=x, y=y, colorscale="Viridis", colorbar=dict(title="Strength")
            ))
            fig.update_layout(title="Signals × Buckets (heatmap)", height=420, template="plotly_white")
            return fig

        # -------------------- Visualization controls --------------------
        st.subheader("📊 Visualizations")
        with st.expander("Visualization options", expanded=True):
            show_bar   = st.checkbox("Sorted bar chart", value=True, key="viz_bar_ss")
            show_radar = st.checkbox("Radar pattern", value=True, key="viz_radar_ss")
            show_donut = st.checkbox("ML vs Rule donut", value=ml_available, key="viz_donut_ss")
            show_table = st.checkbox("Top-5 table", value=True, key="viz_table_ss")
            show_heat  = st.checkbox("Bucket heatmap", value=True, key="viz_heat_ss")
            top_n      = st.number_input("Top-N for table", min_value=3, max_value=15,
                                         value=5, step=1, key="viz_topn_ss")

        # -------------------- Render charts safely --------------------
        try:
            if show_bar:
                st.plotly_chart(_fig_bar_sorted(merged), use_container_width=True)
            if show_radar:
                st.plotly_chart(_fig_radar(merged), use_container_width=True)
            if show_donut and ml_available:
                st.plotly_chart(_fig_donut_ml_vs_rule(rb, ml), use_container_width=True)
            if show_table:
                st.plotly_chart(_fig_top_n_table(merged, n=int(top_n)), use_container_width=True)
            if show_heat:
                st.plotly_chart(_fig_heatmap_buckets(merged, buckets), use_container_width=True)
        except Exception as e:
            st.warning(f"Enhanced visuals unavailable: {e}")

        # -------------------- Download helpers --------------------
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download signals (JSON)",
                data=json.dumps(merged, indent=2),
                file_name="signals.json",
                mime="application/json",
                key="download_signals_json_ss"
            )
        with col2:
            st.download_button(
                "⬇️ Download RCA summary (TXT)",
                data=str(rca),
                file_name="rca_summary.txt",
                mime="text/plain",
                key="download_rca_txt_ss"
            )

        # -------------------- Contextual RCA Summary (reuses cached data) --------------------
        st.subheader("📝 Contextual RCA Summary")
        try:
            summary = contextual_paraphrase(text_cached or text, merged, rca, arch)
            st.write(summary)
        except Exception as e:
            st.warning(f"Paraphrase unavailable: {e}")

    else:
       print("Error")




#################################################################
# PAGE — RCA ASSISTANT (Refined Continuity Version)
#################################################################

if page == "Root Cause Explorer":
    st.header("🧠 Conversational Root Cause Explorer")

    # --- Initialize session state for continuity ---
    if "rca_chat" not in st.session_state:
        st.session_state["rca_chat"] = []  # Chat timeline
    if "rca_last" not in st.session_state:
        st.session_state["rca_last"] = {}  # Last RCA analysis
    if "rca_state" not in st.session_state:
        st.session_state["rca_state"] = {}  # Top signals, causes, archetypes

    # --- Input area for initial scenario ---
    text = st.text_area("Enter scenario text", height=250, placeholder="Paste an incident description...")
    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run_clicked = st.button("Run RCA")
    with col_clear:
        if st.button("Clear Conversation"):
            st.session_state["rca_chat"].clear()
            st.session_state["rca_last"].clear()
            st.session_state["rca_state"].clear()
            st.success("Conversation cleared.")

    # --- Initial RCA analysis ---
    if run_clicked and text.strip():
        rb = extract_signals(text)
        ml = ml_predict(text)
        merged = hybrid_signals(rb, ml)
        rca = infer_root_causes(merged)
        arch = detect_archetypes(text)

        sections = contextual_paraphrase(
            text,
            merged,
            rca,
            arch,
            return_dict=True
        )

        # Persist state
        st.session_state["rca_last"] = {
            "text": text,
            "sections": sections,
            "ml": ml,
            "merged": merged,
            "rca": rca,
            "arch": arch
        }
        st.session_state["rca_state"] = sections.get("state", {})

        # Add to chat timeline
        st.session_state["rca_chat"].append({"role": "user", "content": text})
        st.session_state["rca_chat"].append({"role": "assistant", "content": sections["synthesis"]})

    # --- Render chat timeline ---
    st.subheader("💬 RCA Conversation")
    for msg in st.session_state["rca_chat"]:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # --- Continuity: Follow-ups and Next Prompts ---
    last = st.session_state.get("rca_last", {})
    sections = (last or {}).get("sections", {})
    ml = (last or {}).get("ml", None)

    if sections:
        if ml:
            with st.expander("🤖 ML Contribution"):
                st.json(ml)

        # Follow-up Questions
        followups = sections.get("followups", [])
        if followups:
            st.subheader("🔍 Suggested Follow-up Questions")
            cols = st.columns(2)
            for i, q in enumerate(followups):
                with cols[i % 2]:
                    if st.button(f"Ask: {q}", key=f"ask_{i}"):
                        st.session_state["rca_chat"].append({"role": "user", "content": q})
                        # Treat follow-up as refinement
                        combined_text = f"{last.get('text', '')}\n\nFollow-up query: {q}"
                        rb2 = extract_signals(combined_text)
                        ml2 = ml_predict(combined_text)
                        merged2 = hybrid_signals(rb2, ml2)
                        rca2 = infer_root_causes(merged2)
                        arch2 = detect_archetypes(combined_text)
                        sections2 = contextual_paraphrase(combined_text, merged2, rca2, arch2, return_dict=True)
                        st.session_state["rca_last"] = {
                            "text": combined_text,
                            "sections": sections2,
                            "ml": ml2,
                            "merged": merged2,
                            "rca": rca2,
                            "arch": arch2
                        }
                        st.session_state["rca_state"] = sections2.get("state", {})
                        st.session_state["rca_chat"].append({"role": "assistant", "content": sections2["synthesis"]})
                        st.experimental_rerun()

        # Next-turn prompts
        next_prompts = sections.get("next_prompts", [])
        if next_prompts:
            st.subheader("➡️ Suggested Next Inputs")
            for i, p in enumerate(next_prompts):
                if st.button(f"Use: {p}", key=f"use_{i}"):
                    st.session_state["rca_chat"].append({"role": "user", "content": p})
                    combined_text = f"{last.get('text', '')}\n\nAdditional input: {p}"
                    rb2 = extract_signals(combined_text)
                    ml2 = ml_predict(combined_text)
                    merged2 = hybrid_signals(rb2, ml2)
                    rca2 = infer_root_causes(merged2)
                    arch2 = detect_archetypes(combined_text)
                    sections2 = contextual_paraphrase(combined_text, merged2, rca2, arch2, return_dict=True)
                    st.session_state["rca_last"] = {
                        "text": combined_text,
                        "sections": sections2,
                        "ml": ml2,
                        "merged": merged2,
                        "rca": rca2,
                        "arch": arch2
                    }
                    st.session_state["rca_state"] = sections2.get("state", {})
                    st.session_state["rca_chat"].append({"role": "assistant", "content": sections2["synthesis"]})
                    st.experimental_rerun()

    # --- Free-text continuation ---
    user_next = st.chat_input("Add more context, paste logs, or answer a follow-up…")
    if user_next:
        st.session_state["rca_chat"].append({"role": "user", "content": user_next})
        combined_text = f"{last.get('text', '')}\n\nAdditional context: {user_next}"
        rb2 = extract_signals(combined_text)
        ml2 = ml_predict(combined_text)
        merged2 = hybrid_signals(rb2, ml2)
        rca2 = infer_root_causes(merged2)
        arch2 = detect_archetypes(combined_text)
        sections2 = contextual_paraphrase(combined_text, merged2, rca2, arch2, return_dict=True)
        st.session_state["rca_last"] = {
            "text": combined_text,
            "sections": sections2,
            "ml": ml2,
            "merged": merged2,
            "rca": rca2,
            "arch": arch2
        }
        st.session_state["rca_state"] = sections2.get("state", {})
        st.session_state["rca_chat"].append({"role": "assistant", "content": sections2["synthesis"]})
        


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