Here’s a well-structured **README.md** for your project:

***

# Capgemini Corporate Anomaly Intelligence Platform

**FINAL VERSION — Stable, ML-enabled, RCA Paraphrasing Enabled**

***

## 📌 Overview

The **Capgemini RCA Intelligence Platform** is a hybrid anomaly detection and root cause analysis (RCA) system combining **rule-based logic** with **ML-powered insights**. It supports real-time analysis, process classification, risk prediction, and contextual RCA paraphrasing through an interactive **Streamlit UI**.

***

## ✅ Key Features

*   **Hybrid Analysis**: Combines rule-based signal extraction with ML predictions for improved accuracy.
*   **ML Model Integration**: Uses a spaCy-based model for corporate signal classification.
*   **Root Cause Explorer**: Conversational RCA assistant for scenario interpretation.
*   **Dynamic Visualization**: Graph-based signal visualization using Plotly.
*   **Auto Model Repair**: Detects and relocates ML model if misplaced.
*   **Modular Design**: Pluggable architecture for signal engines, classifiers, and RCA tools.

***

## 🛠 Tech Stack

*   **Python 3.9+**
*   **Streamlit** (UI framework)
*   **spaCy** (ML model for text classification)
*   **Plotly** (visualizations)

***

## 📂 Project Structure

    app.py                      # Main Streamlit application
    modules/                    # Custom rule-based engines and utilities
    training/model_corporate_signals/  # Default ML model directory

***

## 🚀 Getting Started

### 1. **Clone the Repository**

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Verify ML Model**

*   Ensure the spaCy model exists at:

<!---->

    training/model_corporate_signals/

*   If missing, the app will auto-detect and move any legacy model (`corporate_signal_model`) into the correct location.

### 4. **Run the Application**

```bash
streamlit run app.py
```

***

## 🔍 Application Pages

*   **Analyze Case**: Enter case text → Extract signals, classify processes, predict risks, and generate RCA summary.
*   **Root Cause Explorer**: Conversational RCA assistant for scenario-based interpretation.
*   **Model Status**: Check ML model path, metadata, and load status.
*   **Help**: Instructions for training and usage.

***

## 📦 Training the ML Model

1.  Prepare `training_data.jsonl` with labeled examples.
2.  Validate using spaCy CLI:

```bash
python -m spacy validate
```

3.  Train and save the model under `training/model_corporate_signals/`.

***

## ⚠ Notes

*   If ML model is unavailable, the platform falls back to **rule-based analysis only**.
*   Ensure proper permissions for file operations (model relocation).

***

## 🧩 Future Enhancements

*   Integration with enterprise knowledge graphs.
*   Advanced RCA summarization using LLMs.
*   Real-time anomaly alerts via messaging platforms.

