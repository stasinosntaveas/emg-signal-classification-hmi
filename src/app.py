import streamlit as st
import os
import numpy as np
import scipy.io
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import confusion_matrix

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="EMG HMI System", layout="wide")

st.title("EMG Gesture Classification HMI")

# ========================
# LOAD MODELS (CACHE)
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

@st.cache_resource
def load_models():
    scaler = joblib.load(os.path.join(RESULTS_DIR, "scaler.pkl"))
    svm = joblib.load(os.path.join(RESULTS_DIR, "svm.pkl"))
    knn = joblib.load(os.path.join(RESULTS_DIR, "knn.pkl"))
    rf = joblib.load(os.path.join(RESULTS_DIR, "rf.pkl"))
    return scaler, svm, knn, rf

scaler, svm, knn, rf = load_models()

models = {
    "SVM": svm,
    "KNN": knn,
    "Random Forest": rf
}

# ========================
# SIDEBAR CONTROLS
# ========================
st.sidebar.title("HMI Controls")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

window_size = st.sidebar.slider(
    "Window Size",
    min_value=50,
    max_value=1000,
    value=200,
    step=10
)

window_size_manual = st.sidebar.number_input(
    "Window Size (manual override)",
    min_value=50,
    max_value=1000,
    value=window_size,
    step=10
)

step_size = st.sidebar.slider(
    "Step Size",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

step_size_manual = st.sidebar.number_input(
    "Step Size (manual override)",
    min_value=10,
    max_value=500,
    value=step_size,
    step=10
)

# final values (manual overrides slider if changed)
window_size = window_size_manual
step_size = step_size_manual

st.sidebar.markdown("---")
st.sidebar.write(f"Selected model: **{model_name}**")

model = models[model_name]

# ========================
# FEATURE EXTRACTION
# ========================
def extract_features(window):
    features = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]

        mav = np.mean(np.abs(signal))
        rms = np.sqrt(np.mean(signal ** 2))
        wl = np.sum(np.abs(np.diff(signal)))
        zcr = np.sum(np.diff(np.sign(signal)) != 0)
        ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
        var = np.var(signal)
        sk = skew(signal)

        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal))
        fft_sum = np.sum(fft_vals) + 1e-10
        mean_freq = np.sum(freqs * fft_vals) / fft_sum
        cumsum = np.cumsum(fft_vals)
        median_freq = freqs[np.searchsorted(cumsum, cumsum[-1] / 2)]

        features.extend([mav, rms, wl, zcr, ssc, var, sk, mean_freq, median_freq])

    return features


# ========================
# FILE UPLOAD
# ========================
uploaded_file = st.file_uploader("Upload DB2 .mat file", type=["mat"])

if uploaded_file is not None:

    data = scipy.io.loadmat(uploaded_file)

    emg = data["emg"]
    labels = data["restimulus"].flatten()

    st.write("EMG shape:", emg.shape)

    # ========================
    # WINDOWING
    # ========================
    X = []
    y = []

    for i in range(0, len(emg) - window_size, step_size):
        window = emg[i:i + window_size]
        label_window = labels[i:i + window_size]

        label = np.bincount(label_window).argmax()

        if label in (6, 17):
            X.append(extract_features(window))
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    st.write("Generated windows:", X.shape)

    # ========================
    # PREDICTION
    # ========================
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    acc = np.mean(preds == y)

    st.subheader("Results")
    st.write(f"Accuracy: {acc:.4f}")

    # ========================
    # CONFUSION MATRIX
    # ========================
    cm = confusion_matrix(y, preds, normalize="true")

    class_labels = [6, 17]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax
    )

    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    st.pyplot(fig)

    # ========================
    # PER-SAMPLE PREDICTIONS TABLE
    # ========================
    st.subheader("Per-Sample Predictions")

    gesture_names = {6: "Gesture 6 (Flexion)", 17: "Gesture 17 (Extension)"}

    df = pd.DataFrame({
        "Sample #": np.arange(1, len(y) + 1),
        "True Label": [gesture_names[int(v)] for v in y],
        "Predicted Label": [gesture_names[int(v)] for v in preds],
        "Correct": ["✓" if t == p else "✗" for t, p in zip(y, preds)]
    })

    n_correct = (df["Correct"] == "✓").sum()
    n_wrong = (df["Correct"] == "✗").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Correct", n_correct)
    col3.metric("Wrong", n_wrong)

    filter_option = st.radio(
        "Show",
        ["All", "Only Correct", "Only Wrong"],
        horizontal=True
    )

    if filter_option == "Only Correct":
        display_df = df[df["Correct"] == "✓"]
    elif filter_option == "Only Wrong":
        display_df = df[df["Correct"] == "✗"]
    else:
        display_df = df

    def render_predictions_table(df_to_render):
        rows_html = ""
        for _, row in df_to_render.iterrows():
            is_correct = row["Correct"] == "✓"
            bg = "#c3e6cb" if is_correct else "#f5c6cb"
            fg = "#155724" if is_correct else "#721c24"
            icon_color = "#1a7a31" if is_correct else "#9b1c1c"
            rows_html += (
                f'<tr style="background-color:{bg};color:{fg};font-size:15px;">'
                f'<td style="padding:10px 15px;text-align:center;font-weight:bold;">{row["Sample #"]}</td>'
                f'<td style="padding:10px 15px;">{row["True Label"]}</td>'
                f'<td style="padding:10px 15px;">{row["Predicted Label"]}</td>'
                f'<td style="padding:10px 15px;text-align:center;font-size:22px;font-weight:bold;color:{icon_color};">{row["Correct"]}</td>'
                f'</tr>'
            )
        table_html = (
            '<div style="max-height:520px;overflow-y:auto;border-radius:8px;border:2px solid #adb5bd;">'
            '<table style="width:100%;border-collapse:collapse;font-family:Arial,sans-serif;">'
            '<thead><tr style="background-color:#343a40;color:#ffffff;font-size:16px;position:sticky;top:0;">'
            '<th style="padding:13px 15px;text-align:center;">Sample #</th>'
            '<th style="padding:13px 15px;text-align:left;">True Label</th>'
            '<th style="padding:13px 15px;text-align:left;">Predicted Label</th>'
            '<th style="padding:13px 15px;text-align:center;">Result</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            '</table></div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)

    render_predictions_table(display_df)

