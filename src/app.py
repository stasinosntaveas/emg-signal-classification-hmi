import streamlit as st
import os
import numpy as np
import scipy.io
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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

        features.extend([mav, rms, wl])

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

        if label != 0:
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

    class_labels = list(range(1, 18))

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
    # SAMPLE OUTPUT
    # ========================
    st.subheader("Sample Predictions")
    st.write(preds[:30])

