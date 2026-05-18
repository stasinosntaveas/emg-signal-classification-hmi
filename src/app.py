import streamlit as st
import os
import numpy as np
import scipy.io
import joblib
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import skew
from sklearn.metrics import confusion_matrix
import shap

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="EMG Gesture Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================
# CLINICAL SIGNAL INTELLIGENCE — CSS
# ========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --c-primary:       #006b5f;
    --c-primary-lt:    #2dd4bf;
    --c-secondary:     #0058be;
    --c-surface:       #f8f9ff;
    --c-surface-low:   #eff4ff;
    --c-surface-high:  #dce9ff;
    --c-on-surface:    #0b1c30;
    --c-on-variant:    #3c4a46;
    --c-outline:       #6b7a76;
    --c-outline-lt:    #bacac5;
    --c-error:         #ba1a1a;
    --f-head:          'Manrope', sans-serif;
    --f-body:          'Inter', sans-serif;
    --f-mono:          'JetBrains Mono', monospace;
    --r:               8px;
}

html, body, [class*="css"],
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > .main > .block-container,
.main, .block-container {
    font-family: var(--f-body) !important;
    background-color: var(--c-surface) !important;
    color: var(--c-on-surface) !important;
}

section[data-testid="stSidebar"] {
    background-color: #0b1c30 !important;
    border-right: 1px solid #1e3a52;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #eaf1ff !important;
    font-size: 15px !important;
}
section[data-testid="stSidebar"] .stMarkdown hr { border-color: #1e3a52; }

h1, h2, h3 { font-family: var(--f-head) !important; color: var(--c-on-surface) !important; }

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--c-outline-lt);
    border-radius: var(--r);
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] > div {
    font-family: var(--f-body) !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--c-on-variant) !important;
}
[data-testid="stMetricValue"] > div {
    font-family: var(--f-mono) !important;
    color: var(--c-primary) !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: var(--c-surface-low);
    border: 1px solid var(--c-outline-lt);
    border-radius: var(--r);
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--f-body) !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    color: var(--c-on-variant) !important;
    border-radius: 6px !important;
    padding: 8px 22px !important;
}
.stTabs [aria-selected="true"] {
    background-color: var(--c-primary) !important;
    color: white !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: white !important;
    border: 2px dashed var(--c-outline-lt) !important;
    border-radius: 12px !important;
}

.stButton > button {
    background-color: var(--c-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-family: var(--f-body) !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
.stButton > button:hover { background-color: #005249 !important; }

[data-testid="stNumberInput"] input,
[data-baseweb="select"] div {
    font-family: var(--f-mono) !important;
    font-size: 14px !important;
}

[data-testid="stSpinner"] p { font-family: var(--f-body) !important; }
.stRadio label { font-family: var(--f-body) !important; font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# ========================
# CONSTANTS
# ========================
CHANNEL_MUSCLE_MAP = {
    1:  "Flexor Digitorum Superficialis",
    2:  "Flexor Carpi Ulnaris",
    3:  "Flexor Carpi Radialis",
    4:  "Palmaris Longus",
    5:  "Extensor Digitorum",
    6:  "Extensor Carpi Ulnaris",
    7:  "Extensor Carpi Radialis",
    8:  "Brachioradialis",
    9:  "Pronator Teres",
    10: "Biceps Brachii",
    11: "Triceps Brachii",
    12: "Anterior Deltoid",
}

GESTURE_NAME_MAP = {
    # Ninapro DB2 — Exercise B (17 basic movements of the hand and wrist)
    1:  "Thumb Up",
    2:  "Extension of Index & Middle, Flexion of Others",
    3:  "Flexion of Ring & Little, Extension of Others",
    4:  "Thumb Opposing Base of Little Finger",
    5:  "Abduction of All Fingers",
    6:  "Fingers Flexed Together in Fist",
    7:  "Pointing Index",
    8:  "Adduction of Extended Fingers",
    9:  "Wrist Supination (toward Thumb)",
    10: "Wrist Pronation (toward Little Finger)",
    11: "Wrist Supination — Alt Axis",
    12: "Wrist Pronation — Alt Axis",
    13: "Wrist Flexion",
    14: "Wrist Extension",
    15: "Wrist Radial Deviation",
    16: "Wrist Ulnar Deviation",
    17: "Wrist Extension with Closed Hand",
}

N_CH   = 12
N_FEAT = 9

C = {
    "primary":    "#006b5f",
    "primary_lt": "#2dd4bf",
    "secondary":  "#0058be",
    "surface":    "#f8f9ff",
    "surface_lo": "#eff4ff",
    "surface_hi": "#dce9ff",
    "on_surface": "#0b1c30",
    "on_variant": "#3c4a46",
    "outline":    "#bacac5",
    "error":      "#ba1a1a",
    "white":      "#ffffff",
    "navy":       "#0b1c30",
}

# ========================
# MODEL LOADING
# ========================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

@st.cache_resource
def load_models():
    scaler = joblib.load(os.path.join(RESULTS_DIR, "scaler.pkl"))
    svm    = joblib.load(os.path.join(RESULTS_DIR, "svm.pkl"))
    knn    = joblib.load(os.path.join(RESULTS_DIR, "knn.pkl"))
    rf     = joblib.load(os.path.join(RESULTS_DIR, "rf.pkl"))
    return scaler, svm, knn, rf

scaler, svm_model, knn_model, rf_model = load_models()

MODELS = {
    "SVM":           svm_model,
    "KNN":           knn_model,
    "Random Forest": rf_model,
}

# ========================
# FEATURE EXTRACTION
# ========================
def extract_features(window):
    features = []
    for ch in range(window.shape[1]):
        s = window[:, ch]
        mav = np.mean(np.abs(s))
        rms = np.sqrt(np.mean(s ** 2))
        wl  = np.sum(np.abs(np.diff(s)))
        zcr = np.sum(np.diff(np.sign(s)) != 0)
        ssc = np.sum(np.diff(np.sign(np.diff(s))) != 0)
        var = np.var(s)
        sk  = skew(s)
        fft_v   = np.abs(np.fft.rfft(s))
        freqs   = np.fft.rfftfreq(len(s))
        fft_sum = np.sum(fft_v) + 1e-10
        mf      = np.sum(freqs * fft_v) / fft_sum
        cs      = np.cumsum(fft_v)
        medf    = freqs[np.searchsorted(cs, cs[-1] / 2)]
        features.extend([mav, rms, wl, zcr, ssc, var, sk, mf, medf])
    return features

def channel_importance_from_shap(shap_row):
    return {
        ch + 1: float(np.sum(np.abs(shap_row[ch * N_FEAT:(ch + 1) * N_FEAT])))
        for ch in range(N_CH)
    }

def gesture_label(g):
    g = int(g)
    return f"G{g} — {GESTURE_NAME_MAP.get(g, f'Gesture {g}')}"

# ========================
# PLOTLY HELPERS
# ========================
_base_font = dict(family="Inter, sans-serif", size=14, color=C["on_surface"])

def _layout(title="", height=None):
    d = dict(
        title=dict(text=title, font=dict(family="Manrope, sans-serif", size=15, color=C["on_surface"])),
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["white"],
        font=_base_font,
        margin=dict(l=44, r=20, t=48, b=40),
        xaxis=dict(showgrid=True, gridcolor=C["surface_hi"], linecolor=C["outline"],
                   tickfont=dict(family="JetBrains Mono, monospace", size=13)),
        yaxis=dict(showgrid=True, gridcolor=C["surface_hi"], linecolor=C["outline"]),
    )
    if height:
        d["height"] = height
    return d

def fig_confusion_matrix(cm, class_labels, model_name):
    names = [gesture_label(l) for l in class_labels]
    text  = [[f"{v:.2f}" for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred: {n}" for n in names],
        y=[f"True: {n}" for n in names],
        colorscale=[[0, C["surface_lo"]], [1, C["primary"]]],
        zmin=0, zmax=1,
        text=text, texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono, monospace", size=16, color=C["on_surface"]),
        showscale=True,
        colorbar=dict(tickfont=dict(family="JetBrains Mono, monospace", size=11)),
    ))
    layout = _layout(f"Normalized Confusion Matrix — {model_name}", height=380)
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)
    return fig

def fig_shap_bars(ch_imp, top_ch):
    chs    = sorted(ch_imp)
    values = [ch_imp[c] for c in chs]
    labels = [f"CH-{c:02d}" for c in chs]
    colors = [C["primary_lt"] if c == top_ch else "#adc6ff" for c in chs]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        customdata=[CHANNEL_MUSCLE_MAP[c] for c in chs],
        hovertemplate="<b>%{y}</b><br>%{customdata}<br>|SHAP| = %{x:.5f}<extra></extra>",
    ))
    layout = _layout("SHAP Channel Importance", height=420)
    layout["xaxis"]["title"] = dict(text="Summed |SHAP|", font=dict(size=14))
    layout["yaxis"]["title"] = dict(text="Channel",       font=dict(size=14))
    layout["yaxis"]["tickfont"] = dict(family="JetBrains Mono, monospace", size=14)
    fig.update_layout(**layout)
    return fig

def fig_raw_signal(raw_window, ch_idx):
    ch_num = ch_idx + 1
    muscle = CHANNEL_MUSCLE_MAP[ch_num]
    sig    = raw_window[:, ch_idx]
    t      = np.arange(len(sig))
    fig = go.Figure(go.Scatter(
        x=t, y=sig, mode="lines",
        line=dict(color=C["primary_lt"], width=1.5),
        name=f"CH-{ch_num:02d}",
        hovertemplate="t=%{x}<br>Amp=%{y:.5f}<extra></extra>",
    ))
    layout = _layout(f"Raw EMG — CH-{ch_num:02d} · {muscle}", height=270)
    layout["xaxis"]["title"] = dict(text="Sample index", font=dict(size=14))
    layout["yaxis"]["title"] = dict(text="Amplitude",    font=dict(size=14))
    fig.update_layout(**layout)
    return fig

# ========================
# HTML COMPONENTS
# ========================
def card(content_html, padding="20px"):
    return (
        f'<div style="background:white;border:1px solid {C["outline"]};'
        f'border-radius:8px;padding:{padding};">{content_html}</div>'
    )

def chip(label, color, bg):
    return (
        f'<span style="background:{bg};color:{color};border:1px solid {color};'
        f'border-radius:999px;padding:3px 14px;font-family:Inter;'
        f'font-size:12px;font-weight:700;">{label}</span>'
    )

def mono(text, size="14px", color=None):
    color = color or C["on_surface"]
    return f'<span style="font-family:JetBrains Mono,monospace;font-size:{size};color:{color};">{text}</span>'

def label_caps(text):
    return (
        f'<p style="font-family:Inter,sans-serif;font-size:12px;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.05em;color:{C["on_variant"]};margin:0 0 4px 0;">'
        f'{text}</p>'
    )

def per_gesture_bar(g, acc):
    ok  = acc >= 0.80
    bar = C["primary_lt"] if ok else C["error"]
    bg  = "#e8fdf9"       if ok else "#fff1f1"
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'background:{bg};border-left:3px solid {bar};padding:8px 12px;'
        f'border-radius:0 6px 6px 0;margin-bottom:6px;">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:14px;">{gesture_label(g)}</span>'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:14px;font-weight:700;">{acc:.1%}</span>'
        f'</div>'
    )

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.markdown(
        f'<p style="font-family:Manrope,sans-serif;font-size:20px;font-weight:700;'
        f'color:{C["primary_lt"]};margin:8px 0 2px 0;">EMG Gesture HMI</p>'
        f'<p style="font-family:Inter,sans-serif;font-size:10px;color:{C["outline"]};'
        f'text-transform:uppercase;letter-spacing:0.1em;margin:0 0 12px 0;">'
        f'Clinical Signal Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    model_name  = st.selectbox("Model", list(MODELS.keys()))
    st.markdown("**Signal Windowing**")
    window_size = st.slider("Window Size", 50, 1000, 200, 10)
    window_size = st.number_input("Override Window Size", 50, 1000, window_size, 10)
    step_size   = st.slider("Step Size",   10,  500, 100, 10)
    step_size   = st.number_input("Override Step Size",   10,  500, step_size,   10)

model = MODELS[model_name]

# ========================
# HEADER
# ========================
st.markdown(
    f'<h1 style="font-family:Manrope,sans-serif;font-size:30px;font-weight:700;'
    f'color:{C["on_surface"]};margin-bottom:2px;">Explainable EMG Gesture Dashboard</h1>'
    f'<p style="font-family:Inter,sans-serif;font-size:13px;color:{C["on_variant"]};">'
    f'Ninapro DB2 &middot; 12-channel forearm EMG &middot; ML classification with SHAP explainability</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ========================
# FILE UPLOAD
# ========================
uploaded_file = st.file_uploader(
    "Upload Ninapro DB2 `.mat` file",
    type=["mat"],
    help="Expected keys: `emg` (N×12) and `restimulus` (N,)",
)

if uploaded_file is not None:

    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("file_key") != file_key:
        with st.spinner("Loading .mat file…"):
            data       = scipy.io.loadmat(uploaded_file)
            emg_raw    = data["emg"]
            labels_raw = data["restimulus"].flatten()
        unique = np.unique(labels_raw)
        unique = sorted(unique[unique > 0].tolist())
        st.session_state.update({
            "file_key":           file_key,
            "emg_raw":            emg_raw,
            "labels_raw":         labels_raw,
            "available_gestures": unique,
            "shap_values":        None,
            "shap_proc_key":      None,
        })

    emg_raw            = st.session_state["emg_raw"]
    labels_raw         = st.session_state["labels_raw"]
    available_gestures = st.session_state["available_gestures"]

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Gesture Filter**")
        default_sel = [g for g in available_gestures if g in (6, 17)] or available_gestures[:2]
        selected_gestures = st.multiselect(
            "Active Gestures",
            options=available_gestures,
            default=default_sel,
            format_func=gesture_label,
        )
        st.markdown("---")
        st.markdown(
            f'<p style="font-size:12px;color:{C["outline"]};">'
            f'Model: {mono(model_name, "12px", C["primary_lt"])}<br>'
            f'Window: {mono(str(window_size), "12px", C["primary_lt"])} &nbsp;|&nbsp; '
            f'Step: {mono(str(step_size), "12px", C["primary_lt"])}</p>',
            unsafe_allow_html=True,
        )

    if not selected_gestures:
        st.warning("Select at least one gesture in the sidebar.")
        st.stop()

    selected_set = set(int(g) for g in selected_gestures)
    proc_key     = f"{file_key}_{window_size}_{step_size}_{sorted(selected_set)}"

    if st.session_state.get("proc_key") != proc_key:
        st.session_state["shap_values"]   = None
        st.session_state["shap_proc_key"] = None

    X_raw_list, X_feat_list, y_list = [], [], []
    for i in range(0, len(emg_raw) - window_size, step_size):
        win          = emg_raw[i:i + window_size]
        label_window = labels_raw[i:i + window_size]
        label        = int(np.bincount(label_window).argmax())
        if label in selected_set:
            X_raw_list.append(win)
            X_feat_list.append(extract_features(win))
            y_list.append(label)

    if not X_feat_list:
        st.error("No windows matched the selected gestures. Try different parameters.")
        st.stop()

    X_raw = np.array(X_raw_list)
    X     = np.array(X_feat_list)
    y     = np.array(y_list)
    X_sc  = scaler.transform(X)
    preds = model.predict(X_sc)
    acc   = float(np.mean(preds == y))
    class_labels = sorted(selected_set)

    st.session_state["proc_key"] = proc_key

    # ── SHAP (RF only) ───────────────────────────────────────────────
    shap_ok = model_name == "Random Forest"
    if shap_ok and st.session_state.get("shap_proc_key") != proc_key:
        with st.spinner("Computing SHAP values for all samples (one-time, please wait)…"):
            explainer = shap.TreeExplainer(rf_model)
            sv        = explainer.shap_values(X_sc)
            if isinstance(sv, list):
                # list of arrays → one per class
                sv_combined = np.mean([np.abs(s) for s in sv], axis=0)
            else:
                if sv.ndim == 3:
                    # (n_samples, n_features, n_classes)
                    sv_combined = np.mean(np.abs(sv), axis=2)
                else:
                    # (n_samples, n_features)
                    sv_combined = np.abs(sv)
            st.session_state["shap_values"]   = sv_combined
            st.session_state["shap_proc_key"] = proc_key

    shap_values = st.session_state.get("shap_values")

    # ── TABS ────────────────────────────────────────────────────────
    tab_ov, tab_si, tab_pt = st.tabs(["Overview", "Sample Inspector", "Predictions Table"])

    # ═══════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ═══════════════════════════════════════════════
    with tab_ov:
        n_ok  = int(np.sum(preds == y))
        n_err = int(np.sum(preds != y))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",      f"{acc:.1%}")
        m2.metric("Total Windows", f"{len(y):,}")
        m3.metric("Correct",       f"{n_ok:,}")
        m4.metric("Wrong",         f"{n_err:,}")

        st.markdown("")
        col_cm, col_sum = st.columns([3, 2])

        with col_cm:
            cm = confusion_matrix(y, preds, normalize="true", labels=class_labels)
            st.plotly_chart(fig_confusion_matrix(cm, class_labels, model_name), width='stretch')

        with col_sum:
            shap_badge = (
                f'<span style="color:{C["primary_lt"]};">&#10003; Active</span>'
                if shap_ok else
                f'<span style="color:{C["outline"]};">&mdash; RF only</span>'
            )
            st.markdown(card(
                label_caps("Session Summary") +
                f'<p style="font-family:JetBrains Mono,monospace;font-size:14px;margin:10px 0 4px 0;">Model &nbsp;&nbsp;: {model_name}</p>'
                f'<p style="font-family:JetBrains Mono,monospace;font-size:14px;margin:0 0 4px 0;">Gestures : {len(class_labels)}</p>'
                f'<p style="font-family:JetBrains Mono,monospace;font-size:14px;margin:0 0 4px 0;">Win/Step : {window_size}/{step_size}</p>'
                f'<p style="font-family:JetBrains Mono,monospace;font-size:14px;margin:0 0 4px 0;">Features : {X.shape[1]}</p>'
                f'<p style="font-family:JetBrains Mono,monospace;font-size:14px;margin:0 0 0 0;">SHAP &nbsp;&nbsp;&nbsp;&nbsp;: {shap_badge}</p>'
            ), unsafe_allow_html=True)

            st.markdown("")
            st.markdown(
                f'<p style="font-family:Inter,sans-serif;font-size:12px;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.06em;color:{C["on_variant"]};">'
                f'Per-Gesture Accuracy</p>',
                unsafe_allow_html=True,
            )
            for g in class_labels:
                mask = y == g
                if mask.sum() > 0:
                    g_acc = float(np.mean(preds[mask] == y[mask]))
                    st.markdown(per_gesture_bar(g, g_acc), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    # TAB 2 — SAMPLE INSPECTOR
    # ═══════════════════════════════════════════════
    with tab_si:
        if not shap_ok:
            st.info("Switch to **Random Forest** in the sidebar to enable SHAP explainability.")

        col_sel, col_shap = st.columns([1, 2])

        with col_sel:
            sample_idx = int(st.number_input(
                "Sample Window #", min_value=1, max_value=len(y), value=1, step=1
            )) - 1

            true_g  = int(y[sample_idx])
            pred_g  = int(preds[sample_idx])
            correct = true_g == pred_g
            s_color = C["primary"] if correct else C["error"]
            s_bg    = "#e8fdf9"    if correct else "#fff1f1"
            s_text  = "CORRECT"    if correct else "WRONG"

            st.markdown(card(
                label_caps("Window ID") +
                f'<p style="font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;'
                f'color:{C["primary"]};margin:2px 0 14px 0;">WIN-{sample_idx+1:05d}</p>'
                f'<hr style="border:none;border-top:1px solid {C["surface_hi"]};margin:0 0 12px 0;">'
                + label_caps("True Label") +
                f'<p style="font-family:JetBrains Mono,monospace;font-size:16px;margin:2px 0 10px 0;">{gesture_label(true_g)}</p>'
                + label_caps("Predicted") +
                f'<p style="font-family:JetBrains Mono,monospace;font-size:16px;margin:2px 0 14px 0;">{gesture_label(pred_g)}</p>'
                + chip(s_text, s_color, s_bg)
            ), unsafe_allow_html=True)

        with col_shap:
            if shap_ok and shap_values is not None:
                ch_imp     = channel_importance_from_shap(shap_values[sample_idx])
                top_ch     = max(ch_imp, key=ch_imp.get)
                top_muscle = CHANNEL_MUSCLE_MAP[top_ch]

                st.markdown(
                    f'<div style="background:{C["surface_hi"]};border-left:4px solid {C["primary"]};'
                    f'border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:14px;">'
                    + label_caps("Primary Driver") +
                    f'<p style="font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;'
                    f'color:{C["primary"]};margin:4px 0 6px 0;">CH-{top_ch:02d} &middot; {top_muscle}</p>'
                    f'<p style="font-family:Inter,sans-serif;font-size:13px;color:{C["on_surface"]};margin:0;">'
                    f'The classifier relied most on <b>CH-{top_ch:02d} &mdash; {top_muscle}</b> for this prediction.</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig_shap_bars(ch_imp, top_ch), width='stretch')
            else:
                st.markdown(
                    f'<div style="background:{C["surface_lo"]};border:1px dashed {C["outline"]};'
                    f'border-radius:8px;padding:40px;text-align:center;color:{C["on_variant"]};">'
                    f'<p style="font-family:Manrope,sans-serif;font-size:16px;font-weight:600;">SHAP not available</p>'
                    f'<p style="font-family:Inter,sans-serif;font-size:13px;">'
                    f'Select <b>Random Forest</b> in the sidebar to enable channel importance.</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        raw_col_title, raw_col_ch = st.columns([2, 2])
        with raw_col_title:
            st.markdown(
                f'<p style="font-family:Manrope,sans-serif;font-size:15px;font-weight:600;'
                f'color:{C["on_surface"]};">Raw EMG Signal</p>',
                unsafe_allow_html=True,
            )

        ch_idx_default = (top_ch - 1) if (shap_ok and shap_values is not None) else 0

        with raw_col_ch:
            ch_idx_view = st.selectbox(
                "Channel",
                options=list(range(N_CH)),
                index=ch_idx_default,
                format_func=lambda c: f"CH-{c+1:02d}  ·  {CHANNEL_MUSCLE_MAP[c+1]}",
            )

        st.plotly_chart(fig_raw_signal(X_raw[sample_idx], ch_idx_view), width='stretch')

        if shap_ok and shap_values is not None:
            with st.expander("View feature stats for this window"):
                feat_names = [
                    f"CH-{ch+1:02d}_{fname}"
                    for ch in range(N_CH)
                    for fname in ["MAV","RMS","WL","ZCR","SSC","VAR","SK","MeanF","MedianF"]
                ]
                feat_df = pd.DataFrame({
                    "Feature": feat_names,
                    "Value":   X[sample_idx].tolist(),
                    "|SHAP|":  shap_values[sample_idx].tolist(),
                })
                feat_df["Channel"] = feat_df["Feature"].str[:5]
                st.dataframe(
                    feat_df.sort_values("|SHAP|", ascending=False).head(20).reset_index(drop=True),
                    width='stretch',
                )

    # ═══════════════════════════════════════════════
    # TAB 3 — PREDICTIONS TABLE
    # ═══════════════════════════════════════════════
    with tab_pt:
        df = pd.DataFrame({
            "Window":     [f"WIN-{i+1:05d}" for i in range(len(y))],
            "True Label": [gesture_label(v) for v in y],
            "Predicted":  [gesture_label(v) for v in preds],
            "Result":     ["✓" if t == p else "✗" for t, p in zip(y, preds)],
        })

        fc1, _fc2, _fc3 = st.columns([2, 1, 1])
        with fc1:
            filt = st.radio("Show", ["All", "Correct", "Wrong"], horizontal=True)

        ddf = df if filt == "All" else df[df["Result"] == ("✓" if filt == "Correct" else "✗")]

        st.markdown(
            f'<p style="font-family:Inter,sans-serif;font-size:13px;color:{C["on_variant"]};">'
            f'Showing <b>{len(ddf):,}</b> of <b>{len(df):,}</b> windows</p>',
            unsafe_allow_html=True,
        )

        rows_html = ""
        for _, row in ddf.iterrows():
            ok = row["Result"] == "✓"
            bg = "#e8fdf9" if ok else "#fff1f1"
            ic = C["primary"] if ok else C["error"]
            rows_html += (
                f'<tr>'
                f'<td style="padding:9px 16px;font-family:JetBrains Mono,monospace;font-size:12px;border-bottom:1px solid {C["surface_hi"]};">{row["Window"]}</td>'
                f'<td style="padding:9px 16px;font-family:Inter,sans-serif;font-size:13px;border-bottom:1px solid {C["surface_hi"]};">{row["True Label"]}</td>'
                f'<td style="padding:9px 16px;font-family:Inter,sans-serif;font-size:13px;border-bottom:1px solid {C["surface_hi"]};">{row["Predicted"]}</td>'
                f'<td style="padding:9px 16px;text-align:center;border-bottom:1px solid {C["surface_hi"]};">'
                f'<span style="background:{bg};color:{ic};border:1px solid {ic};border-radius:999px;padding:2px 12px;font-family:Inter,sans-serif;font-size:11px;font-weight:700;">{row["Result"]}</span></td>'
                f'</tr>'
            )

        th = (
            f'background:{C["navy"]};padding:12px 16px;text-align:left;'
            f'font-family:Inter,sans-serif;font-size:11px;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.06em;color:{C["outline"]};'
        )
        st.markdown(
            f'<div style="max-height:560px;overflow-y:auto;border-radius:8px;border:1px solid {C["outline"]};">'
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr style="position:sticky;top:0;z-index:1;">'
            f'<th style="{th}">Window</th>'
            f'<th style="{th}">True Label</th>'
            f'<th style="{th}">Predicted</th>'
            f'<th style="{th};text-align:center;">Result</th>'
            f'</tr></thead>'
            f'<tbody style="background:white;">{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;'
        f'padding:80px 40px;background:white;border:1px solid {C["outline"]};'
        f'border-radius:12px;text-align:center;margin-top:24px;">'
        f'<p style="font-size:52px;margin:0 0 20px 0;">&#129504;</p>'
        f'<p style="font-family:Manrope,sans-serif;font-size:22px;font-weight:700;'
        f'color:{C["on_surface"]};margin:0 0 10px 0;">Upload a Ninapro DB2 .mat file to begin</p>'
        f'<p style="font-family:Inter,sans-serif;font-size:14px;color:{C["on_variant"]};'
        f'max-width:520px;line-height:1.6;">'
        f'The dashboard will window the signal, extract 108 features per window, '
        f'classify gestures with your chosen model, and compute SHAP channel importance '
        f'for Random Forest predictions.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )