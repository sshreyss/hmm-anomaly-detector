"""
Interactive Web Dashboard for the Real-Time HMM Anomaly Detector.
Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.data_generator import (
    generate_normal_sensor_data,
    generate_test_stream_with_anomalies,
)
from src.model import select_best_hmm
from src.detector import (
    compute_window_log_likelihoods,
    determine_threshold,
    detect_anomalies_realtime,
    evaluate_predictions,
)

st.set_page_config(
    page_title="HMM Anomaly Detector",
    page_icon="📈",
    layout="wide",
)

st.title("Real-Time Machine Sensor Anomaly Detector (HMM)")
st.markdown(
    "Train a **Gaussian Hidden Markov Model** on normal operating data, select the "
    "optimal hidden states via **BIC**, and flag anomalies in a continuous test stream "
    "using a real-time sliding window."
)

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Simulation & Model Controls")

train_cycles = st.sidebar.slider("Training Operating Cycles", min_value=5, max_value=20, value=12)
test_cycles = st.sidebar.slider("Test Stream Cycles", min_value=3, max_value=10, value=5)
max_states = st.sidebar.slider("Max Hidden States to Evaluate", min_value=4, max_value=8, value=6)
window_size = st.sidebar.slider("Sliding Window Size (samples)", min_value=5, max_value=30, value=15)
sigma_multiplier = st.sidebar.slider(
    "Threshold Sensitivity (σ multiplier)",
    min_value=1.5,
    max_value=5.0,
    value=3.0,
    step=0.1,
    help="Lower values detect more anomalies (higher sensitivity); higher values reduce false alarms.",
)

state_candidates = list(range(2, max_states + 1))

# -----------------------------------------------------------------------------
# Run Pipeline (Cached for Smooth Slider Updates)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training Gaussian HMM models...")
def train_models(n_train_cycles: int, candidates: tuple):
    train_data = generate_normal_sensor_data(n_cycles=n_train_cycles, seed=42)
    best_model, results = select_best_hmm(train_data, list(candidates))
    return train_data, best_model, results


train_data, best_model, selection_results = train_models(
    train_cycles, tuple(state_candidates)
)
test_data, ground_truth = generate_test_stream_with_anomalies(
    n_cycles=test_cycles, seed=99
)

# Compute training window likelihoods & anomaly threshold
_, train_scores = compute_window_log_likelihoods(
    best_model, train_data, window_size=window_size, step_size=1
)
threshold = determine_threshold(train_scores, sigma_multiplier=sigma_multiplier)

# Run sliding-window detection on test stream
window_ends, test_scores, predicted_flags = detect_anomalies_realtime(
    best_model, test_data, threshold=threshold, window_size=window_size, step_size=1
)
metrics = evaluate_predictions(ground_truth, predicted_flags)

# -----------------------------------------------------------------------------
# Top KPI Metrics
# -----------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Selected States (BIC)", f"K = {best_model.n_components}")
col2.metric("Anomaly Threshold", f"{threshold:.2f}")
col3.metric("Precision", f"{metrics['precision']:.2%}")
col4.metric("Recall", f"{metrics['recall']:.2%}")
col5.metric("F1-Score", f"{metrics['f1_score']:.2%}")

st.divider()

# -----------------------------------------------------------------------------
# Main Anomaly Detection Plot
# -----------------------------------------------------------------------------
st.subheader("Real-Time Sensor Stream & Anomaly Detection")

time_steps = np.arange(len(test_data))
fig_det, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

ax1.plot(time_steps, test_data[:, 0], color="#1f77b4", linewidth=1.2, label="Raw Sensor Reading")
ax1.fill_between(
    time_steps,
    test_data[:, 0].min() - 5,
    test_data[:, 0].max() + 5,
    where=predicted_flags == 1,
    color="red",
    alpha=0.25,
    label="Detected Anomaly Region",
)
ax1.fill_between(
    time_steps,
    test_data[:, 0].min() - 5,
    test_data[:, 0].max() + 5,
    where=ground_truth == 1,
    color="orange",
    alpha=0.15,
    hatch="//",
    edgecolor="darkorange",
    label="Ground Truth Anomaly",
)
ax1.set_ylabel("Sensor Reading (Vibration)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

ax2.plot(window_ends, test_scores, color="#2ca02c", linewidth=1.2, label="Window Log-Likelihood")
ax2.axhline(
    threshold,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Threshold ({threshold:.2f})",
)
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Log-Likelihood")
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_det)

# -----------------------------------------------------------------------------
# Model Selection & Confusion Matrix
# -----------------------------------------------------------------------------
st.divider()
left_col, right_col = st.columns([1.4, 1])

with left_col:
    st.subheader("Model Selection (AIC & BIC)")
    states = list(selection_results.keys())
    aics = [selection_results[k]["aic"] for k in states]
    bics = [selection_results[k]["bic"] for k in states]

    fig_bic, ax = plt.subplots(figsize=(7, 4))
    ax.plot(states, bics, marker="o", linewidth=2, label="BIC (Lower is Better)")
    ax.plot(states, aics, marker="s", linestyle="--", linewidth=1.8, label="AIC")
    ax.axvline(
        best_model.n_components,
        color="red",
        linestyle=":",
        label=f"Selected (K={best_model.n_components})",
    )
    ax.set_xlabel("Number of Hidden States (K)")
    ax.set_ylabel("Information Criterion")
    ax.set_xticks(states)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig_bic)

with right_col:
    st.subheader("Model Summary & Confusion Matrix")

    summary_df = pd.DataFrame.from_dict(selection_results, orient="index")
    summary_df.index.name = "States (K)"
    summary_df.columns = ["Log-Likelihood", "AIC", "BIC"]
    st.dataframe(summary_df.style.format("{:.2f}"), use_container_width=True)

    st.markdown("**Confusion Matrix**")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"],
    )
    st.dataframe(cm_df, use_container_width=True)

    means_str = ", ".join(f"{m:.2f}" for m in np.sort(best_model.means_.flatten()))
    st.info(f"**Learned Operating Regime Means:** {means_str}")