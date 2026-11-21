"""
Main Entry Point for the Real-Time HMM Anomaly Detection System.
"""

import os
import numpy as np

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
from src.visualizer import plot_model_selection, plot_anomaly_detection


def main() -> None:
    WINDOW_SIZE = 15
    STEP_SIZE = 1
    STATE_CANDIDATES = [2, 3, 4, 5, 6]
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Generate normal training data and anomalous test stream
    train_data = generate_normal_sensor_data(n_cycles=12, seed=42)
    test_data, ground_truth = generate_test_stream_with_anomalies(n_cycles=5, seed=99)
    print(f"Training samples: {len(train_data)} | Test samples: {len(test_data)}\n")

    # 2. Train candidate Gaussian HMMs and select optimal state count via BIC
    print("--- Model Selection (BIC) ---")
    best_model, selection_results = select_best_hmm(train_data, STATE_CANDIDATES)
    print(f"\nSelected Model: {best_model.n_components} hidden states")
    print("Learned State Means:", np.round(best_model.means_.flatten(), 2))

    # 3. Compute training likelihood distribution & set anomaly threshold
    _, train_scores = compute_window_log_likelihoods(
        best_model, train_data, window_size=WINDOW_SIZE, step_size=STEP_SIZE
    )
    threshold = determine_threshold(train_scores, sigma_multiplier=3.0)
    print(f"\nTraining Window LL -> Mean: {train_scores.mean():.2f}, Std: {train_scores.std():.2f}")
    print(f"Anomaly Threshold (Mean - 3*Std): {threshold:.2f}")

    # 4. Run real-time sliding window anomaly detection on test stream
    window_ends, test_scores, predicted_flags = detect_anomalies_realtime(
        best_model,
        test_data,
        threshold=threshold,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
    )

    # 5. Evaluate detection performance
    metrics = evaluate_predictions(ground_truth, predicted_flags)
    print("\n--- Detection Performance ---")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    # 6. Save visualizations
    bic_plot_path = os.path.join(OUTPUT_DIR, "model_selection_bic.png")
    anomaly_plot_path = os.path.join(OUTPUT_DIR, "anomaly_detection_results.png")

    plot_model_selection(selection_results, save_path=bic_plot_path)
    plot_anomaly_detection(
        test_data=test_data,
        window_ends=window_ends,
        test_scores=test_scores,
        threshold=threshold,
        predicted_flags=predicted_flags,
        ground_truth=ground_truth,
        save_path=anomaly_plot_path,
    )
    print(f"\nSaved plots to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()