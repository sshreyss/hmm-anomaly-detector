# Real-Time Anomaly Detector with Hidden Markov Models (HMM)

A real-time anomaly detection system built with **Gaussian Hidden Markov Models (`hmmlearn`)** to identify unusual operating patterns in sequential machine sensor data.

## Objective
Train a probabilistic model on normal machine sensor readings, establish a statistical log-likelihood anomaly threshold, and process a continuous test stream in real time using a sliding window to flag equipment anomalies before failure occurs.

## Project Structure
```text
hmm-anomaly-detector/
├── .gitignore
├── requirements.txt
├── README.md
├── main.py
├── outputs/
│   ├── model_selection_bic.png
│   └── anomaly_detection_results.png
└── src/
    ├── __init__.py
    ├── data_generator.py
    ├── model.py
    ├── detector.py
    └── visualizer.py