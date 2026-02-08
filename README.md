# Hybrid-Temporal-Flow-Lightweight-Ensemble-repo
HTF-LE: Hybrid Temporal-Flow Lightweight Ensemble
This repository implements HTF-LE, a lightweight anomaly detection framework for IoT networks. The system integrates statistical temporal analysis with machine learning to identify network threats.

How the Code Works
The implementation follows a modular pipeline, processing network traffic through three distinct layers:

1. Temporal Analysis Layer (EWMA)
Instead of analyzing network packets in isolation, the code applies the Exponential Weighted Moving Average (EWMA).

Function: It smooths feature fluctuations and assigns higher weights to recent observations.

Purpose: This allows the model to detect "trend-based" anomalies and reduces noise in the input data.

2. Detection Layer (The Hybrid Ensemble)
The core of the system runs two models in parallel to capture different types of anomalies:

Lightweight Autoencoder (AE): * A neural network trained only on normal traffic.

It compresses and then reconstructs the input.

Logic: High Reconstruction Error indicates a deviation from known normal patterns.

Isolation Forest (IF): * A tree-based algorithm that isolates observations.

Logic: It identifies outliers that are "few and different" from the majority of the data points.

3. Decision Layer (Dynamic Thresholding)
The code merges the results from the detection layer:

Dynamic Threshold: Rather than using a fixed value, the code calculates a threshold based on the 85th–95th percentile of the error distribution.

Ensemble Logic: An anomaly is flagged if either the Autoencoder or the Isolation Forest identifies the traffic as suspicious.

Technical Flow
Input: Flow-level features (e.g., duration, packet size).

Step 1: Preprocessing + EWMA Smoothing.

Step 2: Feature Scaling (Min-Max).

Step 3: Parallel execution of AE and IF models.

Step 4: Decision making via the Dynamic Threshold gate.

Output: Classification (0: Normal, 1: Anomaly).
