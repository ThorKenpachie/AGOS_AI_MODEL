"""
Day-1 flood classification performance using each model's OPTIMAL decision
threshold (tuned on a leak-free validation split), compared against a naive
manual-rainfall-threshold baseline (rain_24h >= 50mm).

Threshold tuning methodology
-----------------------------
train_*.py call model.fit(X_train, y_train, validation_split=0.1, ...).
Keras does NOT shuffle when a validation_split fraction is given directly
(the arrays here are already time-ordered, shuffle=False from the original
train_test_split) -- it takes the LAST 10% of the training array as
validation. This script reconstructs that exact slice, uses it to sweep
candidate thresholds (0.01-0.99) and picks the one maximizing Day-1
F1-score per model, then applies that fixed threshold to the SAME held-out
20% test set used throughout the rest of the results chapter. The
validation slice is never touched by the final reported metrics, so there
is no leakage between threshold selection and evaluation.

Manual baseline
----------------
Predicts flood = 1 iff that day's raw (unscaled) 24-hour cumulative
rainfall (rain_24h) >= 50mm, a common heavy-rainfall advisory cutoff.
AUC-ROC / Average Precision for this baseline use raw rain_24h itself as
a continuous risk score (higher rainfall = higher risk), since the rule
has no native probability output.

Run from the PROJECT ROOT:  python training/threshold_baseline_analysis.py
Writes: threshold_baseline_comparison.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix, cohen_kappa_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import (
    load_flood_dataset, preprocess_flood_data, create_sequences_encdec,
    WINDOW_SIZE, HORIZON,
)

MODEL_FILES = {
    "LSTM": "flood_lstm_14day_encdec_model.h5",
    "GRU": "flood_gru_14day_encdec_model.h5",
    "CNN": "flood_cnn_14day_encdec_model.h5",
}

# Manual baseline = the operational rule: flag a flood when the DAILY
# rainfall total on the target day reaches 50 mm. (Note: the model feature
# 'rain_24h' is a 24-DAY rolling sum on this daily dataset, so it must NOT
# be used for this baseline.)
MANUAL_RAIN_24H_THRESHOLD_MM = 50.0


def day1_probs(model, X_past, X_future):
    probs = np.asarray(model.predict([X_past, X_future], verbose=0))
    return probs[:, 0, 0] if probs.ndim == 3 else probs[:, 0]


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


def best_threshold_by_f1(y_true, probs):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, round(float(t), 2)
    return best_t, best_f1


def main():
    df, csv_path, used_backfilled = load_flood_dataset()
    df_processed, features, target_col, col_moisture, col_signal, enriched_used = preprocess_flood_data(df)

    import json
    with open("feature_metadata.json", "r") as f:
        meta = json.load(f)
    assert features == meta["feature_order"], "Feature order mismatch vs feature_metadata.json"
    future_features = meta["future_feature_order"]
    future_col_idx = [features.index(f) for f in future_features]

    scaler = joblib.load("flood_scaler.pkl")
    X_raw = df_processed[features].values
    y_raw = df_processed[target_col].astype(int).values
    X_scaled = scaler.transform(X_raw)

    X_past_seq, X_future_seq, y_seq = create_sequences_encdec(
        X_scaled, future_col_idx, y_raw, window=WINDOW_SIZE, horizon=HORIZON
    )
    # rain_24h is a PAST-window feature too (used at the most recent past
    # timestep as "today's" accumulation) -- pull the raw (unscaled) value
    # aligned to each sequence's Day-1 target date, i.e. row (i + window).
    daily_rain = df_processed["rainfall_mm"].astype(float).values
    n_seq = len(X_past_seq)
    day1_rain_24h_raw = np.array([daily_rain[i + WINDOW_SIZE] for i in range(n_seq)])

    n = len(X_past_seq)
    split = int(n * 0.8)  # same 80/20, no shuffle, as common.py
    X_past_train, X_past_test = X_past_seq[:split], X_past_seq[split:]
    X_future_train, X_future_test = X_future_seq[:split], X_future_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    rain24_test = day1_rain_24h_raw[split:]

    # Keras validation_split=0.1 on X_train takes the LAST 10% (no shuffle).
    val_start = int(len(X_past_train) * 0.9)
    X_past_val = X_past_train[val_start:]
    X_future_val = X_future_train[val_start:]
    y_val_day1 = y_train[val_start:, 0]

    y_test_day1 = y_test[:, 0]
    print(f"Validation (threshold-tuning) sequences: {len(X_past_val)}")
    print(f"Final test (reported) sequences: {len(X_past_test)}")

    results = {}
    for name, fname in MODEL_FILES.items():
        print(f"\n--- {name} ---")
        model = tf.keras.models.load_model(fname, compile=False)

        val_probs = day1_probs(model, X_past_val, X_future_val)
        best_t, val_f1 = best_threshold_by_f1(y_val_day1, val_probs)
        print(f"Optimal threshold (tuned on validation): {best_t} (val F1={val_f1:.4f})")

        test_probs = day1_probs(model, X_past_test, X_future_test)
        y_pred = (test_probs >= best_t).astype(int)

        results[name] = {
            "F1-Score": f1_score(y_test_day1, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_test_day1, y_pred),
            "Precision": precision_score(y_test_day1, y_pred, zero_division=0),
            "Recall": recall_score(y_test_day1, y_pred, zero_division=0),
            "Specificity": specificity_score(y_test_day1, y_pred),
            "AUC-ROC": roc_auc_score(y_test_day1, test_probs),
            "Average Precision": average_precision_score(y_test_day1, test_probs),
            "Optimal Threshold": best_t,
        }
        for k, v in results[name].items():
            print(f"  {k}: {v}")

        # Same model at the default 0.50 decision threshold (the threshold
        # used for Tables 3.3/3.4 and for the app's WARNING alert level).
        y_def = (test_probs >= 0.5).astype(int)
        results[f"{name} @0.50"] = {
            "F1-Score": f1_score(y_test_day1, y_def, zero_division=0),
            "Accuracy": accuracy_score(y_test_day1, y_def),
            "Precision": precision_score(y_test_day1, y_def, zero_division=0),
            "Recall": recall_score(y_test_day1, y_def, zero_division=0),
            "Specificity": specificity_score(y_test_day1, y_def),
            "Cohen's Kappa": cohen_kappa_score(y_test_day1, y_def),
            "AUC-ROC": roc_auc_score(y_test_day1, test_probs),
            "Average Precision": average_precision_score(y_test_day1, test_probs),
            "Optimal Threshold": 0.50,
        }
        results[name]["Cohen's Kappa"] = cohen_kappa_score(y_test_day1, y_pred)

    # --- Manual rainfall-threshold baseline, same final test set ---
    print("\n--- Manual Threshold (daily rainfall >= 50mm) ---")
    manual_pred = (rain24_test >= MANUAL_RAIN_24H_THRESHOLD_MM).astype(int)
    results["Manual Threshold"] = {
        "F1-Score": f1_score(y_test_day1, manual_pred, zero_division=0),
        "Accuracy": accuracy_score(y_test_day1, manual_pred),
        "Precision": precision_score(y_test_day1, manual_pred, zero_division=0),
        "Recall": recall_score(y_test_day1, manual_pred, zero_division=0),
        "Specificity": specificity_score(y_test_day1, manual_pred),
        "Cohen's Kappa": cohen_kappa_score(y_test_day1, manual_pred),
        "AUC-ROC": roc_auc_score(y_test_day1, rain24_test),
        "Average Precision": average_precision_score(y_test_day1, rain24_test),
        "Optimal Threshold": f"{MANUAL_RAIN_24H_THRESHOLD_MM:.0f} mm/day (fixed)",
    }
    for k, v in results["Manual Threshold"].items():
        print(f"  {k}: {v}")

    out = pd.DataFrame(results)
    out = out.reindex(["F1-Score", "Accuracy", "Precision", "Recall",
                        "Specificity", "Cohen's Kappa", "AUC-ROC", "Average Precision", "Optimal Threshold"])
    out.to_csv("threshold_baseline_comparison.csv")
    print("\n-> Saved: threshold_baseline_comparison.csv")
    print(out.to_string())


if __name__ == "__main__":
    main()
