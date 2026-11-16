"""
Global (aggregate) KernelSHAP feature-importance analysis for the
production model, run OFFLINE across a sample of the held-out test set.

This is distinct from app/models/explain.py, which explains a single
LIVE prediction (today's window) for the on-demand dashboard endpoint.
For a thesis-style "relative significance of hydrometeorological
parameters" analysis, a single day isn't representative -- this script
instead explains a random sample of held-out test sequences and
averages |SHAP value| per feature to rank global importance, using the
exact same 16-feature grouping / baseline / vectorized-predict design
as the live endpoint so the two stay methodologically consistent.

Run from the PROJECT ROOT:

    python training/shap_global_analysis.py [--model lstm] [--n 60]

Writes:
    shap_global_importance_{model}.csv
    shap_global_importance_{model}.png
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import (
    load_flood_dataset, preprocess_flood_data, create_sequences_encdec,
    WINDOW_SIZE, HORIZON,
)

MODEL_FILES = {
    "lstm": "flood_lstm_14day_encdec_model.h5",
    "gru": "flood_gru_14day_encdec_model.h5",
    "cnn": "flood_cnn_14day_encdec_model.h5",
}

FEATURE_INFO = {
    "rain_3h":               {"label": "3-Day Cumulative Rainfall", "unit": "mm"},
    "rain_6h":                "6-Day Cumulative Rainfall",
    "rain_12h":               "12-Day Cumulative Rainfall",
    "rain_24h":               "24-Day Cumulative Rainfall",
    "rain_7d_avg":            "7-Day Rainfall Average",
    "api_5d":                 "Antecedent Precipitation Index (k=0.85)",
    "typhoon_signal":         "Typhoon Signal Level (PAGASA)",
    "antecedent_moisture":    "Antecedent Moisture Proxy",
    "month_sin":              "Seasonal Encoding (sin)",
    "month_cos":              "Seasonal Encoding (cos)",
    "prev_flood":             "Previous-Day Flood Occurrence",
    "typhoon_max_wind_kph":   "Max Wind Speed",
    "soil_moisture_mean":     "Soil Moisture",
    "pressure_msl_mean":      "Sea-Level Pressure",
    "surface_pressure_mean":  "Surface Pressure",
    "wind_gusts_10m_max":     "Wind Gusts (max)",
}


def label_for(name):
    info = FEATURE_INFO.get(name, name)
    return info["label"] if isinstance(info, dict) else info


def build_test_sequences():
    """Mirrors common.py's prepare_training_data() but returns the RAW
    (pre-scaled) past windows too, and skips writing scaler/metadata
    files again (they already exist from training)."""
    df, csv_path, used_backfilled = load_flood_dataset()
    df_processed, features, target_col, col_moisture, col_signal, enriched_used = preprocess_flood_data(df)

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
    X_past_raw_seq, _, _ = create_sequences_encdec(
        X_raw, future_col_idx, y_raw, window=WINDOW_SIZE, horizon=HORIZON
    )

    n = len(X_past_seq)
    split = int(n * 0.8)  # same 80/20, no shuffle, as common.py's train_test_split(shuffle=False)
    return {
        "X_past_test": X_past_seq[split:],
        "X_future_test": X_future_seq[split:],
        "X_past_raw_test": X_past_raw_seq[split:],
        "y_test": y_seq[split:],
        "feature_order": features,
    }


def day1_probs_from_masks(model, mask_batch, past_window, future_window):
    n = mask_batch.shape[0]
    mask_expanded = mask_batch[:, np.newaxis, :]
    past_batch = mask_expanded * past_window[np.newaxis, :, :]
    future_batch = np.repeat(future_window[np.newaxis, :, :], n, axis=0)
    probs = np.asarray(model.predict([past_batch, future_batch], verbose=0))
    if probs.ndim == 3:
        return probs[:, 0, 0]
    return probs[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=list(MODEL_FILES.keys()))
    ap.add_argument("--n", type=int, default=60, help="number of test sequences to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.model.upper()} model and building test sequences...")
    model = tf.keras.models.load_model(MODEL_FILES[args.model], compile=False)
    data = build_test_sequences()
    feature_order = data["feature_order"]
    n_features = len(feature_order)
    n_test = len(data["X_past_test"])
    print(f"Held-out test sequences available: {n_test}")

    rng = np.random.RandomState(args.seed)
    n_sample = min(args.n, n_test)
    sample_idx = rng.choice(n_test, size=n_sample, replace=False)
    print(f"Sampling {n_sample} test sequences (seed={args.seed}) for global SHAP analysis...")

    background = np.zeros((1, n_features))
    all_shap = np.zeros((n_sample, n_features))

    t0 = time.time()
    for k, idx in enumerate(sample_idx):
        past_window = data["X_past_test"][idx]
        future_window = data["X_future_test"][idx]

        def f(mask_batch):
            return day1_probs_from_masks(model, np.asarray(mask_batch), past_window, future_window)

        explainer = shap.KernelExplainer(f, background)
        raw = explainer.shap_values(np.ones((1, n_features)), nsamples="auto", silent=True)
        all_shap[k] = np.asarray(raw).reshape(-1)

        if (k + 1) % 10 == 0 or (k + 1) == n_sample:
            elapsed = time.time() - t0
            print(f"  [{k + 1}/{n_sample}] done ({elapsed:.1f}s elapsed, "
                  f"{elapsed / (k + 1):.2f}s/instance)")

    mean_abs_shap = np.mean(np.abs(all_shap), axis=0)
    mean_signed_shap = np.mean(all_shap, axis=0)

    result = pd.DataFrame({
        "feature": feature_order,
        "label": [label_for(f) for f in feature_order],
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["pct_of_total_importance"] = 100 * result["mean_abs_shap"] / result["mean_abs_shap"].sum()

    out_csv = f"shap_global_importance_{args.model}.csv"
    result.to_csv(out_csv, index=False)
    print(f"\n-> Saved: {out_csv}")
    print(result[["rank", "label", "mean_abs_shap", "pct_of_total_importance"]].to_string(index=False))

    plt.figure(figsize=(9, 6))
    plot_df = result.iloc[::-1]  # smallest at bottom -> largest at top when barh
    colors = ["#1F77B4" if v >= 0 else "#D62728" for v in plot_df["mean_signed_shap"]]
    plt.barh(plot_df["label"], plot_df["mean_abs_shap"], color=colors)
    plt.xlabel("Mean |SHAP value| (impact on Day-1 flood probability)")
    plt.title(f"Global Feature Importance ({args.model.upper()}) — "
              f"n={n_sample} held-out test sequences")
    plt.tight_layout()
    out_png = f"shap_global_importance_{args.model}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {out_png}")


if __name__ == "__main__":
    main()
