"""
Re-plots the horizon-skill charts, Day-1-vs-Day-14 confusion matrices, and
SHAP global-importance charts in black-and-white/grayscale for print, using
the metrics already saved by evaluate_multihorizon() and
shap_global_analysis.py -- no model reloading or recomputation needed.

Run from the PROJECT ROOT: python training/regenerate_bw_charts.py
Overwrites: {MODEL}_horizon_skill_chart.png, {MODEL}_confusion_matrix_day1_vs_day14.png,
            shap_global_importance_{model}.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "sans-serif"

MODELS = ["LSTM", "GRU", "CNN"]


def plot_skill_chart(name):
    df = pd.read_csv(f"{name}_per_horizon_metrics.csv")
    plt.figure(figsize=(9, 5))
    plt.plot(df["horizon_day"], df["f1_score"], color="black", linestyle="-",
              marker="o", markersize=5, linewidth=1.5, label="F1-Score")
    plt.plot(df["horizon_day"], df["auc_roc"], color="black", linestyle="--",
              marker="s", markersize=5, linewidth=1.5, label="AUC-ROC")
    plt.plot(df["horizon_day"], df["accuracy"], color="black", linestyle=":",
              marker="^", markersize=5, linewidth=1.5, label="Accuracy")
    plt.xlabel("Forecast Horizon (days ahead)")
    plt.ylabel("Score")
    plt.title(f"{name} — Forecast Skill vs. Horizon Day (with forecast input)")
    plt.ylim(0, 1)
    plt.legend(facecolor="white", edgecolor="black")
    plt.grid(alpha=0.3, color="gray")
    plt.savefig(f"{name}_horizon_skill_chart.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Rewrote {name}_horizon_skill_chart.png (grayscale)")


def plot_confusion_matrices(name, horizon=14):
    df = pd.read_csv(f"{name}_per_horizon_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, day_idx, label in [(axes[0], 0, "Day 1"), (axes[1], horizon - 1, f"Day {horizon}")]:
        row = df.iloc[day_idx]
        cm = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
        im = ax.imshow(cm, cmap="Greys", vmin=0)
        for i in range(2):
            for j in range(2):
                val = cm[i][j]
                # choose text color for contrast against the grayscale cell
                text_color = "white" if val > (max(map(max, cm)) * 0.55) else "black"
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                         color=text_color, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No Flood", "Flood"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No Flood", "Flood"])
        ax.set_title(f"{name} — {label}")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix_day1_vs_day{horizon}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Rewrote {name}_confusion_matrix_day1_vs_day{horizon}.png (grayscale)")


def plot_shap_chart(model_key):
    df = pd.read_csv(f"shap_global_importance_{model_key}.csv")
    df = df.sort_values("mean_abs_shap", ascending=True)
    colors = ["black" if v >= 0 else "white" for v in df["mean_signed_shap"]]
    edgecolors = ["black"] * len(df)
    hatches = [None if v >= 0 else "///" for v in df["mean_signed_shap"]]

    plt.figure(figsize=(9, 6))
    bars = plt.barh(df["label"], df["mean_abs_shap"], color=colors,
                     edgecolor=edgecolors, linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    plt.xlabel("Mean |SHAP value| (impact on Day-1 flood probability)")
    plt.title(f"Global Feature Importance ({model_key.upper()}) — "
              f"n=200 held-out test sequences")
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="black", edgecolor="black", label="Increases flood probability"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Decreases flood probability"),
    ]
    plt.legend(handles=legend_elems, loc="lower right", facecolor="white", edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"shap_global_importance_{model_key}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Rewrote shap_global_importance_{model_key}.png (grayscale)")


if __name__ == "__main__":
    for name in MODELS:
        plot_skill_chart(name)
        plot_confusion_matrices(name)
    for key in ["lstm", "gru", "cnn"]:
        plot_shap_chart(key)
