"""
train_eval.py
-------------
Trains two unsupervised anomaly detectors on the engineered CAN-bus
features and evaluates both against the (held-out) ground-truth labels:

  (a) Isolation Forest        - simple, fast baseline
  (b) Autoencoder (MLP-based) - reconstruction-error anomaly detector

Both models are trained ONLY on normal traffic (the realistic setting
for an IDS: you rarely have labeled attack data in advance, but you can
easily capture a clean baseline). They are then scored on a held-out
test split that contains both normal and attack traffic, and a
threshold is chosen from a validation slice of normal-only data
(e.g. the 99th percentile of the normal reconstruction/anomaly score)
so that thresholding itself never peeks at attack labels.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                              roc_auc_score, roc_curve)

FEATURE_COLS = [
    "arbitration_id", "dlc", "iat_global", "iat_same_id", "hamming_same_id",
    "id_freq_window", "id_entropy_window", "local_mean_iat", "local_std_iat",
    "payload_mean", "payload_std", "payload_max", "payload_min",
    "payload_unique_bytes",
]

RESULTS_DIR = "results"
np.random.seed(0)


def time_split(feat: pd.DataFrame, train_frac=0.5, val_frac=0.1):
    """Chronological split: train (normal-only) / val (normal-only, for
    threshold calibration) / test (normal + attack), so the model never
    trains or calibrates on attack traffic, and evaluation is causal
    (test always comes after train in time, like real deployment)."""
    feat = feat.sort_values("timestamp").reset_index(drop=True)
    n = len(feat)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = feat.iloc[:train_end]
    val_df = feat.iloc[train_end:val_end]
    test_df = feat.iloc[val_end:]

    # keep only normal traffic for train/val calibration
    train_df = train_df[train_df["label"] == 0]
    val_df = val_df[val_df["label"] == 0]

    print(f"Train (normal only): {len(train_df):,} rows")
    print(f"Val   (normal only): {len(val_df):,} rows")
    print(f"Test  (mixed)      : {len(test_df):,} rows "
          f"({test_df['label'].mean():.1%} attack)")
    return train_df, val_df, test_df


def evaluate(y_true, y_pred, scores, name):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    metrics = {
        "model": name,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_positive_rate": fpr,
        "roc_auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print(f"\n=== {name} ===")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  "
          f"FPR: {fpr:.4f}  ROC-AUC: {auc:.4f}")
    print(f"Confusion matrix [tn fp / fn tp]: [{tn} {fp} / {fn} {tp}]")
    return metrics, cm


def plot_confusion(cm, name, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Attack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, score_dict, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Isolation Forest vs Autoencoder")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_timeline(test_df, score_dict, out_path, window=(None, None)):
    """Plots anomaly score vs. time for a legible slice of the test set,
    with ground-truth attack windows shown as a thin shaded band (not a
    full-height fill, so the score line stays visible), and a log1p
    transform applied for display only to any score series whose range
    spans more than 3 orders of magnitude (e.g. autoencoder reconstruction
    error, which occasionally spikes hard during floods)."""
    t_all = test_df["timestamp"].values
    labels_all = test_df["label"].values

    t0, t1 = window
    if t0 is None:
        t0 = np.percentile(t_all, 60)
    if t1 is None:
        t1 = t0 + 15.0  # a 15s slice is enough to show a couple of bursts
    mask = (t_all >= t0) & (t_all <= t1)
    if mask.sum() < 20:  # fall back to full range if the slice is too small
        mask = np.ones_like(t_all, dtype=bool)

    fig, axes = plt.subplots(len(score_dict), 1, figsize=(11, 3.2 * len(score_dict)), sharex=True)
    if len(score_dict) == 1:
        axes = [axes]

    for ax, (name, scores) in zip(axes, score_dict.items()):
        s = np.asarray(scores)
        span = np.percentile(s, 99) / max(np.percentile(s, 1), 1e-9)
        use_log = span > 1000
        s_plot = np.log1p(np.clip(s, 0, None)) if use_log else s
        label_suffix = " (log1p)" if use_log else ""

        tm, sm, lm = t_all[mask], s_plot[mask], labels_all[mask]
        ax.plot(tm, sm, linewidth=0.6, color="steelblue")
        ylo, yhi = ax.get_ylim()
        band_top = ylo + (yhi - ylo) * 0.08
        ax.fill_between(tm, ylo, band_top, where=lm == 1, color="red", alpha=0.6,
                         step="pre", linewidth=0)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(f"anomaly score{label_suffix}")
        ax.set_title(name)

    axes[-1].set_xlabel("time (s)")
    axes[0].plot([], [], color="red", alpha=0.6, linewidth=6, label="ground-truth attack window")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Anomaly score over time vs. ground-truth attack windows "
                 f"(test slice {t0:.0f}s\u2013{t1:.0f}s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    feat = pd.read_csv("data/can_features.csv")
    train_df, val_df, test_df = time_split(feat)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS].values)
    X_val = scaler.transform(val_df[FEATURE_COLS].values)
    X_test = scaler.transform(test_df[FEATURE_COLS].values)
    y_test = test_df["label"].values

    all_metrics = []
    score_dict = {}

    # ---------------- Isolation Forest ---------------------------------
    print("\nTraining Isolation Forest ...")
    iso = IsolationForest(
        n_estimators=200, contamination="auto", max_samples="auto",
        random_state=0, n_jobs=-1
    )
    iso.fit(X_train)
    # score_samples: higher = more normal. Flip sign so higher = more anomalous.
    val_scores_if = -iso.score_samples(X_val)
    test_scores_if = -iso.score_samples(X_test)
    threshold_if = np.percentile(val_scores_if, 99)  # calibrated on normal-only val data
    pred_if = (test_scores_if > threshold_if).astype(int)
    metrics_if, cm_if = evaluate(y_test, pred_if, test_scores_if, "Isolation Forest")
    metrics_if["threshold"] = float(threshold_if)
    all_metrics.append(metrics_if)
    score_dict["Isolation Forest"] = test_scores_if
    plot_confusion(cm_if, "Isolation Forest", f"{RESULTS_DIR}/confusion_isolation_forest.png")

    # ---------------- Autoencoder (MLP reconstruction) ------------------
    print("\nTraining Autoencoder (MLP-based reconstruction network) ...")
    n_features = X_train.shape[1]
    ae = MLPRegressor(
        hidden_layer_sizes=(16, 6, 16),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=0,
    )
    ae.fit(X_train, X_train)  # reconstruct input -> autoencoder objective

    def recon_error(X):
        X_hat = ae.predict(X)
        return np.mean((X - X_hat) ** 2, axis=1)

    val_scores_ae = recon_error(X_val)
    test_scores_ae = recon_error(X_test)
    threshold_ae = np.percentile(val_scores_ae, 99)
    pred_ae = (test_scores_ae > threshold_ae).astype(int)
    metrics_ae, cm_ae = evaluate(y_test, pred_ae, test_scores_ae, "Autoencoder")
    metrics_ae["threshold"] = float(threshold_ae)
    all_metrics.append(metrics_ae)
    score_dict["Autoencoder"] = test_scores_ae
    plot_confusion(cm_ae, "Autoencoder", f"{RESULTS_DIR}/confusion_autoencoder.png")

    # ---------------- comparison plots ----------------------------------
    plot_roc(y_test, score_dict, f"{RESULTS_DIR}/roc_curve.png")
    plot_score_timeline(test_df, score_dict, f"{RESULTS_DIR}/score_timeline.png")

    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # bar chart comparing precision/recall/F1/FPR
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["precision", "recall", "f1", "false_positive_rate"]
    x = np.arange(len(labels))
    width = 0.35
    for i, m in enumerate(all_metrics):
        vals = [m[k] for k in labels]
        ax.bar(x + (i - 0.5) * width, vals, width, label=m["model"])
    ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1", "FPR"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Isolation Forest vs Autoencoder")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/metric_comparison.png", dpi=150)
    plt.close(fig)

    # persist models + scaler + threshold for the live demo
    import joblib
    joblib.dump({
        "scaler": scaler,
        "iso_forest": iso,
        "iso_threshold": threshold_if,
        "autoencoder": ae,
        "ae_threshold": threshold_ae,
        "feature_cols": FEATURE_COLS,
    }, f"{RESULTS_DIR}/models.joblib")

    test_df.to_csv(f"{RESULTS_DIR}/test_split.csv", index=False)

    print("\nSaved plots + metrics.json + models.joblib to results/")
    return all_metrics


if __name__ == "__main__":
    main()
