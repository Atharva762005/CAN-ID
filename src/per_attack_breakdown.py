"""Per-attack-type detection breakdown: does the IDS catch DoS, fuzzy,
RPM-spoof and gear-spoof equally well, or does it miss subtler ones
(spoofing attacks reuse legitimate IDs/payload shapes, so they are the
hardest to catch -- worth surfacing explicitly rather than hiding behind
one aggregate recall number)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

def recon_error_mlp(ae, X):
    X_hat = ae.predict(X)
    return np.mean((X - X_hat) ** 2, axis=1)

def main():
    bundle = joblib.load("results/models.joblib")
    scaler = bundle["scaler"]
    iso = bundle["iso_forest"]
    iso_thr = bundle["iso_threshold"]
    ae = bundle["autoencoder"]
    ae_thr = bundle["ae_threshold"]
    feature_cols = bundle["feature_cols"]

    test_df = pd.read_csv("results/test_split.csv")
    X_test = scaler.transform(test_df[feature_cols].values)

    scores_if = -iso.score_samples(X_test)
    scores_ae = recon_error_mlp(ae, X_test)
    pred_if = (scores_if > iso_thr).astype(int)
    pred_ae = (scores_ae > ae_thr).astype(int)

    test_df = test_df.copy()
    test_df["pred_if"] = pred_if
    test_df["pred_ae"] = pred_ae

    rows = []
    for atk in ["dos", "fuzzy", "rpm_spoof", "gear_spoof"]:
        sub = test_df[test_df["attack_type"] == atk]
        if len(sub) == 0:
            continue
        rec_if = sub["pred_if"].mean()
        rec_ae = sub["pred_ae"].mean()
        rows.append({"attack_type": atk, "n": len(sub),
                      "recall_isolation_forest": rec_if, "recall_autoencoder": rec_ae})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv("results/per_attack_recall.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width/2, summary["recall_isolation_forest"], width, label="Isolation Forest")
    ax.bar(x + width/2, summary["recall_autoencoder"], width, label="Autoencoder")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["attack_type"])
    ax.set_ylabel("Recall (detection rate)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection rate by attack type")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/per_attack_recall.png", dpi=150)
    print("\nSaved results/per_attack_recall.csv and .png")

if __name__ == "__main__":
    main()
