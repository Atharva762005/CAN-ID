"""Exports a message-by-message replay slice (with streaming-computed
features + both models' scores) as JSON, for the browser-based live
dashboard demo (demo/dashboard.html), since that artifact can't run
python/sklearn itself.
"""
import json
import numpy as np
import pandas as pd
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from live_demo import StreamingFeaturizer, recon_error_mlp


def main():
    bundle = joblib.load("results/models.joblib")
    scaler = bundle["scaler"]
    iso = bundle["iso_forest"]
    iso_thr = bundle["iso_threshold"]
    ae = bundle["autoencoder"]
    ae_thr = bundle["ae_threshold"]

    raw = pd.read_csv("data/can_dataset.csv").sort_values("timestamp").reset_index(drop=True)

    # narrative slice: covers a gear-spoof burst -> quiet period -> DoS flood
    sub = raw[(raw["timestamp"] >= 93.0) & (raw["timestamp"] <= 102.5)].reset_index(drop=True)
    # thin dense attack sections a bit so the browser doesn't have to animate
    # tens of thousands of rows, while keeping every normal message
    keep_mask = np.ones(len(sub), dtype=bool)
    attack_idx = np.where(sub["label"].values == 1)[0]
    normal_idx = np.where(sub["label"].values == 0)[0]
    # keep 1-in-8 attack-frames (dense floods) and 1-in-2 normal frames,
    # to shrink the browser payload while preserving the narrative arc
    thin_attack = attack_idx[np.arange(len(attack_idx)) % 8 != 0]
    thin_normal = normal_idx[np.arange(len(normal_idx)) % 2 != 0]
    keep_mask[thin_attack] = False
    keep_mask[thin_normal] = False
    sub = sub[keep_mask].reset_index(drop=True)

    feat_engine = StreamingFeaturizer()
    payload_cols = [f"d{i}" for i in range(8)]

    records = []
    scores_if_all, scores_ae_all = [], []
    for _, row in sub.iterrows():
        t = row["timestamp"]
        mid = int(row["arbitration_id"])
        payload = row[payload_cols].values.astype(np.int64)
        truth = int(row["label"])
        attack_type = row["attack_type"]

        feat_vec = feat_engine.update(t, mid, payload)
        X = scaler.transform([feat_vec])
        score_if = float(-iso.score_samples(X)[0])
        score_ae = float(recon_error_mlp(ae, X)[0])
        scores_if_all.append(score_if)
        scores_ae_all.append(score_ae)

        records.append({
            "t": round(float(t), 5),
            "id": f"0x{mid:03X}",
            "payload": " ".join(f"{b:02X}" for b in payload),
            "truth": truth,
            "attack_type": attack_type if truth else "normal",
            "score_if": score_if,
            "score_ae": score_ae,
            "alert_if": int(score_if > iso_thr),
            "alert_ae": int(score_ae > ae_thr),
        })

    out = {
        "iso_threshold": float(iso_thr),
        "ae_threshold": float(ae_thr),
        "ae_score_p99": float(np.percentile(scores_ae_all, 99)),
        "if_score_range": [float(np.min(scores_if_all)), float(np.max(scores_if_all))],
        "ae_score_range": [float(np.min(scores_ae_all)), float(np.percentile(scores_ae_all, 99.5))],
        "records": records,
    }
    with open("demo/replay_data.json", "w") as f:
        json.dump(out, f)
    print(f"Exported {len(records):,} messages to demo/replay_data.json")
    print(f"Attack fraction in slice: {sub['label'].mean():.1%}")


if __name__ == "__main__":
    main()
