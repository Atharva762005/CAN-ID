"""
live_demo.py
------------
Replays the held-out test split message-by-message, computing features
causally (same code path as features.py, just streaming) and scoring
each message with the trained models in real time. Prints a live-style
console feed and flags anomalies as they're detected.

Usage:
    python3 src/live_demo.py                # replay at ~2000 msg/sec, capped
    python3 src/live_demo.py --speed 0.0005  # seconds of real sleep per message
    python3 src/live_demo.py --limit 3000    # only replay first N test messages
"""
import argparse
import time
import numpy as np
import pandas as pd
import joblib
from collections import deque, Counter


def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


class StreamingFeaturizer:
    """Recomputes the same features as features.py, but incrementally,
    the way a real in-vehicle IDS would have to (no look-ahead)."""

    def __init__(self, id_freq_window=50, rate_window=20):
        self.id_freq_window = id_freq_window
        self.rate_window = rate_window
        self.last_seen_time = {}
        self.last_seen_payload = {}
        self.window_ids = deque()
        self.window_counter = Counter()
        self.iat_buffer = deque(maxlen=rate_window)
        self.last_t = None
        self.same_id_iats = deque(maxlen=500)

    def update(self, t, mid, payload):
        iat_global = 0.0 if self.last_t is None else (t - self.last_t)
        self.last_t = t

        if mid in self.last_seen_time:
            iat_same_id = t - self.last_seen_time[mid]
            hamming = int(np.count_nonzero(payload != self.last_seen_payload[mid]))
        else:
            iat_same_id = float(np.median(self.same_id_iats)) if self.same_id_iats else 0.0
            hamming = 0
        self.last_seen_time[mid] = t
        self.last_seen_payload[mid] = payload
        if iat_same_id >= 0:
            self.same_id_iats.append(iat_same_id)

        self.window_ids.append(mid)
        self.window_counter[mid] += 1
        if len(self.window_ids) > self.id_freq_window:
            old = self.window_ids.popleft()
            self.window_counter[old] -= 1
            if self.window_counter[old] == 0:
                del self.window_counter[old]
        id_freq = self.window_counter[mid] / len(self.window_ids)
        id_entropy = shannon_entropy(list(self.window_counter.values()))

        if iat_global > 0:
            self.iat_buffer.append(iat_global)
        if self.iat_buffer:
            arr = np.array(self.iat_buffer)
            local_mean = arr.mean()
            local_std = arr.std()
        else:
            local_mean = local_std = 0.0

        payload_mean = payload.mean()
        payload_std = payload.std()
        payload_max = payload.max()
        payload_min = payload.min()
        payload_unique = len(np.unique(payload))

        return [mid, 8, iat_global, iat_same_id, hamming, id_freq, id_entropy,
                local_mean, local_std, payload_mean, payload_std, payload_max,
                payload_min, payload_unique]


def recon_error_mlp(ae, X):
    X_hat = ae.predict(X)
    return np.mean((X - X_hat) ** 2, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.0,
                         help="seconds of wall-clock sleep per message (0 = as fast as possible)")
    parser.add_argument("--limit", type=int, default=4000,
                         help="number of raw CAN messages to replay from the test set")
    parser.add_argument("--model", choices=["if", "ae", "both"], default="both")
    args = parser.parse_args()

    bundle = joblib.load("results/models.joblib")
    scaler = bundle["scaler"]
    iso = bundle["iso_forest"]
    iso_thr = bundle["iso_threshold"]
    ae = bundle["autoencoder"]
    ae_thr = bundle["ae_threshold"]

    raw = pd.read_csv("data/can_dataset.csv").sort_values("timestamp").reset_index(drop=True)
    # align to the same held-out test window used in train_eval.py (last ~40%)
    n = len(raw)
    val_end = int(n * 0.6)
    raw_test = raw.iloc[val_end:val_end + args.limit].reset_index(drop=True)

    feat_engine = StreamingFeaturizer()
    payload_cols = [f"d{i}" for i in range(8)]

    print("=" * 92)
    print("  CAN BUS IDS -- LIVE REPLAY DEMO  (streaming feature extraction + trained models)")
    print("=" * 92)
    print(f"{'t(s)':>9} | {'ID':>6} | {'payload':<24} | {'truth':<7} | {'IF':<10} | {'AE':<10}")
    print("-" * 92)

    n_alerts_if = n_alerts_ae = 0
    n_true_attacks = 0
    tp_if = tp_ae = 0

    for i, row in raw_test.iterrows():
        t = row["timestamp"]
        mid = int(row["arbitration_id"])
        payload = row[payload_cols].values.astype(np.int64)
        truth = int(row["label"])
        n_true_attacks += truth

        feat_vec = feat_engine.update(t, mid, payload)
        X = scaler.transform([feat_vec])

        score_if = -iso.score_samples(X)[0]
        pred_if = int(score_if > iso_thr)
        score_ae = recon_error_mlp(ae, X)[0]
        pred_ae = int(score_ae > ae_thr)

        n_alerts_if += pred_if
        n_alerts_ae += pred_ae
        if pred_if and truth:
            tp_if += 1
        if pred_ae and truth:
            tp_ae += 1

        if pred_if or pred_ae or i % 500 == 0:
            payload_str = " ".join(f"{b:02X}" for b in payload)
            truth_str = "ATTACK" if truth else "normal"
            if_str = f"ALERT({score_if:.2f})" if pred_if else "ok"
            ae_str = f"ALERT({score_ae:.2f})" if pred_ae else "ok"
            flag = " <-- MISS" if (truth and not (pred_if or pred_ae)) else ""
            print(f"{t:9.4f} | 0x{mid:03X} | {payload_str:<24} | {truth_str:<7} | "
                  f"{if_str:<10} | {ae_str:<10}{flag}")

        if args.speed > 0:
            time.sleep(args.speed)

    print("-" * 92)
    print(f"Replayed {len(raw_test):,} messages "
          f"({n_true_attacks:,} true attack frames, {n_true_attacks/len(raw_test):.1%})")
    print(f"Isolation Forest : {n_alerts_if:,} alerts raised, "
          f"{tp_if:,} were true positives "
          f"(precision {tp_if/max(n_alerts_if,1):.1%}, recall {tp_if/max(n_true_attacks,1):.1%})")
    print(f"Autoencoder      : {n_alerts_ae:,} alerts raised, "
          f"{tp_ae:,} were true positives "
          f"(precision {tp_ae/max(n_alerts_ae,1):.1%}, recall {tp_ae/max(n_true_attacks,1):.1%})")


if __name__ == "__main__":
    main()
