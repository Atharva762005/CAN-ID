"""
features.py
------------
Turns raw CAN frames into a per-message feature vector suitable for
anomaly detection. Features are computed causally (only using past
messages) so they would be valid in a real streaming/online IDS.

Feature groups
  1. Timing        : inter-arrival time (global and per-ID), short-term
                      arrival-rate burstiness.
  2. ID statistics  : rolling-window frequency of this ID, rolling
                      Shannon entropy of the ID stream (low entropy =>
                      flooding, high/unusual entropy => fuzzing).
  3. Payload        : byte-level stats of the payload, Hamming distance
                      from the previous message with the same ID (large
                      unexpected jumps flag spoofing/fuzzing), DLC.
"""
import numpy as np
import pandas as pd

ID_FREQ_WINDOW = 50       # messages, for rolling ID frequency / entropy
GLOBAL_RATE_WINDOW = 20   # messages, for local burstiness


def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    ts = df["timestamp"].values
    ids = df["arbitration_id"].values
    payload_cols = [f"d{i}" for i in range(8)]
    payload = df[payload_cols].values.astype(np.int64)
    dlc = df["dlc"].values

    # ---- global inter-arrival time -----------------------------------
    iat_global = np.empty(n)
    iat_global[0] = 0.0
    iat_global[1:] = np.diff(ts)

    # ---- per-ID inter-arrival time & hamming distance from last same-ID msg
    last_seen_time = {}
    last_seen_payload = {}
    iat_same_id = np.empty(n)
    hamming_same_id = np.empty(n)
    id_seen_count = {}
    running_id_count = np.empty(n)  # how many times we've seen this id so far

    # ---- rolling window buffers for entropy / local frequency --------
    from collections import deque, Counter
    window_ids = deque()
    window_counter = Counter()

    id_freq_in_window = np.empty(n)
    id_entropy_in_window = np.empty(n)

    # ---- local burstiness: mean IAT over last GLOBAL_RATE_WINDOW msgs
    iat_buffer = deque(maxlen=GLOBAL_RATE_WINDOW)
    local_mean_iat = np.empty(n)
    local_std_iat = np.empty(n)

    for i in range(n):
        mid = ids[i]
        t = ts[i]

        # per-id timing / payload delta
        if mid in last_seen_time:
            iat_same_id[i] = t - last_seen_time[mid]
            hamming_same_id[i] = np.count_nonzero(payload[i] != last_seen_payload[mid])
        else:
            iat_same_id[i] = -1.0        # sentinel: first time seen
            hamming_same_id[i] = 0
        last_seen_time[mid] = t
        last_seen_payload[mid] = payload[i]

        running_id_count[i] = id_seen_count.get(mid, 0)
        id_seen_count[mid] = running_id_count[i] + 1

        # rolling window of last ID_FREQ_WINDOW ids -> freq of this id + entropy
        window_ids.append(mid)
        window_counter[mid] += 1
        if len(window_ids) > ID_FREQ_WINDOW:
            old = window_ids.popleft()
            window_counter[old] -= 1
            if window_counter[old] == 0:
                del window_counter[old]
        id_freq_in_window[i] = window_counter[mid] / len(window_ids)
        id_entropy_in_window[i] = shannon_entropy(list(window_counter.values()))

        # local burstiness over global IAT
        if i > 0:
            iat_buffer.append(iat_global[i])
        if len(iat_buffer) > 0:
            arr = np.array(iat_buffer)
            local_mean_iat[i] = arr.mean()
            local_std_iat[i] = arr.std()
        else:
            local_mean_iat[i] = 0.0
            local_std_iat[i] = 0.0

    # first-seen sentinel -> fill with global median iat for that context
    med_same_id = np.median(iat_same_id[iat_same_id >= 0]) if np.any(iat_same_id >= 0) else 0.0
    iat_same_id[iat_same_id < 0] = med_same_id

    # ---- payload byte statistics --------------------------------------
    payload_mean = payload.mean(axis=1)
    payload_std = payload.std(axis=1)
    payload_max = payload.max(axis=1)
    payload_min = payload.min(axis=1)
    payload_unique = np.array([len(np.unique(row)) for row in payload])

    feat = pd.DataFrame({
        "arbitration_id": ids,
        "dlc": dlc,
        "iat_global": iat_global,
        "iat_same_id": iat_same_id,
        "hamming_same_id": hamming_same_id,
        "id_freq_window": id_freq_in_window,
        "id_entropy_window": id_entropy_in_window,
        "local_mean_iat": local_mean_iat,
        "local_std_iat": local_std_iat,
        "payload_mean": payload_mean,
        "payload_std": payload_std,
        "payload_max": payload_max,
        "payload_min": payload_min,
        "payload_unique_bytes": payload_unique,
    })

    # keep labels alongside (not used as model input)
    feat["label"] = df["label"].values
    feat["attack_type"] = df["attack_type"].values
    feat["timestamp"] = df["timestamp"].values
    return feat


if __name__ == "__main__":
    df = pd.read_csv("data/can_dataset.csv")
    feat = engineer_features(df)
    feat.to_csv("data/can_features.csv", index=False)
    print(f"Wrote {len(feat):,} feature rows to data/can_features.csv")
    print(feat.describe().T)
