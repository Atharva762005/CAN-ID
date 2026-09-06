"""
generate_dataset.py
--------------------
Synthetic CAN bus traffic generator built to reproduce the documented
statistical structure of the HCRL "Car Hacking: CAN Intrusion Detection
Dataset" (Song, Woo & Kim, 2020) as closely as possible without direct
network access to the original files (see README for why).

Design choices mirror what is published about the real dataset:
  - Normal traffic: ~25-30 periodic CAN IDs, each broadcast at a roughly
    fixed period (10-100 ms) with small jitter, carrying payloads with a
    mix of rolling counters, slowly-varying "sensor" bytes, and a
    checksum-like last byte.
  - DoS attack:      CAN ID 0x000 injected every ~0.3 ms.
  - Fuzzy attack:    random CAN ID + random 8-byte payload every ~0.5 ms.
  - RPM spoofing:    CAN ID 0x316 injected every ~1 ms with a crafted payload.
  - Gear spoofing:   CAN ID 0x43F injected every ~1 ms with a crafted payload.
Attacks are injected in bursts of 3-5 seconds, interleaved with the
continuing background normal traffic, matching the burst structure
described in the HCRL papers.

Output: data/can_dataset.csv with columns
  timestamp, arbitration_id, dlc, d0..d7, label, attack_type
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

NORMAL_IDS = [
    0x043, 0x0c9, 0x0d1, 0x0f0, 0x110, 0x120, 0x130, 0x140,
    0x153, 0x164, 0x18f, 0x1a0, 0x1b0, 0x1c0, 0x1d0, 0x1f1,
    0x220, 0x260, 0x2b0, 0x316, 0x329, 0x350, 0x372, 0x3a0,
    0x43f, 0x4b0, 0x4f0,
]
# each normal id gets its own nominal period (ms) and jitter fraction
ID_PERIODS_MS = {mid: RNG.choice([10, 15, 20, 25, 50, 100]) for mid in NORMAL_IDS}

DOS_ID = 0x000
RPM_ID = 0x316
GEAR_ID = 0x43F


def make_normal_payload(mid, counter):
    """8-byte payload: 2 counter bytes, 4 slowly varying bytes, 1 spare, 1 checksum."""
    payload = np.zeros(8, dtype=np.uint8)
    payload[0] = counter % 256
    payload[1] = (counter // 256) % 256
    # slowly varying pseudo-sensor bytes, seeded by id so each id has its own walk
    walk_seed = (mid * 2654435761 + counter) & 0xffffffff
    local_rng = np.random.default_rng(walk_seed)
    payload[2:6] = local_rng.integers(0, 256, size=4)
    payload[6] = RNG.integers(0, 4)  # near-constant flag byte
    payload[7] = int(payload[:7].sum()) % 256  # checksum-like last byte
    return payload


def gen_normal_stream(duration_s):
    """Generate background normal traffic for the whole capture duration."""
    rows = []
    counters = {mid: 0 for mid in NORMAL_IDS}
    for mid in NORMAL_IDS:
        period_s = ID_PERIODS_MS[mid] / 1000.0
        t = RNG.uniform(0, period_s)
        while t < duration_s:
            counters[mid] += 1
            payload = make_normal_payload(mid, counters[mid])
            rows.append((t, mid, 8, *payload, 0, "normal"))
            jitter = RNG.normal(0, period_s * 0.05)
            t += max(period_s + jitter, 0.0005)
    return rows


def gen_attack_burst(attack_type, t_start, t_end):
    rows = []
    if attack_type == "dos":
        period_s = 0.0003
        payload = np.zeros(8, dtype=np.uint8)
        t = t_start
        while t < t_end:
            rows.append((t, DOS_ID, 8, *payload, 1, "dos"))
            t += max(period_s + RNG.normal(0, period_s * 0.05), 0.00005)
    elif attack_type == "fuzzy":
        period_s = 0.0005
        t = t_start
        while t < t_end:
            mid = int(RNG.integers(0, 0x7FF))
            payload = RNG.integers(0, 256, size=8, dtype=np.uint8)
            rows.append((t, mid, 8, *payload, 1, "fuzzy"))
            t += max(period_s + RNG.normal(0, period_s * 0.05), 0.00005)
    elif attack_type == "rpm":
        period_s = 0.001
        payload = np.array([0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], dtype=np.uint8)
        t = t_start
        while t < t_end:
            rows.append((t, RPM_ID, 8, *payload, 1, "rpm_spoof"))
            t += max(period_s + RNG.normal(0, period_s * 0.05), 0.0002)
    elif attack_type == "gear":
        period_s = 0.001
        payload = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00], dtype=np.uint8)
        t = t_start
        while t < t_end:
            rows.append((t, GEAR_ID, 8, *payload, 1, "gear_spoof"))
            t += max(period_s + RNG.normal(0, period_s * 0.05), 0.0002)
    return rows


def build_dataset(duration_s=120, n_bursts_per_attack=5, out_path="data/can_dataset.csv"):
    print(f"Generating {duration_s}s of background normal CAN traffic ...")
    normal_rows = gen_normal_stream(duration_s)
    print(f"  -> {len(normal_rows):,} normal messages")

    # schedule non-overlapping attack windows
    attack_types = ["dos", "fuzzy", "rpm", "gear"]
    windows = []
    for atk in attack_types:
        for _ in range(n_bursts_per_attack):
            dur = RNG.uniform(3.0, 5.0)
            start = RNG.uniform(1.0, duration_s - dur - 1.0)
            windows.append((atk, start, start + dur))
    windows.sort(key=lambda w: w[1])
    # nudge overlaps apart a bit (simple greedy fix)
    for i in range(1, len(windows)):
        atk, s, e = windows[i]
        _, _, prev_e = windows[i - 1]
        if s < prev_e + 0.2:
            shift = (prev_e + 0.2) - s
            windows[i] = (atk, s + shift, e + shift)

    attack_rows = []
    for atk, s, e in windows:
        rows = gen_attack_burst(atk, s, e)
        attack_rows.extend(rows)
        print(f"  attack burst: {atk:10s} [{s:6.2f}s - {e:6.2f}s]  ({len(rows):,} msgs)")

    all_rows = normal_rows + attack_rows
    cols = ["timestamp", "arbitration_id", "dlc",
            "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
            "label", "attack_type"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.sort_values("timestamp", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows to {out_path}")
    print(df["attack_type"].value_counts())
    print(f"Overall attack ratio: {df['label'].mean():.3%}")
    return df


if __name__ == "__main__":
    build_dataset()
