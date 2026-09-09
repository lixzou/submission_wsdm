#!/usr/bin/env python3
"""Scan latency vs compression ratio, for section 4.6 (Efficiency).

Measures per-query scan latency (ms, single CPU core, numpy prototype) for the
full-width float32 index and for the truncated (compressed) index at the three
retention levels of the main budget sweep (87.5% / 75% / 50% -> 8x / 16x / 32x),
on the three BEIR corpora.

The scan is a numpy GEMV (N, m) @ (m,) per query; its cost scales with the
retained dimension m, so higher compression (smaller m) scans faster.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
import time

BASE = "/ssd1/zoulixin/tencent_previous_compression/experiments/data/beir"
DATASETS = ["scifact", "fiqa", "nfcorpus"]
MODEL = "e5"
# (retention, budget label) in the paper's 32x / 16x / 8x order
BUDGETS = [(0.5, "32x"), (0.75, "16x"), (0.875, "8x")]
MAX_TIMING_QUERIES = 500

def main():
    rows = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        emb = np.load(f"{BASE}/{ds}/emb_{MODEL}.npy")
        qemb = np.load(f"{BASE}/{ds}/qemb_{MODEL}.npy")
        N, d = emb.shape
        Q = min(MAX_TIMING_QUERIES, qemb.shape[0])
        qs = qemb[:Q]
        _ = emb @ qs[0]  # warmup

        def time_scan(X, Q_):
            best = float("inf")
            for _ in range(5):  # min over repetitions is robust to noise
                t0 = time.perf_counter_ns()
                for q in Q_:
                    _ = X @ q
                best = min(best, (time.perf_counter_ns() - t0) / 1e6 / Q)
            return best

        rows[ds]["full"] = time_scan(emb, qs)

        for keep, label in BUDGETS:
            m = max(8, (int(round(keep * d)) // 8) * 8)
            emb_m = np.ascontiguousarray(emb[:, :m])  # stored (N, m) index
            qm = qs[:, :m]
            rows[ds][keep] = time_scan(emb_m, qm)
        print(f"{ds:9s} N={N:6d} d={d:4d}  full={rows[ds]['full']:.3f}  " +
              "  ".join(f"{label}={rows[ds][keep]:.3f}"
                        for keep, label in BUDGETS), flush=True)

    print()
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Scan latency (ms) & scifact & fiqa & nfcorpus \\\\")
    print("\\midrule")
    print(f"float32 (full) & {rows['scifact']['full']:.3f} & "
          f"{rows['fiqa']['full']:.3f} & {rows['nfcorpus']['full']:.3f} \\\\")
    for keep, label in BUDGETS:
        print(f"{label} ({keep*100:.1f}\\% kept) & "
              f"{rows['scifact'][keep]:.3f} & {rows['fiqa'][keep]:.3f} & "
              f"{rows['nfcorpus'][keep]:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print()
    for keep, label in BUDGETS:
        sp = [rows[ds]["full"] / rows[ds][keep] for ds in DATASETS]
        print(f"speedup at {label}: " +
              "  ".join(f"{ds}={v:.2f}x" for ds, v in zip(DATASETS, sp)))

if __name__ == "__main__":
    main()
