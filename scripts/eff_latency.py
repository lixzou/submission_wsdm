#!/usr/bin/env python3
"""Speed/latency experiment for section 4.6 (Efficiency).

Measures per-query latency (ms, single CPU core, numpy prototype) for the
RECO query path on the three BEIR corpora, following the KDD-era tab:eff
format:
  - full float32 scan (brute-force cosine over the full d-dim index)
  - query projection (m x d, the relevance-calibrated rotation of RECO,
    constant in corpus size)
  - truncated scan at 12.5% retention (m = d/8 kept coordinates)
  - RECO total = projection + truncated scan (no per-query decoding)

The projection of RECO keeps m of d coordinates, so the query-side matrix
is (m, d), not (d, d). Both scans are measured as numpy GEMV per query;
the projection is measured per query (constant in corpus size).
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
RETENTION = 0.125           # 12.5% -> m = d/8
MAX_TIMING_QUERIES = 500    # cap the timing loop (fiqa has 6.6k test queries)

def main():
    rng = np.random.default_rng(0)
    res = {}
    for ds in DATASETS:
        emb = np.load(f"{BASE}/{ds}/emb_{MODEL}.npy")        # (N, d)
        qemb = np.load(f"{BASE}/{ds}/qemb_{MODEL}.npy")      # (Q, d)
        N, d = emb.shape
        m = max(1, int(round(RETENTION * d)))
        # relevance-calibrated projection keeps m of d coordinates: V is (m, d)
        V = rng.standard_normal((m, d))
        V /= np.linalg.norm(V, axis=1, keepdims=True)

        # warmup
        _ = emb @ qemb[0]
        _ = V @ qemb[0]
        _ = emb[:, :m] @ qemb[0][:m]

        Q = min(MAX_TIMING_QUERIES, qemb.shape[0])
        qs = qemb[:Q]

        # full float32 scan: per-query brute-force over the full d-dim index
        t0 = time.perf_counter_ns()
        for q in qs:
            scores = emb @ q
        full_ms = (time.perf_counter_ns() - t0) / 1e6 / Q

        # query projection: (m, d) @ (d,), constant in corpus size
        reps = 20000
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            q_rot = V @ qs[0]
        proj_ms = (time.perf_counter_ns() - t0) / 1e6 / reps

        # truncated scan at 12.5% retention: (N, m) @ (m,)
        t0 = time.perf_counter_ns()
        for q in qs:
            scores = emb[:, :m] @ q[:m]
        trunc_ms = (time.perf_counter_ns() - t0) / 1e6 / Q

        reco_ms = proj_ms + trunc_ms
        red_full = (full_ms - trunc_ms) / full_ms      # relative reduction of scan
        res[ds] = dict(N=N, full=full_ms, proj=proj_ms, trunc=trunc_ms,
                       reco=reco_ms, red=red_full)
        print(f"{ds:9s} N={N:6d} m={m:3d}  full={full_ms:.4f}  proj={proj_ms:.5f}  "
              f"trunc={trunc_ms:.4f}  RECO={reco_ms:.4f} ms/query  "
              f"scan reduction={red_full*100:.1f}%")

    print()
    print("% LaTeX table (ms per query, single CPU core, numpy prototype)")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Operation & scifact (5{,}183) & fiqa (57{,}638) & nfcorpus (3{,}633) \\\\")
    print("\\midrule")
    print(f"float32 full scan & {res['scifact']['full']:.3f} & {res['fiqa']['full']:.3f} & {res['nfcorpus']['full']:.3f} \\\\")
    print(f"query projection (${m}\\times{d}$) & {res['scifact']['proj']:.4f} & {res['fiqa']['proj']:.4f} & {res['nfcorpus']['proj']:.4f} \\\\")
    print(f"truncated scan (12.5\\%) & {res['scifact']['trunc']:.3f} & {res['fiqa']['trunc']:.3f} & {res['nfcorpus']['trunc']:.3f} \\\\")
    print(f"RECO total (proj + scan) & {res['scifact']['reco']:.3f} & {res['fiqa']['reco']:.3f} & {res['nfcorpus']['reco']:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

if __name__ == "__main__":
    main()
