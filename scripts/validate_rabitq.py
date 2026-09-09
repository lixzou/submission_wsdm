#!/usr/bin/env python3
"""RaBitQ Python 实现校验: ①无偏性（回归斜率≈1）②单调性（B↑ recall 单调不减）③论文锚点（B=5 应 >90% recall）。
"""
import sys, json, argparse, time
sys.path.insert(0, "/ssd1/zoulixin/tencent_previous_compression/experiments")
import numpy as np
from rabitq_py import random_orthogonal, quantize_Bbit, estimate_scores
from pilot_truncation import load, recall_at_k, brute_rank

ap = argparse.ArgumentParser()
ap.add_argument("--ds_dir", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--bits", nargs="+", type=int, default=[1, 2, 3, 4, 5, 8])
ap.add_argument("--out", default="/tmp/validate_rabitq.json")
args = ap.parse_args()

emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
N, d = emb.shape
P = random_orthogonal(d, seed=7)
rot = (P.T @ emb.T).T.astype(np.float64)   # P⁻¹o = Pᵀo
qrot = (P.T @ qemb.T).T.astype(np.float32)

ref_rank = brute_rank(qemb, emb)
ref = recall_at_k(ref_rank, qids, qrels, ids)[0]
print("reference recall@10:", round(ref, 4), flush=True)

res = {"reference": ref}
# 无偏性检查（B=1, 抽样 2000 对）
codes1, fac1 = quantize_Bbit(rot[:3000], 1)
o_rec = (P @ (codes1.astype(np.float32) - 0.5).T).T
o_rec /= np.linalg.norm(o_rec, axis=1, keepdims=True) + 1e-12
true = (emb[:3000] @ qemb[0])
est = (o_rec @ qemb[0]) / fac1[:3000]
r2 = np.corrcoef(true, est)[0, 1]
print("unbiasedness (B=1, query 0, 3000 docs): corr=%.4f  est/true slope=%.3f" % (
    r2, float(np.polyfit(true, est, 1)[0])), flush=True)

for B in args.bits:
    t0 = time.time()
    codes, fac = quantize_Bbit(rot, B)
    o = (P @ (codes.astype(np.float32) - ((2**B)-1)/2.0).T).T
    o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    rec = recall_at_k(brute_rank(qemb, o.astype(np.float32)), qids, qrels, ids)[0]
    est_rec = recall_at_k(np.argpartition(-estimate_scores(qrot, P, codes, B, fac), 10, axis=1)[:, :10],
                          qids, qrels, ids)[0]
    res["B=%d" % B] = {"recall": round(rec, 4), "recall_estimator": round(est_rec, 4),
                       "sec": round(time.time() - t0, 1)}
    print("B=%d: recall=%.4f (estimator %.4f) [%.1fs]" % (B, rec, est_rec, time.time() - t0), flush=True)

json.dump(res, open(args.out, "w"), indent=1)
print("VALIDATE DONE ->", args.out)
