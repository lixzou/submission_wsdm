#!/usr/bin/env python3
"""reco_pq_nq2m.py — RECO perm + PQ 联合 vs 纯 PQ @ NQ 2M（50k 子集）"""
import sys, json
import numpy as np
import faiss
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments')
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance

ds = sys.argv[1]; model = sys.argv[2]; out = sys.argv[3]
emb, qemb, qids, qrels, ids = load(ds, model)
d = emb.shape[1]
n_docs = len(emb); n_q = len(qemb)
imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=0.7)
perm_order = np.argsort(-imp)
res = {"ds_dir": ds, "model": model, "d": d, "n_docs": n_docs,
       "reference": round(recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0], 4)}
M_full = max(1, int(d * 2 / 8))
qf = faiss.IndexPQ(d, M_full, 8); qf.train(emb[:50000])
of = qf.sa_decode(qf.sa_encode(emb))
of = of / (np.linalg.norm(of, axis=1, keepdims=True) + 1e-12)
rf = recall_at_k(brute_rank(qemb, of), qids, qrels, ids)[0]
res[f"full_pqM{M_full}"] = round(rf, 4)
for keep in [0.5, 0.25, 0.125]:
    m = max(1, int(d * keep)); M = max(1, int(d * keep * 2 / 8))
    sel = perm_order[:m]
    x = np.ascontiguousarray(emb[:, sel], dtype=np.float32)
    qx = np.ascontiguousarray(qemb[:, sel], dtype=np.float32)
    q = faiss.IndexPQ(m, M, 8); q.train(x[:50000])
    o = q.sa_decode(q.sa_encode(x))
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    r = recall_at_k(brute_rank(qx, o), qids, qrels, ids)[0]
    res[f"perm_{keep}_pqM{M}"] = round(r, 4)
    print(f"keep={keep} perm+pq {r:.4f}", flush=True)
print(f"full pq {rf:.4f}", flush=True)
json.dump(res, open(out, 'w'), indent=1)
print("SAVED", out)
