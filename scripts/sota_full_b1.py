#!/usr/bin/env python3
"""sota_full_b1.py — NQ 2M 全维 B1（32x）: 不中心化 vs 中心化 vs 校准符号级。"""
import json, sys, time
import numpy as np
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments')
from pilot_truncation import load, recall_at_k, brute_rank
from rabitq_py import random_orthogonal, quantize_Bbit

ds = '/ssd1/zoulixin/tencent_previous_compression/experiments/data/beir_big/nq'
model = sys.argv[1]; out = sys.argv[2]
emb = np.load(f'{ds}/emb_{model}.npy')
qemb = np.load(f'{ds}/qemb_{model}.npy')
meta = json.load(open(f'{ds}/meta.json'))
qids = [str(q) for q in np.load(f'{ds}/query_ids.npy')]
qrels = {str(q): {str(d) for d in docs} for q, docs in meta['qrels'].items()}
ids = meta['ids']
d = emb.shape[1]
ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
P = random_orthogonal(d, seed=0)
mu = emb.mean(0)
res = {"model": model, "reference": round(ref, 4)}
t0 = time.time()
for name, X in [("uncentered", emb), ("centered", emb - mu)]:
    rot = (X @ P).astype(np.float32)
    codes, _ = quantize_Bbit(rot, 1)
    y = codes.astype(np.float64) - 0.5
    o = (P @ y.T).T
    if name == "centered":
        o = o + mu
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    r = recall_at_k(brute_rank(qemb.astype(np.float32), o.astype(np.float32)), qids, qrels, ids)[0]
    res[name] = round(r, 4)
    print(f"{model} {name} B1: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open(out, 'w'), indent=1)
print("SAVED", out)
