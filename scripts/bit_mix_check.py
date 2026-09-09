#!/usr/bin/env python3
"""bit_mix_check.py — 位宽微调（3/1 混合 vs 均匀 2, 总字节不变）: 校准重要性高维 3 位、低维 1 位。
试探 Thm 2 边界: 校准网格的逐维误差函数不同 → 位宽可微调?"""
import json, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from rabitq_py import random_orthogonal


def lloyd_max_from_samples(vals, K, tol=1e-6, max_iter=200):
    vals = np.asarray(vals)
    levels = np.quantile(vals, np.linspace(0, 1, K))
    for it in range(max_iter):
        c = np.argmin(np.abs(vals[:, None] - levels[None, :]), axis=1)
        new = np.array([vals[c == k].mean() if np.any(c == k) else levels[k] for k in range(K)])
        if np.max(np.abs(new - levels)) < tol: break
        levels = new
    return np.sort(levels)


def main():
    ds = sys.argv[1]; model = sys.argv[2]; out = sys.argv[3]
    emb, qemb, qids, qrels, ids = load(ds, model)
    d = emb.shape[1]; m = d // 2
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=1.0)
    perm_order = np.argsort(-imp)
    sel = perm_order[:m]
    P = random_orthogonal(m, seed=0)
    rot = (emb[:, sel] @ P).astype(np.float64)
    qidx = {q: i for i, q in enumerate(qids)}
    idset = {s: i for i, s in enumerate(ids)}
    rel_vals = []
    for qid in qids:
        if qid not in qrels: continue
        for did in qrels[qid]:
            if did in idset:
                rel_vals.append(rot[idset[did]])
    rel_vals = np.array(rel_vals)
    # 均匀 2 位 vs 3/1 混合（高 w_j 维 3 位, 低 w_j 维 1 位, 总位不变: m/2*3 + m/2*1 = 2m）
    res = {"model": model}
    for name, B_assign in [("uniform2", np.full(m, 2)), ("mix31", None)]:
        if name == "mix31":
            half = m // 2
            B_assign = np.concatenate([np.full(half, 3), np.full(m - half, 1)])
        codes = np.zeros((len(rot), m), np.int32)
        for j in range(m):
            K = 2 ** B_assign[j]
            lv = lloyd_max_from_samples(rel_vals[:, j], K)
            codes[:, j] = np.argmin(np.abs(rot[:, j, None] - lv[None, :]), axis=1)
            if j == 0:
                globals()['lv0'] = lv
        # 重建（逐维反查级——简化: 存级表）
        levels_map = {}
        for j in range(m):
            K = 2 ** B_assign[j]
            levels_map[j] = lloyd_max_from_samples(rel_vals[:, j], K)
        recon = np.zeros_like(rot)
        for j in range(m):
            recon[:, j] = levels_map[j][codes[:, j]]
        o = (P @ recon.T).T
        o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
        r = recall_at_k(brute_rank(qemb[:, sel].astype(np.float32), o.astype(np.float32)), qids, qrels, ids)[0]
        res[name] = round(r, 4)
        print(f"{model} {name}: {r:.4f}", flush=True)
    json.dump(res, open(out, 'w'), indent=1)
    print("SAVED", out)


if __name__ == "__main__":
    main()
