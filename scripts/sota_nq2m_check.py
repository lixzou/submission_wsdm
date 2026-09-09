#!/usr/bin/env python3
"""sota_nq2m_check.py — 真实规模 SOTA 验证: NQ 2M 全量 50% 截断 + 校准网格 B2 vs PQ。

校准网格用 5k 子集拟合（快）, 全量 2M 编码 + 3452 查询暴力检索。
"""
import json, time
import numpy as np
import faiss
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
    import sys
    sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments')
    ds = '/ssd1/zoulixin/tencent_previous_compression/experiments/data/beir_big/nq'
    model = sys.argv[1]
    out = sys.argv[2]
    emb = np.load(f'{ds}/emb_{model}.npy')
    qemb = np.load(f'{ds}/qemb_{model}.npy')
    meta = json.load(open(f'{ds}/meta.json'))
    qids = [str(q) for q in np.load(f'{ds}/query_ids.npy')]
    qrels = {str(q): {str(d) for d in docs} for q, docs in meta['qrels'].items()}
    ids = meta['ids']
    d = emb.shape[1]
    m = d // 2
    t0 = time.time()
    # 校准（5k 子集）
    sub = np.load(f'{ds}/emb5k_{model}.npy') if model in ('bgem3', 'qwen3') else emb[:5000]
    ids5k = ids[:5000]
    imp = margin_importance(sub, qemb[:500] if model in ('bgem3', 'qwen3') else qemb[:500],
                            qids[:500], qrels, ids5k, calib_frac=1.0)
    perm_order = np.argsort(-imp)[:m]
    sel = perm_order
    P = random_orthogonal(m, seed=0)
    rot_sub = (sub[:, sel] @ P).astype(np.float64)
    qidx = {q: i for i, q in enumerate(qids[:500])}
    idset = {s: i for i, s in enumerate(ids5k)}
    rel_vals = []
    for qid in qids[:500]:
        if qid not in qrels: continue
        for did in qrels[qid]:
            if did in idset:
                rel_vals.append(rot_sub[idset[did]])
    rel_vals = np.array(rel_vals)
    levels = np.array([lloyd_max_from_samples(rel_vals[:, j], 4) for j in range(m)])
    print(f"calib done [{time.time()-t0:.0f}s]", flush=True)
    # 全量编码（分批）
    B = 200000
    recon_all = []
    for i in range(0, len(emb), B):
        x = emb[i:i+B, sel]
        rot = (x @ P).astype(np.float64)
        codes = np.zeros((len(rot), m), np.int32)
        for j in range(m):
            codes[:, j] = np.argmin(np.abs(rot[:, j, None] - levels[j][None, :]), axis=1)
        rec = levels[np.arange(m)[None, :], codes]
        o = (P @ rec.T).T
        o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
        recon_all.append(o.astype(np.float32))
        if (i // B) % 2 == 1:
            print(f"encoded {i+B}/{len(emb)} [{time.time()-t0:.0f}s]", flush=True)
    o = np.vstack(recon_all)
    qsel = qemb[:, sel]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    r = recall_at_k(brute_rank(qsel, o), qids, qrels, ids)[0]
    print(f"{model}: RPC 50%+calibB2 {r:.4f} (ref {ref:.4f})", flush=True)
    # PQ 同预算（M = m*B/8 = 96 on 768d）
    M = max(1, int(d * 0.5 * 2 / 8))
    pq = faiss.IndexPQ(d, M, 8)
    pq.train(emb[:50000])
    oq = pq.sa_decode(pq.sa_encode(emb))
    oq = oq / (np.linalg.norm(oq, axis=1, keepdims=True) + 1e-12)
    rpq = recall_at_k(brute_rank(qemb, oq.astype(np.float32)), qids, qrels, ids)[0]
    print(f"{model}: PQ M={M} {rpq:.4f} | Δ={round((r-rpq)*100,1)}pp", flush=True)
    json.dump({"model": model, "reference": round(ref, 4), "rpc_50_calibB2": round(r, 4),
               "pq_M": round(rpq, 4), "delta_pp": round((r - rpq) * 100, 1)},
              open(out, 'w'), indent=1)
    print("SAVED", out)


if __name__ == "__main__":
    main()
