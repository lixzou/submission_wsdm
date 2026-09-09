#!/usr/bin/env python3
"""reco_pq_check.py — RECO 置换 + PQ 联合管线 vs 纯 PQ @32x（scifact + NQ 50k）。

预算对齐: 全维 PQ M=m (m = d*2/8) ↔ 50% 截断 + PQ M=m/2。
如果置换增益叠加 PQ 码本质量, 联合管线应赢纯 PQ。
"""
import sys, json
import numpy as np
import faiss
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments')
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance


def main():
    ds = sys.argv[1]; model = sys.argv[2]; out = sys.argv[3]
    calib_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    emb, qemb, qids, qrels, ids = load(ds, model)
    d = emb.shape[1]
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=calib_frac)
    perm_order = np.argsort(-imp)
    res = {"ds_dir": ds, "model": model, "d": d,
           "reference": round(recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0], 4)}
    for keep in [0.5, 0.25, 0.125]:
        m = max(1, int(d * keep))
        M = max(1, int(d * keep * 2 / 8))  # PQ 子量化器
        # perm 截断 + PQ on 选中维度
        sel = perm_order[:m]
        x = np.ascontiguousarray(emb[:, sel], dtype=np.float32)
        qx = np.ascontiguousarray(qemb[:, sel], dtype=np.float32)
        q = faiss.IndexPQ(m, M, 8)
        q.train(x)
        o = q.sa_decode(q.sa_encode(x))
        o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
        r = recall_at_k(brute_rank(qx, o), qids, qrels, ids)[0]
        res[f"perm_{keep}_pqM{M}"] = round(r, 4)
        # prefix 截断 + PQ
        x2 = np.ascontiguousarray(emb[:, :m], dtype=np.float32)
        qx2 = np.ascontiguousarray(qemb[:, :m], dtype=np.float32)
        q2 = faiss.IndexPQ(m, M, 8)
        q2.train(x2)
        o2 = q2.sa_decode(q2.sa_encode(x2))
        o2 = o2 / (np.linalg.norm(o2, axis=1, keepdims=True) + 1e-12)
        r2 = recall_at_k(brute_rank(qx2, o2), qids, qrels, ids)[0]
        res[f"pfx_{keep}_pqM{M}"] = round(r2, 4)
        # 全维 PQ M_full（同总字节）
        M_full = max(1, int(d * 2 / 8))
        if keep == 0.5:  # 只跑一次全维
            qf = faiss.IndexPQ(d, M_full, 8)
            qf.train(emb)
            of = qf.sa_decode(qf.sa_encode(emb))
            of = of / (np.linalg.norm(of, axis=1, keepdims=True) + 1e-12)
            rf = recall_at_k(brute_rank(qemb, of), qids, qrels, ids)[0]
            res[f"full_pqM{M_full}"] = round(rf, 4)
        print(f"keep={keep} perm+pq {r:.4f} pfx+pq {r2:.4f}", flush=True)
    print(f"full pq {res.get('full_pqM96', '?')}", flush=True)
    json.dump(res, open(out, 'w'), indent=1)
    print("SAVED", out)


if __name__ == "__main__":
    main()
