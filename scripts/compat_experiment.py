#!/usr/bin/env python3
"""compat_experiment.py — 兼容性实验（用户指令 2026-08-17）:
置换的截断增益是否与下游量化器无关。

设计: 同一总字节预算下,
  (a) Prefix 选前 m 维 vs Perm 选 m 个 ranking-importance 维
  (b) 选中 m 维向量上跑同一量化器（PQ m_sub / LSQ M_sub）
总预算: keep 12.5% (96d) + PQ m12/LSQ M12 = 32x; keep 25% (192d) + m24 = 32x
（96d float 384B → 96B 量化 = 4x × 截断 8x = 32x 总; 192d 同理）
"""
import argparse, json, time
import numpy as np
import faiss
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance


def recall_of(o, qsel, qids, qrels, ids):
    o = np.ascontiguousarray(o, dtype=np.float32)
    norms = np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    o /= norms
    qsel = np.ascontiguousarray(qsel, dtype=np.float32)
    return recall_at_k(brute_rank(qsel, o), qids, qrels, ids)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keeps", nargs="+", type=float, default=[0.25, 0.125])
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d, "reference": round(ref, 4),
           "protocol": "truncate(prefix|perm) -> same quantizer (PQ/LSQ) at matched total bytes"}
    imp = margin_importance(emb, qemb, qids, qrels, ids)
    perm_order = np.argsort(-imp)
    for keep in args.keeps:
        m = max(1, int(d * keep))
        M = m // 8  # m 维向量, 每维 4B -> m 字节预算 = 8 子量化器 × 8bit（总字节对齐 32x）
        for order_name, order in [("prefix", np.arange(d)), ("perm", perm_order)]:
            sel = order[:m]
            x = np.ascontiguousarray(emb[:, sel], dtype=np.float32)
            q = np.ascontiguousarray(qemb[:, sel], dtype=np.float32)
            for kind in ["pq", "lsq"]:
                t0 = time.time()
                if kind == "pq":
                    quant = faiss.IndexPQ(m, M, 8)
                else:
                    quant = faiss.LocalSearchQuantizer(m, M, 8)
                    quant.icm_iters = 5
                    quant.train_iters = 10
                quant.train(x)
                if kind == "pq":
                    codes = quant.sa_encode(x)
                    o = quant.sa_decode(codes)
                else:
                    o = quant.decode(quant.compute_codes(x))
                r = recall_of(o, q, qids, qrels, ids)
                res[f"keep={keep}_{order_name}_{kind}"] = round(r, 4)
                print(f"keep={keep} {order_name} {kind}: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
