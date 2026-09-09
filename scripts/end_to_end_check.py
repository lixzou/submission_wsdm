#!/usr/bin/env python3
"""end_to_end_check.py — 审查点 3: 端到端管线（Perm+B=1+熵）vs 同预算 PQ/LSQ。

预算对齐: 全管线总字节 = 保留维度码字实测熵/8; PQ/LSQ 用相同总字节的 m 子量化器。
（768d float32 = 3072B; 12.5% 截断 + B=1 码 = 96 bit ≈ 12B 码, 熵≈50% → ~6B 总）
"""
import argparse, json
import numpy as np
import faiss
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from rabitq_py import random_orthogonal, quantize_Bbit


def entropy_bits(codes):
    total = 0.0
    for j in range(codes.shape[1]):
        _, cnt = np.unique(codes[:, j], return_counts=True)
        p = cnt / cnt.sum()
        total += float(-(p * np.log2(p)).sum())
    return total


def recall_of(qemb, o, qids, qrels, ids):
    o = np.ascontiguousarray(o, dtype=np.float32)
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    return recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0]


def pipeline(emb, qemb, qids, qrels, ids, order, keep, B, P):
    d = emb.shape[1]
    m = max(1, int(d * keep))
    sel = order[:m]
    x = emb[:, sel]
    rot = (x @ P).astype(np.float32) if m == d else (x @ random_orthogonal(m, seed=0)).astype(np.float32)
    codes, _ = quantize_Bbit(rot, B)
    H = entropy_bits(codes)
    half = ((2 ** B) - 1) / 2.0
    y = codes.astype(np.float64) - half
    P_use = P if m == d else random_orthogonal(m, seed=0)
    o = (P_use @ y.T).T
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    r = recall_of(qemb[:, sel].astype(np.float32), o, qids, qrels, ids)
    return r, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset_docs", type=int, default=None)
    ap.add_argument("--subset_queries", type=int, default=None)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    if args.subset_docs:
        emb, ids = emb[:args.subset_docs], ids[:args.subset_docs]
    if args.subset_queries:
        qemb, qids = qemb[:args.subset_queries], qids[:args.subset_queries]
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d, "reference": round(ref, 4)}
    imp = margin_importance(emb, qemb, qids, qrels, ids)
    perm_order = np.argsort(-imp)
    # 管线两个点: 12.5%+B1, 25%+B1（prefix 与 perm 各一）
    for keep in [0.125, 0.25]:
        m = max(1, int(d * keep))
        for name, order in [("prefix", np.arange(d)), ("perm", perm_order)]:
            r, H = pipeline(emb, qemb, qids, qrels, ids, order, keep, 1, None)
            res[f"keep={keep}_{name}_pipeline_recall"] = round(r, 4)
            res[f"keep={keep}_{name}_pipeline_bytes"] = round(H / 8, 1)
            print(f"pipeline keep={keep} {name}: recall {r:.4f}, entropy-bytes {H/8:.1f}B", flush=True)
    # 同预算 PQ/LSQ: 用两个管线点的字节（perm 的字节作对齐目标）
    for keep, key in [(0.125, "keep=0.125_perm"), (0.25, "keep=0.25_perm")]:
        target_bytes = res[f"{key}_pipeline_bytes"]
        M = max(1, int(round(target_bytes)))
        if M * 8 != round(target_bytes * 8):
            M = max(1, int(target_bytes))
        if d % M != 0:
            M = max(m for m in range(1, 97) if d % m == 0 and m <= target_bytes)
        for kind in ["pq", "lsq"]:
            if kind == "pq":
                q = faiss.IndexPQ(d, M, 8)
                q.train(emb)
                codes = q.sa_encode(emb)
                o = q.sa_decode(codes)
            else:
                q = faiss.LocalSearchQuantizer(d, M, 8)
                q.icm_iters = 5
                q.train_iters = 10
                q.train(emb)
                o = q.decode(q.compute_codes(emb))
            r = recall_of(qemb, o, qids, qrels, ids)
            res[f"{key}_{kind}_M{M}_recall"] = round(r, 4)
            print(f"{kind} M={M} ({M*8/8:.0f}B): recall {r:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
