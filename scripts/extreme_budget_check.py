#!/usr/bin/env python3
"""extreme_budget_check.py — 极端压缩区（2-6B 总字节）全管线 vs 同预算 VQ。

管线: perm 截断 + B=1 标量 + 熵（总字节 = 实测熵/8）; VQ: PQ/LSQ 同总字节 m 子量化器。
关键: 该区 VQ 质心数 ≤ 6×256=1536 << 语料, 码本记忆消失, 比较公平。
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
    # 管线: keep 6.25% / 12.5% / 25% + B=1（prefix 与 perm）
    for keep in [0.0625, 0.125, 0.25]:
        m = max(1, int(d * keep))
        P = random_orthogonal(m, seed=0)
        for name, order in [("prefix", np.arange(d)), ("perm", perm_order)]:
            sel = order[:m]
            rot = (emb[:, sel] @ P).astype(np.float32)
            codes, _ = quantize_Bbit(rot, 1)
            H = entropy_bits(codes)
            y = codes.astype(np.float64) - 0.5
            o = (P @ y.T).T
            o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
            r = recall_of(qemb[:, sel].astype(np.float32), o, qids, qrels, ids)
            res[f"keep={keep}_{name}_recall"] = round(r, 4)
            res[f"keep={keep}_{name}_bytes"] = round(H / 8, 2)
            print(f"pipe keep={keep} {name}: recall {r:.4f} @ {H/8:.2f}B", flush=True)
    # VQ 同预算: 2/3/4/6/8/12B（M 整除 d）
    for B_target in [2, 3, 4, 6, 8, 12]:
        Ms = [m for m in range(1, 97) if d % m == 0 and m * 1 == B_target]
        if not Ms:
            continue
        M = Ms[0]
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
            res[f"{kind}_M{M}_{B_target}B_recall"] = round(r, 4)
            print(f"{kind} M={M} ({B_target}B): {r:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
