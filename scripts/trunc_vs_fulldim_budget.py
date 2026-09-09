#!/usr/bin/env python3
"""trunc_vs_fulldim_budget.py — 同字节预算: 截断+精细量化 vs 全维+粗量化。

回答: 截断的损失能否被"更精细量化剩余维"补偿？
预算 32x (96B): 全维B1 | perm/prefix 12.5%+B8 | 25%+B4 | 50%+B2
预算 16x (192B): 全维B2 | perm/prefix 25%+B8 | 50%+B4
预算 8x  (384B): 全维B4 | perm/prefix 50%+B8
协议: 截断后随机旋转(seed=0) + B 位半整数网格 + 归一化重建。
"""
import argparse, json
import numpy as np
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from rabitq_py import random_orthogonal, quantize_Bbit


def quant_recall(qemb, emb, sel, B, qids, qrels, ids):
    m = len(sel)
    P = random_orthogonal(m, seed=0)
    rot = (emb[:, sel] @ P).astype(np.float32)
    codes, _ = quantize_Bbit(rot, B)
    y = codes.astype(np.float64) - ((2**B - 1) / 2.0)
    o = (P @ y.T).T
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    return recall_at_k(brute_rank(qemb[:, sel].astype(np.float32), o.astype(np.float32)),
                       qids, qrels, ids)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib_frac", type=float, default=1.0)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d, "reference": round(ref, 4)}
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=args.calib_frac)
    perm_order = np.argsort(-imp)
    for keep, B in [(0.125, 8), (0.25, 4), (0.5, 2)]:  # 32x 预算: m*B/8 = 96B
        m = max(1, int(d * keep))
        for name, order in [("prefix", np.arange(d)), ("perm", perm_order)]:
            sel = order[:m]
            r = quant_recall(qemb, emb, sel, B, qids, qrels, ids)
            res[f"32x_keep{keep}_{name}_B{B}"] = round(r, 4)
            print(f"32x keep={keep} {name} B={B}: {r:.4f}", flush=True)
    # 全维 32x = B1
    r = quant_recall(qemb, emb, np.arange(d), 1, qids, qrels, ids)
    res["32x_full_B1"] = round(r, 4)
    print(f"32x full B1: {r:.4f}", flush=True)
    # 16x 预算: 25%+B8, 50%+B4, 全维 B2
    for keep, B in [(0.25, 8), (0.5, 4)]:
        m = max(1, int(d * keep))
        for name, order in [("prefix", np.arange(d)), ("perm", perm_order)]:
            sel = order[:m]
            r = quant_recall(qemb, emb, sel, B, qids, qrels, ids)
            res[f"16x_keep{keep}_{name}_B{B}"] = round(r, 4)
            print(f"16x keep={keep} {name} B={B}: {r:.4f}", flush=True)
    r = quant_recall(qemb, emb, np.arange(d), 2, qids, qrels, ids)
    res["16x_full_B2"] = round(r, 4)
    print(f"16x full B2: {r:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
