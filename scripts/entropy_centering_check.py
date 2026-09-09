#!/usr/bin/env python3
"""entropy_centering_check.py — 审查点 4 验证（2026-08-17 用户转外部审查）:
"随机正交旋转后符号位熵应≈1bit，节省应≈0；E5>BGE>MiniLM 模式更像未做均值中心化"

对比: 不中心化（现协议）vs 中心化（减语料均值后旋转）:
- B=1/2 逐坐标码字熵（= 字节节省）
- recall@10（中心化是否伤害检索）
"""
import argparse, json
import numpy as np
from pilot_truncation import load, recall_at_k, brute_rank
from rabitq_py import random_orthogonal, quantize_Bbit


def entropy_bits(codes):
    total = 0.0
    for j in range(codes.shape[1]):
        _, cnt = np.unique(codes[:, j], return_counts=True)
        p = cnt / cnt.sum()
        total += float(-(p * np.log2(p)).sum())
    return total


def eval_recon(qemb, o, qids, qrels, ids):
    norms = np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    o = o / norms
    return recall_at_k(brute_rank(qemb.astype(np.float32), o.astype(np.float32)), qids, qrels, ids)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    mu = emb.mean(0)
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d,
           "mu_norm": round(float(np.linalg.norm(mu)), 4)}
    for B in [1, 2]:
        P = random_orthogonal(d, seed=0)
        for name, X in [("uncentered", emb), ("centered", emb - mu)]:
            rot = (X @ P).astype(np.float32)
            codes, _ = quantize_Bbit(rot, B)
            H = entropy_bits(codes)
            res[f"B={B}_{name}_entropy_bits"] = round(H, 1)
            res[f"B={B}_{name}_saving"] = round(1 - H / (d * B), 4)
            # 重建（中心化路径加回均值再归一化）
            half = ((2 ** B) - 1) / 2.0
            y = codes.astype(np.float64) - half
            o = (P @ y.T).T
            if name == "centered":
                o = o + mu
            r = eval_recon(qemb, o, qids, qrels, ids)
            res[f"B={B}_{name}_recall"] = round(r, 4)
            print(f"B={B} {name}: entropy {H:.1f}/{d*B} saving {1-H/(d*B):.3f} recall {r:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
