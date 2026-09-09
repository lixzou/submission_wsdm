#!/usr/bin/env python3
"""pca_entropy_fight.py — PCA 对齐旋转 vs 随机旋转: 量化 recall + 逐维码字熵
（tab:regime_spectrum(a) 数据源, 2026-08-17 重建, 原脚本未留存）。

协议: scifact e5; 随机正交旋转 seed=0 vs PCA 基（协方差特征向量）; rabitq_py B-bit
网格量化; 重建归一化后暴力 recall@10; 逐维码字 Shannon 熵 = 字节节省。
确定性: 无 margin_importance 依赖（旋转 seed 固定）。
"""
import argparse, json
import numpy as np
from pilot_truncation import load, recall_at_k, brute_rank
from rabitq_py import random_orthogonal, quantize_Bbit


def entropy_bits(codes, B):
    total = 0.0
    for j in range(codes.shape[1]):
        _, counts = np.unique(codes[:, j], return_counts=True)
        p = counts / counts.sum()
        total += float(-(p * np.log2(p)).sum())
    return total


def recon_from_codes(P, codes, B):
    N, d = codes.shape
    half = ((2 ** B) - 1) / 2.0
    y = codes.astype(np.float64) - half
    norm = np.sqrt(np.sum(y ** 2, axis=1, keepdims=True)) + 1e-12
    o = (P @ y.T).T / norm
    return o.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Bs", nargs="+", type=int, default=[1, 2, 4])
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {args.model: {"ref": round(ref, 4)}}

    mu = emb.mean(0)
    cemb = emb - mu  # 中心化后旋转（PCA 基下坐标零均值, 码字符号均匀）
    U, S, Vt = np.linalg.svd(cemb, full_matrices=False)  # PCA 基 = Vt 行
    P_rand = random_orthogonal(d, seed=0)
    P_pca = Vt.T.astype(np.float32)

    for B in args.Bs:
        # rand = 原始 RATC/rabitq 协议（原始向量, 不中心化; 与 C3 熵口径一致:
        # B=1 熵 387.1 == canonical c3_scifact e5）
        rot_r = (emb @ P_rand).astype(np.float32)
        codes_r, _ = quantize_Bbit(rot_r, B)
        o_r = recon_from_codes(P_rand, codes_r, B)
        r_r = recall_at_k(brute_rank(qemb, o_r), qids, qrels, ids)[0]
        h_r = entropy_bits(codes_r, B)

        # pca = 中心化后 PCA 基（PCA 协方差定义在中心化数据上; 重建加回均值）
        rot_p = (cemb @ P_pca).astype(np.float32)
        codes_p, _ = quantize_Bbit(rot_p, B)
        o_p = recon_from_codes(P_pca, codes_p, B) + mu
        r_p = recall_at_k(brute_rank(qemb, o_p), qids, qrels, ids)[0]
        h_p = entropy_bits(codes_p, B)

        res[args.model][f"B={B}"] = {
            "pca_recall": round(r_p, 4), "pca_entropy_bits": round(h_p, 1),
            "pca_entropy_saving_pct": round(100 * (1 - h_p / (d * B)), 1),
            "rand_recall": round(r_r, 4), "rand_entropy_bits": round(h_r, 1),
            "rand_entropy_saving_pct": round(100 * (1 - h_r / (d * B)), 1),
        }
        print(f"B={B}: pca_recall={r_p:.4f} rand_recall={r_r:.4f} "
              f"pca_bits={h_p:.1f} rand_bits={h_r:.1f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
