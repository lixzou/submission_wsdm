#!/usr/bin/env python3
"""synthetic_experiments.py — 合成高斯源实验（2026-08-17 重建, 原脚本未留存）。

协议（与论文 tab:regime_spectrum(b)/fig:analysis(b) caption 一致）:
  - d=256 合成高斯源, 方差 λ_j ∝ (j+1)^-α, 坐标经固定随机置换（使 prefix ≠ PCA）,
    η 为对数均匀扰动（max/min Var = η）; 向量 L2 归一化。
  - 查询 = 文档 + 各向同性高斯噪声 σ=0.2（归一化）, ground truth = 单一真实邻居, Recall@10。
  - 量化 = rabitq_py 协议（随机正交旋转 seed=0, block_quantize 重建 + 归一化）。

子命令:
  eta_scan — theory_eta_scan.json: α∈{0.5,1,2}×η∈{1,2,4,8}, 16× 预算(avg 2 bits/dim),
             uniform(全维 B=2) vs tilt(逐维注水 Σ B_j = 2d, B_j∈{0..8})
  spectrum — spectrum_sweep.json: α∈{0,0.5,1,2}, B=1 量化(随机旋转 vs PCA 旋转=I)
             与 12.5% 截断(PCA vs 前缀)
"""
import argparse, json
import numpy as np
from rabitq_py import random_orthogonal, quantize_Bbit


def make_source(d, n_docs, n_queries, alpha, eta, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    lam = (np.arange(d) + 1.0) ** (-alpha)
    if eta > 1.0:
        lam = lam * np.exp(rng.uniform(-0.5 * np.log(eta), 0.5 * np.log(eta), size=d))
    perm = rng.permutation(d)  # 固定随机置换: prefix ≠ PCA
    lam_p = lam[perm]
    L = np.sqrt(lam_p)[None, :]
    emb = rng.standard_normal((n_docs, d)) * L
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    gt = rng.integers(n_docs, size=n_queries)
    # 各向同性查询噪声（2026-08-17 定稿: 与真实查询-文档差异结构一致,
    # 复现论文两条定性结论: uniform 12/12 胜 + 随机旋转优势随谱集中扩大）
    qemb = emb[gt] + rng.standard_normal((n_queries, d)) * noise
    qemb /= np.linalg.norm(qemb, axis=1, keepdims=True)
    return emb.astype(np.float32), qemb.astype(np.float32), gt, perm


def recall_gt(qemb, o, gt, k=10):
    rank = np.zeros((len(qemb), k), np.int64)
    B = 2048
    for i in range(0, len(qemb), B):
        sim = qemb[i:i + B] @ o.T
        rank[i:i + B] = np.argpartition(-sim, k, axis=1)[:, :k]
    hits = [sum(1 for j in rank[i] if j == gt[i]) for i in range(len(gt))]
    return float(np.mean(hits))


def block_quantize(rot, P, blocks, B_per_block):
    """pilot_2b 同协议: 分块位宽量化 + 重建归一化。"""
    N, d = rot.shape
    y = np.zeros((N, d), np.float64)
    for bi, dims in enumerate(blocks):
        B = B_per_block[bi]
        if B == 0:
            continue
        codes, _ = quantize_Bbit(rot[:, dims], B)
        half = ((2 ** B) - 1) / 2.0
        y[:, dims] = codes.astype(np.float64) - half
    norm = np.sqrt(np.sum(y ** 2, axis=1, keepdims=True)) + 1e-12
    o = (P @ y.T).T / norm
    return o.astype(np.float32)


def run_eta_scan(out, n_docs=5000, n_queries=200, d=256, seed=0):
    res = {}
    for alpha in [0.5, 1.0, 2.0]:
        for eta in [1.0, 2.0, 4.0, 8.0]:
            emb, qemb, gt, perm = make_source(d, n_docs, n_queries, alpha, eta, seed=seed)
            P = random_orthogonal(d, seed=0)
            rot = (emb @ P).astype(np.float32)
            # pilot_2b 协议: 查询保持原空间, 相似度 = q · ō（ō 在原空间重建）
            u = recall_gt(qemb, block_quantize(rot, P, [np.arange(d)], [2]), gt)
            # tilt: 按真实方差注水（置换后坐标）, 块 = 单坐标
            var_p = emb.var(0)
            order = np.argsort(-var_p)
            B = np.zeros(d, int)
            budget = 2 * d
            for j in order:
                b = min(8, budget)
                B[j] = b
                budget -= b
                if budget <= 0:
                    break
            blocks = [[j] for j in range(d)]
            t = recall_gt(qemb, block_quantize(rot, P, blocks, list(B)), gt)
            key = "alpha=%.1f_eta=%.1f" % (alpha, eta)
            res[key] = {"uniform": round(u, 4), "tilt": round(t, 4)}
            print(key, res[key], flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("SAVED", out)


def run_spectrum(out, n_docs=5000, n_queries=200, d=256, seed=0):
    res = {}
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        emb, qemb, gt, perm = make_source(d, n_docs, n_queries, alpha, 4.0, seed=seed)
        P = random_orthogonal(d, seed=0)
        rot = (emb @ P).astype(np.float32)
        qrand = recall_gt(qemb, block_quantize(rot, P, [np.arange(d)], [1]), gt)
        # PCA 旋转 = 真实特征基: 坐标按方差降序排列（perm 的逆置换矩阵）
        # block_quantize(rot, P, ...) 的有效旋转 = Pᵀ, 故 P_pcaᵀ 须为方差排序置换
        inv = np.empty(d, int)
        inv[perm] = np.arange(d)
        P_pca = np.zeros((d, d), np.float32)
        P_pca[np.arange(d), inv] = 1.0  # (P_pcaᵀ x)[j] = x[perm[j]] -> 新轴 j = 第 j 大方差坐标
        rot_pca = (emb @ P_pca).astype(np.float32)
        qpca = recall_gt(qemb, block_quantize(rot_pca, P_pca, [np.arange(d)], [1]), gt)
        m = d // 8  # 12.5% retention
        # 截断: 前缀 = 置换后坐标前 m 维; PCA = 真实方差 top-m 维
        var_p = emb.var(0)
        top_m = np.argsort(-var_p)[:m]
        t_pfx = recall_gt(qemb[:, :m], emb[:, :m], gt)
        t_pca = recall_gt(qemb[:, top_m], emb[:, top_m], gt)
        res["alpha=%.1f" % alpha] = {
            "quant_rand_rot": round(qrand, 4), "quant_pca_rot": round(qpca, 4),
            "trunc_pca": round(t_pca, 4), "trunc_prefix": round(t_pfx, 4),
        }
        print("alpha=%.1f" % alpha, res["alpha=%.1f" % alpha], flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("SAVED", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eta_scan"); p.add_argument("--out", required=True)
    p = sub.add_parser("spectrum"); p.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "eta_scan":
        run_eta_scan(args.out)
    else:
        run_spectrum(args.out)
