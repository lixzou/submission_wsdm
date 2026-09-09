#!/usr/bin/env python3
"""turboquant_py.py — TurboQuant 论文版协议自实现（2026-08-17, self-implemented）。

依据 arXiv:2504.19874（Zandieh et al., TurboQuant）协议:
  1. 随机正交旋转 Π（输入向量 x -> y = Πx）
  2. 旋转后每个坐标服从 Beta 型分布（单位球面上 y_j^2 ~ Beta(1/2,(d-1)/2),
     坐标密度 f(y) ∝ (1-y^2)^{(d-3)/2} on [-1,1]）→ 逐坐标最优 Lloyd-Max 标量量化
     （K=2^b 级, 由分布密度数值求解 1D k-means: 边界=中点, 质心=条件均值, 迭代至收敛）
  3. 逐坐标码字熵编码（与论文 C3 同口径的字节节省）
检索协议与论文一致: decode(反旋转)+normalize+cosine, Recall@10。
"""
import argparse, json, time
import numpy as np
from scipy import integrate
from pilot_truncation import load, recall_at_k, brute_rank
from rabitq_py import random_orthogonal


def beta_coord_density(d):
    """单位球面上旋转后单坐标的密度函数（未归一化, ∝ (1-x^2)^((d-3)/2)）。"""
    def f(x):
        if x <= -1.0 or x >= 1.0:
            return 0.0
        return (1.0 - x * x) ** ((d - 3) / 2.0)
    return f


def lloyd_max(d, K, tol=1e-6, max_iter=500):
    """对坐标密度做 1D Lloyd-Max（连续 1 维 k-means）: 返回 K 个重建级。"""
    f = beta_coord_density(d)
    # 初值: 样本点 k-means++ 风格（用离散化密度取样）
    xs = np.linspace(-0.999, 0.999, 20001)
    p = np.array([f(x) for x in xs])
    p /= p.sum()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xs), size=K, replace=False, p=p)
    levels = np.sort(xs[idx])
    for it in range(max_iter):
        bounds = np.concatenate([[-1.0], (levels[:-1] + levels[1:]) / 2.0, [1.0]])
        new_levels = np.zeros(K)
        for k in range(K):
            a, b = bounds[k], bounds[k + 1]
            num, _ = integrate.quad(lambda x: x * f(x), a, b, limit=200)
            den, _ = integrate.quad(lambda x: f(x), a, b, limit=200)
            new_levels[k] = num / den if den > 1e-300 else (a + b) / 2.0
        if np.max(np.abs(new_levels - levels)) < tol:
            break
        levels = new_levels
    return levels


def quantize_recall(emb, qemb, qids, qrels, ids, B):
    d = emb.shape[1]
    P = random_orthogonal(d, seed=0)
    rot = (emb @ P).astype(np.float64)
    levels = lloyd_max(d, 2 ** B)
    # 逐坐标最近级编码
    codes = np.zeros_like(rot, dtype=np.int32)
    for j in range(d):
        codes[:, j] = np.argmin(np.abs(rot[:, j, None] - levels[None, :]), axis=1)
    # 重建: 逐坐标级向量（levels 高级索引）-> 反旋转 + 归一化
    recon_rot = np.asarray(levels)[codes]  # (N,d)
    o = (P @ recon_rot.T).T
    o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    # 熵（与 C3 同口径: 逐坐标码字 Shannon 熵 / (d*B)）
    Hs = 0.0
    for j in range(d):
        _, cnt = np.unique(codes[:, j], return_counts=True)
        pr = cnt / cnt.sum()
        Hs += float(-(pr * np.log2(pr)).sum())
    r = recall_at_k(brute_rank(qemb.astype(np.float32), o.astype(np.float32)), qids, qrels, ids)[0]
    return r, Hs, d * B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Bs", nargs="+", type=int, default=[1, 2, 4])
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
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d,
           "reference": round(ref, 4), "protocol": "TurboQuant (2504.19874): random rotation + per-coordinate Lloyd-Max on Beta coordinates + entropy"}
    for B in args.Bs:
        t0 = time.time()
        r, Hs, fixed = quantize_recall(emb, qemb, qids, qrels, ids, B)
        res[f"B={B}"] = {"recall": round(r, 4), "entropy_bits": round(Hs, 1),
                         "byte_saving": round(1 - Hs / fixed, 4)}
        print(f"B={B}: recall {r:.4f}, entropy {Hs:.1f}/{fixed}, saving {1-Hs/fixed:.3f} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
