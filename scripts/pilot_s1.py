#!/usr/bin/env python3
"""Pilot 1b: RATC S1 完整版——相关性分桶 + 桶内白化 + 桶间重要性排序 + 截断
vs 前缀 / 随机 / 全局PCA / 逐维margin置换。"""
import argparse, json, os
import numpy as np
from sklearn.cluster import SpectralClustering
sys_path = "/ssd1/zoulixin/tencent_previous_compression/experiments"
import sys
sys.path.insert(0, sys_path)
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bucket_size", type=int, default=64)
    ap.add_argument("--out", default="/tmp/pilot_s1.json")
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    n_buckets = d // args.bucket_size
    print("model", args.model, "| d", d, "| buckets", n_buckets, flush=True)

    imp = margin_importance(emb, qemb, qids, qrels, ids)

    # 相关性分桶: 协方差相关图谱聚类
    X = emb - emb.mean(0)
    cov = X.T @ X / len(X)
    std = np.sqrt(np.diag(cov)) + 1e-8
    rho = np.abs(cov) / np.outer(std, std)
    np.fill_diagonal(rho, 0)
    aff = rho  # 谱聚类亲和度
    sc = SpectralClustering(n_clusters=n_buckets, affinity="precomputed",
                            assign_labels="kmeans", random_state=0, n_init=10)
    labels = sc.fit_predict(aff)
    buckets = [np.where(labels == b)[0] for b in range(n_buckets)]
    sizes = [len(b) for b in buckets]
    print("bucket sizes:", sorted(sizes, reverse=True), flush=True)

    # 桶内白化（每桶 PCA 旋转，保持内积）+ 桶间重要性排序
    bucket_order = sorted(range(n_buckets),
                          key=lambda b: -float(imp[buckets[b]].mean()))
    # 全局 PCA（对照）
    mu = emb.mean(0)
    U, S, Vt = np.linalg.svd(emb - mu, full_matrices=False)

    results = {}
    for keep in [0.5, 0.25, 0.125]:
        m = max(1, int(d * keep))
        # 1. prefix
        results["keep=%.3f" % keep] = {
            "prefix": recall_at_k(brute_rank(qemb[:, :m], emb[:, :m]), qids, qrels, ids)[0],
            "pca": recall_at_k(brute_rank(qemb @ Vt[:m].T, emb @ Vt[:m].T), qids, qrels, ids)[0],
        }
        # 2. RATC-S1: 取重要性最高的桶直到 >= m 维，桶内白化
        sel, cur = [], 0
        for b in bucket_order:
            if cur >= m: break
            sel.append(b); cur += len(buckets[b])
        sel_dims = np.concatenate([buckets[b] for b in sel])[:m]
        # 桶内白化: 对每个选中桶做 PCA 旋转（在白化桶的维度上）
        out_emb, out_qemb = [], []
        for b in sel:
            bd = buckets[b]
            bX = emb[:, bd] - emb[:, bd].mean(0)
            _, _, bVt = np.linalg.svd(bX, full_matrices=False)
            out_emb.append(bX @ bVt.T)
            out_qemb.append((qemb[:, bd] - emb[:, bd].mean(0)) @ bVt.T)
        we_emb = np.concatenate(out_emb, axis=1)[:, :m]
        we_qemb = np.concatenate(out_qemb, axis=1)[:, :m]
        results["keep=%.3f" % keep]["ratc_s1"] = recall_at_k(
            brute_rank(we_qemb, we_emb), qids, qrels, ids)[0]
        # 3. 逐维 margin 置换（上轮 pilot 的 best 对照）
        order = np.argsort(-imp)
        results["keep=%.3f" % keep]["perm_margin"] = recall_at_k(
            brute_rank(qemb[:, order[:m]], emb[:, order[:m]]), qids, qrels, ids)[0]
        print("keep=%.3f:" % keep, {k: round(v, 4) for k, v in results["keep=%.3f" % keep].items()}, flush=True)

    json.dump(results, open(args.out, "w"), indent=1)
    print("PILOT S1 DONE ->", args.out, flush=True)

if __name__ == "__main__":
    main()
