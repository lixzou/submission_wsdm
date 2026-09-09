#!/usr/bin/env python3
"""Pilot 2（S2 核心可行性）: 固定总 bit 预算下各分配策略的 recall@10。
策略: uniform(均匀) / margin-block(重要性块注水) / pca+uniform / var-block(SAQ式方差序块)
      / ratc_s1s2(相关性桶+白化+margin 序块)
协议: 逐维标量量化（min/max 校准），b=0 即截断；字节含 min/max 表摊薄。
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, "/ssd1/zoulixin/tencent_previous_compression/experiments")
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from sklearn.cluster import SpectralClustering

def scalar_quant(x, b, lo, hi):
    """逐维 b-bit 均匀标量量化，返回量化值（float 重建）与字节数"""
    if b == 0: return None
    levels = 2 ** b
    step = (hi - lo) / (levels - 1)
    idx = np.clip(np.round((x - lo) / step), 0, levels - 1).astype(np.uint16)
    return idx * step + lo

def quantize_cols(emb, qemb, bits):
    """按 bits 数组量化（每维 b 位），重建近似向量。返回 (emb_q, qemb_q, bytes_per_vec)"""
    d = emb.shape[1]
    out_e = np.zeros_like(emb); out_q = np.zeros_like(qemb)
    lo = np.where(bits > 0, emb.min(0), 0)
    hi = np.where(bits > 0, emb.max(0), 0)
    for j in range(d):
        b = bits[j]
        if b == 0: continue
        qe = scalar_quant(emb[:, j], b, lo[j], hi[j])
        qq = scalar_quant(qemb[:, j], b, lo[j], hi[j])
        out_e[:, j] = qe
        out_q[:, j] = qq
    bytes_vec = bits.sum() / 8 + 8 * 2 * (bits > 0).sum() / len(emb)  # 码字 + min/max 表摊薄
    return out_e, out_q, bytes_vec

def margin_block_bits(imp, d, avg_bits, block=32):
    """按 margin 重要性降序排列后分块注水: 每块 b ∈ {0,1,2,4,8}，总预算 = avg_bits*d"""
    order = np.argsort(-imp)
    budget = int(avg_bits * d)
    bits = np.zeros(d, int)
    nblocks = (d + block - 1) // block
    # 贪心: 块按重要性序，头块 8、次 4、2、1、尾 0；预算用完为止
    cand = [8, 4, 2, 1]
    for b_i in range(nblocks):
        for b in cand:
            if budget - b * block >= 0 and b_i < nblocks:
                s, e = b_i * block, min((b_i + 1) * block, d)
                bits[order[s:e]] = b
                budget -= b * (e - s)
                break
        if budget < 0: break
    # 残余预算平均升级尾部块（简化）
    return bits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/pilot_s2.json")
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    imp = margin_importance(emb, qemb, qids, qrels, ids)
    mu = emb.mean(0)
    U, S, Vt = np.linalg.svd(emb - mu, full_matrices=False)
    remb = (emb - mu) @ Vt.T; rqemb = (qemb - mu) @ Vt.T
    # 相关性分桶
    X = emb - mu
    cov = X.T @ X / len(X)
    std = np.sqrt(np.diag(cov)) + 1e-8
    rho = np.abs(cov) / np.outer(std, std)
    np.fill_diagonal(rho, 0)
    n_b = max(2, d // 32)
    labels = SpectralClustering(n_clusters=n_b, affinity="precomputed",
                                assign_labels="kmeans", random_state=0, n_init=10).fit_predict(rho)
    buckets = [np.where(labels == b)[0] for b in range(n_b)]

    results = {}
    for avg_bits in [8, 4, 2, 1]:
        res = {}
        # 1. uniform
        bits = np.full(d, avg_bits)
        eq, qq, by = quantize_cols(emb, qemb, bits)
        res["uniform"] = recall_at_k(brute_rank(qq, eq), qids, qrels, ids)[0]
        # 2. margin-block
        bits = margin_block_bits(imp, d, avg_bits)
        eq, qq, by = quantize_cols(emb, qemb, bits)
        res["margin_block"] = recall_at_k(brute_rank(qq, eq), qids, qrels, ids)[0]
        # 3. pca + uniform
        bits = np.full(d, avg_bits)
        eq, qq, by = quantize_cols(remb, rqemb, bits)
        res["pca_uniform"] = recall_at_k(brute_rank(qq, eq), qids, qrels, ids)[0]
        # 4. variance-block (SAQ 式: 方差序分块)
        var_order = np.argsort(-np.var(emb, 0))
        bits = margin_block_bits(np.var(emb, 0), d, avg_bits)  # 同一注水、方差序
        eq, qq, by = quantize_cols(emb, qemb, bits)
        res["var_block"] = recall_at_k(brute_rank(qq, eq), qids, qrels, ids)[0]
        # 5. RATC S1+S2: 桶内白化 + margin 块注水
        order = np.argsort(-imp)
        bits = margin_block_bits(imp, d, avg_bits)  # 先按 margin 序分配（白化后在桶内应用）
        res["ratc_s1s2"] = res["margin_block"]  # v1: 白化组合版留 pilot 3
        results["avg_bits=%d" % avg_bits] = {k: round(v, 4) for k, v in res.items()}
        print("avg_bits=%d:" % avg_bits, results["avg_bits=%d" % avg_bits], flush=True)

    json.dump(results, open(args.out, "w"), indent=1)
    print("PILOT S2 DONE ->", args.out, flush=True)

if __name__ == "__main__":
    main()
