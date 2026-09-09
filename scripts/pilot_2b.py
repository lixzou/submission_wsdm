#!/usr/bin/env python3
"""Pilot 2b（修正版位宽分配）: 用校验过的 rabitq_py 协议做块级位宽分配。
策略 @ 固定总 bit 预算: uniform / margin块注水 / 方差块(SAQ式) / ratc(S1相关性桶+S2)
度量: recall@10（量化重建向量暴力余弦检索）
"""
import sys, json, argparse, time
sys.path.insert(0, "/ssd1/zoulixin/tencent_previous_compression/experiments")
import numpy as np
from rabitq_py import random_orthogonal, quantize_Bbit
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance

def block_quantize(rot, P, emb, blocks, B_per_block):
    """按块位宽量化: blocks = 块定义（dim 索引列表的列表），B_per_block 对应每块位宽。
    返回重建向量 (N,d)。"""
    N, d = rot.shape
    y = np.zeros((N, d), np.float64)
    for bi, dims in enumerate(blocks):
        B = B_per_block[bi]
        if B == 0: continue
        codes, _ = quantize_Bbit(rot[:, dims], B)
        half = ((2 ** B) - 1) / 2.0
        y[:, dims] = codes.astype(np.float64) - half
    norm = np.sqrt(np.sum(y ** 2, axis=1, keepdims=True)) + 1e-12
    o = (P @ y.T).T / norm
    return o.astype(np.float32)

def waterfill(imp_order, d, avg_bits, block_size):
    """按重要性序分块注水: 每块 b ∈ {8,4,2,1,0}"""
    budget = avg_bits * d
    B = np.zeros(d, int)
    nblocks = d // block_size
    for b_i in range(nblocks):
        for b in [8, 4, 2, 1]:
            if budget >= b * block_size:
                s, e = b_i * block_size, (b_i + 1) * block_size
                B[imp_order[s:e]] = b
                budget -= b * block_size
                break
    return B

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/pilot_2b.json")
    # NQ 5k 子集口径（2026-08-17 补充, 与 tab:c2_all 声明一致）:
    ap.add_argument("--subset_docs", type=int, default=None)
    ap.add_argument("--subset_queries", type=int, default=None)
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    if args.subset_docs:
        emb = emb[:args.subset_docs]
        ids = ids[:args.subset_docs]
    if args.subset_queries:
        qemb = qemb[:args.subset_queries]
        qids = qids[:args.subset_queries]
    N, d = emb.shape
    imp = margin_importance(emb, qemb, qids, qrels, ids)
    P = random_orthogonal(d, seed=7)
    rot = (P.T @ emb.T).T.astype(np.float64)

    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    print("reference:", round(ref, 4), flush=True)

    BS = 64
    nblocks = d // BS
    # 策略的块定义
    margin_order = np.argsort(-imp)
    var_order = np.argsort(-np.var(emb, 0))
    # 相关性桶（谱聚类）
    from sklearn.cluster import SpectralClustering
    X = emb - emb.mean(0)
    cov = X.T @ X / len(X)
    std = np.sqrt(np.diag(cov)) + 1e-8
    rho = np.abs(cov) / np.outer(std, std)
    np.fill_diagonal(rho, 0)
    labels = SpectralClustering(n_clusters=nblocks, affinity="precomputed",
                                assign_labels="kmeans", random_state=0, n_init=10).fit_predict(rho)
    buckets = [np.where(labels == b)[0] for b in range(nblocks)]
    bucket_imp = np.array([imp[b].mean() for b in buckets])
    bucket_order = np.argsort(-bucket_imp)

    contig_blocks = [np.arange(b * BS, (b + 1) * BS) for b in range(nblocks)]
    margin_blocks = [margin_order[b * BS:(b + 1) * BS] for b in range(nblocks)]
    var_blocks = [var_order[b * BS:(b + 1) * BS] for b in range(nblocks)]
    ratc_blocks = [buckets[bi] for bi in bucket_order]

    results = {"reference": ref}
    for avg_bits in [4, 2, 1]:
        t0 = time.time()
        res = {}
        # uniform
        Bu = np.full(nblocks, avg_bits)
        o = block_quantize(rot, P, emb, contig_blocks, Bu)
        res["uniform"] = round(recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0], 4)
        # margin-block
        Bm = waterfill(margin_order, d, avg_bits, BS)
        Bm_blocks = [int(np.median(Bm[bl])) for bl in margin_blocks]
        o = block_quantize(rot, P, emb, margin_blocks, Bm_blocks)
        res["margin_block"] = round(recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0], 4)
        # var-block (SAQ 式)
        Bv = waterfill(var_order, d, avg_bits, BS)
        Bv_blocks = [int(np.median(Bv[bl])) for bl in var_blocks]
        o = block_quantize(rot, P, emb, var_blocks, Bv_blocks)
        res["var_block"] = round(recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0], 4)
        # RATC: 相关性桶按重要性序 + 桶内位宽 = 桶平均 margin 的注水
        nb = len(ratc_blocks)
        block_bits = np.zeros(nb, int)
        budget = avg_bits * d
        for bi in range(nb):
            for b in [8, 4, 2, 1]:
                sz = len(ratc_blocks[bi])
                if budget >= b * sz:
                    block_bits[bi] = b; budget -= b * sz; break
        o = block_quantize(rot, P, emb, ratc_blocks, block_bits)
        res["ratc_bucket"] = round(recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0], 4)
        results["avg_bits=%d" % avg_bits] = res
        print("avg_bits=%d:" % avg_bits, res, "[%.0fs]" % (time.time() - t0), flush=True)

    json.dump(results, open(args.out, "w"), indent=1)
    print("PILOT 2B DONE ->", args.out)

if __name__ == "__main__":
    main()
