#!/usr/bin/env python3
"""Spectral-guided PCA 完整管线 vs FAISS baselines @ 同字节预算.

管线: SG-PCA 投影(d->m) -> 随机正交旋转(m) -> 校准 Lloyd-Max 量化(B bits) -> 重建
对照: perm 截断+同量化 (现有 SOTA 管线) / byte-aligned PQ / 全维 B=1
字节预算: keep*B bits/dim, 32x => m*B/8 = 96 bytes (768d), 与 PQ 同预算对齐.

用法: python sota_spectral_pipeline.py --ds_dir data/beir/scifact --model e5 --out results/runs/... --keep 0.5 --B 2
"""
import argparse, json, numpy as np, faiss, sys
sys.path.insert(0, '.')
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from rabitq_py import random_orthogonal
from sota_pipeline_check import fit_calib_levels, quant_recall, lloyd_max_from_samples
from merge_pca_retrieval import pca_directions

def sg_pca_directions(emb, imp, m, alpha=1.0, sel_K=None):
    """SG-PCA: importance 加权 SVD, 返回 (m,d) 投影. alpha 控制加权强度."""
    if sel_K is not None:
        sel = np.argsort(-imp)[:sel_K]
        emb_sel = emb[:, sel]
        V = pca_directions(emb_sel, m, w=(imp[sel]/imp[sel].max()+1e-6))
        # 需要把 m 维映射回原维度? 不, 我们直接对 sel 子空间投影
        return V, sel
    V = pca_directions(emb, m, w=(imp/imp.max()+1e-6))
    return V, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--sel_K", type=int, default=0, help="0=off; >0 用选子空间版")
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d,
           "keep": args.keep, "B": args.B, "reference": round(ref, 4)}
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=1.0)
    m = max(1, int(d * args.keep))
    perm_order = np.argsort(-imp)

    # 1. perm 截断 + 校准量化 (现有 SOTA 管线)
    sel_p = perm_order[:m]
    P = random_orthogonal(m, seed=0)
    rot_perm = (emb[:, sel_p] @ P).astype(np.float64)
    levels = fit_calib_levels(rot_perm, qids, qrels, ids, args.B)
    res[f"perm_calibB{args.B}"] = round(quant_recall(qemb, emb, sel_p, args.B, levels, qids, qrels, ids), 4)

    # 2. SG-PCA 投影 + 校准量化
    if args.sel_K > 0:
        K = min(d, max(m, args.sel_K))
        V, sel_s = sg_pca_directions(emb, imp, m, args.alpha, sel_K=K)
        # sel 子空间 SG-PCA: 先选 K 维, 投影到 m
        emb_proj = emb[:, sel_s] @ V.T  # (N, m)
        q_proj = qemb[:, sel_s] @ V.T
        sel_q = np.arange(m)  # 已经是 m 维
        rot_sg = (emb_proj @ P).astype(np.float64)
        levels_sg = fit_calib_levels(rot_sg, qids, qrels, ids, args.B)
        res[f"sgsel_K{K}_calibB{args.B}"] = round(
            quant_recall(q_proj, emb_proj, sel_q, args.B, levels_sg, qids, qrels, ids), 4)
    else:
        V = pca_directions(emb, m, w=(imp/imp.max()+1e-6))
        emb_proj = emb @ V.T; q_proj = qemb @ V.T
        rot_sg = (emb_proj @ P).astype(np.float64)
        levels_sg = fit_calib_levels(rot_sg, qids, qrels, ids, args.B)
        res[f"sg_calibB{args.B}"] = round(
            quant_recall(q_proj, emb_proj, np.arange(m), args.B, levels_sg, qids, qrels, ids), 4)

    # 3. byte-aligned PQ 同预算
    M = max(1, int(d * args.keep * args.B / 8))
    if d % M == 0:
        q = faiss.IndexPQ(d, M, 8)
        q.train(emb)
        o = q.sa_decode(q.sa_encode(emb))
        o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
        res[f"pq_M{M}"] = round(recall_at_k(brute_rank(qemb, o.astype(np.float32)), qids, qrels, ids)[0], 4)

    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{args.model} keep={args.keep} B={args.B}:", json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
