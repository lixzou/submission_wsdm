#!/usr/bin/env python3
"""sota_pipeline_check.py — 全管线 SOTA 冲刺（用户指令 2026-08-19）:
perm 50% 截断 + 校准 Lloyd-Max B2 网格（排序目标）@ 32x 总字节 vs PQ/LSQ/RQ。

网格拟合: judged 相关对上逐维 Lloyd-Max（排序目标, beat_tq_v2 的 calib 复用）。
"""
import argparse, json, time
import numpy as np
import faiss
from scipy import integrate
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from rabitq_py import random_orthogonal


def lloyd_max_from_samples(vals, K, tol=1e-6, max_iter=300):
    vals = np.asarray(vals)
    levels = np.quantile(vals, np.linspace(0, 1, K))
    for it in range(max_iter):
        c = np.argmin(np.abs(vals[:, None] - levels[None, :]), axis=1)
        new = np.array([vals[c == k].mean() if np.any(c == k) else levels[k] for k in range(K)])
        if np.max(np.abs(new - levels)) < tol: break
        levels = new
    return np.sort(levels)


def fit_calib_levels(rot_emb, qids, qrels, ids, B, calib_frac=1.0):
    """在旋转坐标上拟合逐维 Lloyd-Max（与量化同坐标系, bug 修复 2026-08-19）。"""
    qidx = {q: i for i, q in enumerate(qids)}
    judged = [q for q in qids if q in qrels]
    n_cal = max(1, int(len(judged) * calib_frac))
    cal_q = judged[:n_cal]
    idset = {s: i for i, s in enumerate(ids)}
    rel_vals = []
    for qid in cal_q:
        qi = qidx[qid]
        for did in qrels[qid]:
            if did in idset:
                rel_vals.append(rot_emb[idset[did]])
    rel_vals = np.array(rel_vals)
    K = 2 ** B
    levels = np.zeros((rot_emb.shape[1], K))
    for j in range(rot_emb.shape[1]):
        levels[j] = lloyd_max_from_samples(rel_vals[:, j], K)
    return levels


def quant_recall(qemb, emb, sel, B, levels, qids, qrels, ids, return_arrays=False):
    m = len(sel)
    P = random_orthogonal(m, seed=0)
    rot = (emb[:, sel] @ P).astype(np.float64)
    codes = np.zeros((len(rot), m), np.int32)
    for j in range(m):
        codes[:, j] = np.argmin(np.abs(rot[:, j, None] - levels[j][None, :]), axis=1)
    recon = levels[np.arange(m)[None, :], codes]
    o = (P @ recon.T).T
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    rec = recall_at_k(brute_rank(qemb[:, sel].astype(np.float32), o.astype(np.float32)),
                      qids, qrels, ids, return_arrays=return_arrays)
    return rec[2] if return_arrays else rec[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--B", type=int, default=2)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d, "reference": round(ref, 4)}
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=1.0)
    perm_order = np.argsort(-imp)
    m = max(1, int(d * args.keep))
    for name, order in [("prefix", np.arange(d)), ("perm", perm_order)]:
        sel = order[:m]
        P = random_orthogonal(m, seed=0)
        rot_emb = (emb[:, sel] @ P).astype(np.float64)
        levels = fit_calib_levels(rot_emb, qids, qrels, ids, args.B)
        r = quant_recall(qemb, emb, sel, args.B, levels, qids, qrels, ids)
        res[f"{name}_{args.keep}_calibB{args.B}"] = round(r, 4)
        print(f"{name} keep={args.keep} calib B{args.B}: {r:.4f}", flush=True)
    # PQ 同预算对照（m*B/8 = 字节）
    M = max(1, int(d * args.keep * args.B / 8))
    if d % M == 0:
        q = faiss.IndexPQ(d, M, 8)
        q.train(emb)
        o = q.sa_decode(q.sa_encode(emb))
        o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
        r = recall_at_k(brute_rank(qemb, o.astype(np.float32)), qids, qrels, ids)[0]
        res[f"pq_M{M}"] = round(r, 4)
        print(f"PQ M={M} ({M}B): {r:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
