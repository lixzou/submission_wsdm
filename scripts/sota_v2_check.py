#!/usr/bin/env python3
"""sota_v2_check.py — SOTA v2: 中心化 + 校准网格 + 扫掠步长 三增益叠加（用户: 全指标最优）。

配置: perm 50% 截断 + 中心化（减语料均值）+ 校准 Lloyd-Max B2 + 每向量扫掠最优缩放。
"""
import argparse, json
import numpy as np
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--B", type=int, default=2)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"model": args.model, "reference": round(ref, 4)}
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=1.0)
    perm_order = np.argsort(-imp)
    m = max(1, int(d * args.keep))
    sel = perm_order[:m]
    mu = emb[:, sel].mean(0)
    X = emb[:, sel] - mu           # 中心化
    Q = qemb[:, sel]               # 查询不中心化（检索时查询侧）
    # 校准网格: 旋转后中心化坐标上拟合
    P = random_orthogonal(m, seed=0)
    rot = (X @ P).astype(np.float64)
    qidx = {q: i for i, q in enumerate(qids)}
    judged = [q for q in qids if q in qrels]
    cal_q = judged[:len(judged)]
    idset = {s: i for i, s in enumerate(ids)}
    rel_vals = []
    for qid in cal_q:
        for did in qrels[qid]:
            if did in idset:
                rel_vals.append(rot[idset[did]])
    rel_vals = np.array(rel_vals)
    K = 2 ** args.B
    levels = np.array([lloyd_max_from_samples(rel_vals[:, j], K) for j in range(m)])
    # 编码 + 扫掠（每向量最优缩放, 中心化坐标上）
    cand_s = np.linspace(0.3, 2.5, 30)
    codes = np.zeros((len(rot), m), np.int32)
    for i in range(len(rot)):
        best_c2, best_c = -1, None
        for s in cand_s:
            c = np.argmin(np.abs(rot[i][None, :] - (levels * s).T), axis=0)
            r = levels[np.arange(m), c] * s
            c2 = float(np.dot(r, rot[i]) ** 2 / (np.dot(r, r) * np.dot(rot[i], rot[i]) + 1e-12))
            if c2 > best_c2:
                best_c2, best_c = c2, c
        codes[i] = best_c
    # 重建: 反旋转 + 加回均值 + 归一化
    recon = np.zeros_like(rot)
    for i in range(len(rot)):
        s = 1.0; best_c2 = -1
        for sc in cand_s:
            r = levels[np.arange(m), codes[i]] * sc
            c2 = float(np.dot(r, rot[i]) ** 2 / (np.dot(r, r) * np.dot(rot[i], rot[i]) + 1e-12))
            if c2 > best_c2:
                best_c2, best_c = c2, sc
        recon[i] = levels[np.arange(m), codes[i]] * best_c
    o = (P @ recon.T).T + mu
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    r = recall_at_k(brute_rank(Q.astype(np.float32), o.astype(np.float32)), qids, qrels, ids)[0]
    res["perm50_centered_calib_sweep"] = round(r, 4)
    print(f"{args.model}: {r:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
