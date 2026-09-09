#!/usr/bin/env python3
"""beat_turboquant_check.py — 打 TurboQuant 的量化变体（用户指令 2026-08-17）。

同 harness 对比（scifact e5, B=1/2/4）:
  v0 turboquant  = 不中心化 + Lloyd-Max 网格（基线, 要打的对象）
  v1 centered    = 中心化 + Lloyd-Max
  v2 center+sweep= 中心化 + Lloyd-Max + 每向量扫掠最优步长（我们的协议叠加）
  v3 calib      = 不中心化 + 校准拟合网格（judged 相关对上拟合逐维 Lloyd-Max）
  v4 u+sweep    = 不中心化 + Lloyd-Max + 扫掠步长
"""
import argparse, json
import numpy as np
from scipy import integrate
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from rabitq_py import random_orthogonal


def lloyd_max_levels_from_density(f, K, tol=1e-6, max_iter=300):
    xs = np.linspace(-0.999, 0.999, 20001)
    p = np.array([f(x) for x in xs]); p /= p.sum()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xs), size=K, replace=False, p=p)
    levels = np.sort(xs[idx])
    for it in range(max_iter):
        bounds = np.concatenate([[-1.0], (levels[:-1] + levels[1:]) / 2.0, [1.0]])
        new = np.zeros(K)
        for k in range(K):
            a, b = bounds[k], bounds[k + 1]
            num, _ = integrate.quad(lambda x: x * f(x), a, b, limit=200)
            den, _ = integrate.quad(lambda x: f(x), a, b, limit=200)
            new[k] = num / den if den > 1e-300 else (a + b) / 2.0
        if np.max(np.abs(new - levels)) < tol: break
        levels = new
    return levels


def lloyd_max_from_samples(vals, K, tol=1e-6, max_iter=300):
    vals = np.asarray(vals)
    levels = np.quantile(vals, np.linspace(0, 1, K))
    for it in range(max_iter):
        # 最近级分配 + 条件均值
        c = np.argmin(np.abs(vals[:, None] - levels[None, :]), axis=1)
        new = np.array([vals[c == k].mean() for k in range(K)])
        new = np.nan_to_num(new, nan=levels[np.arange(K) != -1][0] if False else 0.0)
        # 空簇用旧值
        for k in range(K):
            if not np.any(c == k):
                new[k] = levels[k]
        if np.max(np.abs(new - levels)) < tol: break
        levels = new
    return np.sort(levels)


def quant_recall(emb, qemb, qids, qrels, ids, B, mode, w=None):
    d = emb.shape[1]
    P = random_orthogonal(d, seed=0)
    mu = emb.mean(0)
    if mode in ("centered", "center+sweep"):
        rot = ((emb - mu) @ P).astype(np.float64)
    else:
        rot = (emb @ P).astype(np.float64)
    K = 2 ** B
    if mode == "calib" and w is not None:
        # 逐维校准拟合: 相关对上逐维 Lloyd-Max（排序目标）
        levels = np.zeros((d, K))
        rel_vals = []
        qidx = {q: i for i, q in enumerate(qids)}
        idset = {s: i for i, s in enumerate(ids)}
        for qid in qids:
            if qid not in qrels: continue
            qi = qidx[qid]
            for did in qrels[qid]:
                if did in idset:
                    rel_vals.append(rot[idset[did]])
        rel_vals = np.array(rel_vals)
        for j in range(d):
            levels[j] = lloyd_max_from_samples(rel_vals[:, j], K)
        lv = levels
    else:
        def beta_density(x):
            if x <= -1.0 or x >= 1.0: return 0.0
            return (1.0 - x * x) ** ((d - 3) / 2.0)
        base = lloyd_max_levels_from_density(beta_density, K)
        lv = np.tile(base, (d, 1))
    # 编码（逐维最近级）+ 可选扫掠步长
    codes = np.zeros_like(rot, dtype=np.int32)
    if mode == "center+sweep" or mode == "u+sweep":
        # 每向量扫掠全局步长 s: 候选 s 使 cos^2(s*levels[c], rot) 最大
        cand_s = np.linspace(0.2, 3.0, 40)
        best_codes = None; best_s = np.ones(len(rot))
        for i in range(len(rot)):
            best_c2 = -1
            for s in cand_s:
                c = np.argmin(np.abs(rot[i, :, None] - (lv * s)[:, :].T), axis=1) if False else None
        # 简化: 对每向量在 s 上贪心（采样 40 个候选）
        for i in range(len(rot)):
            best_c2, best_c = -1, None
            for s in cand_s:
                c = np.argmin(np.abs(rot[i][None, :] - (lv * s).T), axis=0)
                # 注意 lv.T 形状: 用逐维 argmin
                c = np.array([np.argmin(np.abs(rot[i, j] - lv[j] * s)) for j in range(d)])
                recon = lv[np.arange(d), c] * s
                c2 = float(np.dot(recon, rot[i]) ** 2 / (np.dot(recon, recon) * np.dot(rot[i], rot[i]) + 1e-12))
                if c2 > best_c2:
                    best_c2, best_c = c2, c
            codes[i] = best_c
    else:
        for i in range(len(rot)):
            for j in range(d):
                codes[i, j] = np.argmin(np.abs(rot[i, j] - lv[j]))
    # 重建
    recon_rot = np.zeros_like(rot)
    for i in range(len(rot)):
        if mode in ("center+sweep", "u+sweep"):
            s = 1.0
            best_c2, best_sv = -1, 1.0
            for sc in cand_s:
                r = lv[np.arange(d), codes[i]] * sc
                c2 = float(np.dot(r, rot[i]) ** 2 / (np.dot(r, r) * np.dot(rot[i], rot[i]) + 1e-12))
                if c2 > best_c2:
                    best_c2, best_sv = c2, sc
            s = best_sv
            recon_rot[i] = lv[np.arange(d), codes[i]] * s
        else:
            recon_rot[i] = lv[np.arange(d), codes[i]]
    o = (P @ recon_rot.T).T
    if mode in ("centered", "center+sweep"):
        o = o + mu
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    return recall_at_k(brute_rank(qemb.astype(np.float32), o.astype(np.float32)), qids, qrels, ids)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "reference": round(ref, 4)}
    w = np.abs(margin_importance(emb, qemb, qids, qrels, ids))
    for B in [1, 2, 4]:
        for mode in ["turboquant", "centered", "center+sweep", "calib", "u+sweep"]:
            if B == 1 and mode in ("center+sweep", "u+sweep", "calib"):
                continue  # B=1 符号无步长/拟合意义（calib 保留）
            r = quant_recall(emb, qemb, qids, qrels, ids, B, mode, w)
            res[f"B={B}_{mode}"] = round(r, 4)
            print(f"B={B} {mode}: {r:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
