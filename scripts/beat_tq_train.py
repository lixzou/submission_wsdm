#!/usr/bin/env python3
"""beat_tq_train.py — 诚实 train 协议下重跑 beat_tq。

对比 (全维, B=1/2/4):
  - turboquant   : 随机旋转 + 重建最优 Beta 网格 (不中心化)      [原版协议]
  - centered     : 随机旋转 + 中心化 + Beta 网格                  [无需 qrels, 协议无关]
  - calib        : 随机旋转 + 校准网格(拟合 judged 相关文档)      [★ 改: 拟合 TRAIN qrels, 评估 test]
原 beat_tq_v2.py 的 calib 网格拟合在 test qrels 上 = 泄漏; 本脚本用 train_qrels + train 查询嵌入。
"""
import argparse, json
import numpy as np
from scipy import integrate
from pilot_truncation import load, recall_at_k, brute_rank
from rabitq_py import random_orthogonal


def beta_density(d):
    def f(x):
        if x <= -1.0 or x >= 1.0:
            return 0.0
        return (1.0 - x * x) ** ((d - 3) / 2.0)
    return f


def lloyd_max_from_density(f, K, tol=1e-6, max_iter=300):
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
        c = np.argmin(np.abs(vals[:, None] - levels[None, :]), axis=1)
        new = np.array([vals[c == k].mean() if np.any(c == k) else levels[k] for k in range(K)])
        if np.max(np.abs(new - levels)) < tol: break
        levels = new
    return np.sort(levels)


def encode_vectorized(rot, levels):
    """codes[i,j] = argmin_k |rot[i,j] - levels[j,k]|  (levels: (d,K))"""
    N, d = rot.shape
    K = levels.shape[1]
    codes = np.zeros((N, d), np.int32)
    BATCH = 256
    for i0 in range(0, N, BATCH):
        x = rot[i0:i0 + BATCH, :, None]      # (b,d,1)
        lv = levels[None, :, :]              # (1,d,K)
        codes[i0:i0 + BATCH] = np.argmin(np.abs(x - lv), axis=2)
    return codes


def recall_of(qemb, o, qids, qrels, ids):
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    return recall_at_k(brute_rank(qemb.astype(np.float32), o.astype(np.float32)), qids, qrels, ids)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    # train 校准数据
    tq = json.load(open(f'{args.ds_dir}/train_qids.json'))
    tr = json.load(open(f'{args.ds_dir}/train_qrels.json'))
    res = {"ds_dir": args.ds_dir, "model": args.model, "reference": round(ref, 4),
           "calib": "train", "n_cal": len(tq)}
    P = random_orthogonal(d, seed=0)
    mu = emb.mean(0)
    rot_u = (emb @ P).astype(np.float64)
    rot_c = ((emb - mu) @ P).astype(np.float64)
    for B in [1, 2, 4]:
        K = 2 ** B
        f = beta_density(d)
        lv_u = np.tile(lloyd_max_from_density(f, K), (d, 1))
        lv_c = np.tile(lloyd_max_from_density(f, K), (d, 1))
        # turboquant（不中心化）
        codes = encode_vectorized(rot_u, lv_u)
        recon = lv_u[np.arange(d)[None, :], codes]
        o = (P @ recon.T).T
        r = recall_of(qemb, o, qids, qrels, ids)
        res[f"B={B}_turboquant"] = round(r, 4)
        print(f"B={B} turboquant: {r:.4f}", flush=True)
        # centered + Lloyd-Max
        codes = encode_vectorized(rot_c, lv_c)
        recon = lv_c[np.arange(d)[None, :], codes]
        o = (P @ recon.T).T + mu
        r = recall_of(qemb, o, qids, qrels, ids)
        res[f"B={B}_centered"] = round(r, 4)
        print(f"B={B} centered: {r:.4f}", flush=True)
        # calib 网格: 拟合 TRAIN 相关文档（诚实）, 评估 test
        idset = {s: i for i, s in enumerate(ids)}
        rel_vals = []
        for qid in tq:
            if qid not in tr: continue
            for did in tr[qid]:
                if did in idset:
                    rel_vals.append(rot_u[idset[did]])
        rel_vals = np.array(rel_vals)
        lv_cl = np.zeros((d, K))
        for j in range(d):
            lv_cl[j] = lloyd_max_from_samples(rel_vals[:, j], K)
        codes = encode_vectorized(rot_u, lv_cl)
        recon = lv_cl[np.arange(d)[None, :], codes]
        o = (P @ recon.T).T
        r = recall_of(qemb, o, qids, qrels, ids)
        res[f"B={B}_calib"] = round(r, 4)
        print(f"B={B} calib: {r:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
