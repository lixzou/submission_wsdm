#!/usr/bin/env python3
"""query_selection_check.py — 校准查询选择策略（用户指令 2026-08-18）。

现状: margin_importance 用 judged[:n_cal]（按 qids 顺序取前 n 条, 任意!）。
对比（25% = 75 条 / 50% = 150 条查询下）:
  seq     顺序前 n（现状）
  rand    随机 n（5 seed 平均）
  div     k-means 多样性选择（查询嵌入聚类, 每簇按簇大小取代表）
  hard    难查询优先（该查询自身在全集上的 recall@10 低 → 校准信号强）
"""
import argparse, json
import numpy as np
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance


def select_queries(emb, qemb, qids, qrels, ids, n, mode, seed=0):
    qidx = {q: i for i, q in enumerate(qids)}
    judged = [q for q in qids if q in qrels]
    if mode == "seq":
        return judged[:n]
    if mode == "rand":
        rng = np.random.default_rng(seed)
        return list(rng.choice(judged, size=n, replace=False))
    if mode == "div":
        # k-means 聚类查询嵌入（k=n）, 每簇取离簇心最近的一条
        from sklearn.cluster import KMeans
        Q = np.vstack([qemb[qidx[q]] for q in judged])
        km = KMeans(n_clusters=n, n_init=5, random_state=seed).fit(Q)
        sel = []
        for c in range(n):
            idxs = np.where(km.labels_ == c)[0]
            if len(idxs) == 0: continue
            d = np.linalg.norm(Q[idxs] - km.cluster_centers_[c], axis=1)
            sel.append(judged[idxs[int(np.argmin(d))]])
        # 不足 n 时补随机
        rng = np.random.default_rng(seed)
        rest = [q for q in judged if q not in sel]
        sel += list(rng.choice(rest, size=n - len(sel), replace=False))
        return sel
    if mode == "hard":
        # 该查询在全集上的 recall@10 低 → 校准信号强（难查询）
        qmask = np.array([q in set(judged) for q in qids])
        qi = [qidx[q] for q in judged]
        sims = qemb[qi] @ emb.T
        ranks = np.argsort(-sims, axis=1)[:, :10]
        rel_set = [set(qrels[q]) & set(ids) for q in judged]
        scores = []
        for k, q in enumerate(judged):
            got = {ids[j] for j in ranks[k] if j < len(ids)}
            scores.append(len(got & rel_set[k]) / max(len(rel_set[k]), 1))
        order = np.argsort(scores)  # 低 recall 优先
        return [judged[i] for i in order[:n]]


def margin_with_calib(emb, qemb, qids, qrels, ids, cal_q, seed=0, pairs_per_q=100):
    rng = np.random.default_rng(seed)
    qidx = {q: i for i, q in enumerate(qids)}
    idset = {s: i for i, s in enumerate(ids)}
    pos, neg = [], []
    for qid in cal_q:
        qi = qidx[qid]
        rel_docs = sorted(idset[d] for d in qrels[qid] if d in idset)
        if not rel_docs: continue
        for _ in range(pairs_per_q):
            di = rel_docs[rng.integers(len(rel_docs))]
            ni = rng.integers(len(emb))
            pos.append(qemb[qi] * emb[di])
            neg.append(qemb[qi] * emb[ni])
    pos, neg = np.array(pos), np.array(neg)
    mu = pos.mean(0) - neg.mean(0)
    se = np.sqrt(pos.var(0)/len(pos) + neg.var(0)/len(neg)) + 1e-12
    return np.abs(mu / se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.25)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    keep = 0.125
    m = max(1, int(d * keep))
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "reference": round(ref, 4),
           "frac": args.frac, "keep": keep}
    judged = [q for q in qids if q in qrels]
    n = max(1, int(len(judged) * args.frac))
    for mode in ["seq", "rand", "div", "hard"]:
        vals = []
        for seed in [0, 1, 2] if mode == "rand" else [0]:
            cal_q = select_queries(emb, qemb, qids, qrels, ids, n, mode, seed)
            w = margin_with_calib(emb, qemb, qids, qrels, ids, cal_q)
            order = np.argsort(-w)[:m]
            r = recall_at_k(brute_rank(qemb[:, order], emb[:, order]), qids, qrels, ids)[0]
            vals.append(r)
            print(f"{mode} seed={seed}: {r:.4f}", flush=True)
        res[mode] = round(float(np.mean(vals)), 4)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
