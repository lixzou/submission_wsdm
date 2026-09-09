#!/usr/bin/env python3
"""calib_quality_check.py — 校准数据质量诊断（用户直觉: 数据有问题导致效果不好）。

三轴: ①校准查询数 10%~100% ②每查询采样对 100~1000 ③负例: 均匀 vs 难负例(top-K 近邻)。
输出: keep=12.5% 时各配置的 recall, 供"怎么选校准数据"guideline 的证据。
"""
import argparse, json
import numpy as np
from pilot_truncation import load, recall_at_k, brute_rank


def margin_importance_cfg(emb, qemb, qids, qrels, ids, calib_frac=0.7, pairs_per_q=100,
                          hard_negs=False, hard_K=50, seed=0):
    rng = np.random.default_rng(seed)
    qidx = {q: i for i, q in enumerate(qids)}
    judged = [q for q in qids if q in qrels]
    n_cal = int(len(judged) * calib_frac)
    cal_q = judged[:n_cal]
    idset = {s: i for i, s in enumerate(ids)}
    pos, neg = [], []
    # 难负例: 全局前 hard_K 近邻索引（对每个 cal 查询的文档逐查询算太慢, 用查询级近似:
    # 取该查询的 top-hard_K 相似文档作为负例候选池）
    for qid in cal_q:
        qi = qidx[qid]
        rel_docs = sorted(idset[d] for d in qrels[qid] if d in idset)
        if not rel_docs:
            continue
        if hard_negs:
            sims = qemb[qi] @ emb.T
            order = np.argsort(-sims)
            neg_pool = [j for j in order[:hard_K + len(rel_docs)] if j not in rel_docs][:hard_K]
            for _ in range(pairs_per_q):
                di = rel_docs[rng.integers(len(rel_docs))]
                ni = neg_pool[rng.integers(len(neg_pool))]
                pos.append(qemb[qi] * emb[di])
                neg.append(qemb[qi] * emb[ni])
        else:
            for _ in range(pairs_per_q):
                di = rel_docs[rng.integers(len(rel_docs))]
                ni = rng.integers(len(emb))
                pos.append(qemb[qi] * emb[di])
                neg.append(qemb[qi] * emb[ni])
    pos = np.array(pos)
    neg = np.array(neg)
    mu = pos.mean(0) - neg.mean(0)
    se = np.sqrt(pos.var(0) / len(pos) + neg.var(0) / len(neg)) + 1e-12
    return np.abs(mu / se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "reference": round(ref, 4)}
    keep = 0.125
    m = max(1, int(d * keep))
    for frac in [0.1, 0.25, 0.5, 0.7, 1.0]:
        w = margin_importance_cfg(emb, qemb, qids, qrels, ids, calib_frac=frac)
        r = recall_at_k(brute_rank(qemb[:, np.argsort(-w)[:m]], emb[:, np.argsort(-w)[:m]]),
                        qids, qrels, ids)[0]
        res[f"frac{frac}"] = round(r, 4)
        print(f"calib_frac={frac}: {r:.4f}", flush=True)
    for ppq in [300, 1000]:
        w = margin_importance_cfg(emb, qemb, qids, qrels, ids, calib_frac=0.7, pairs_per_q=ppq)
        r = recall_at_k(brute_rank(qemb[:, np.argsort(-w)[:m]], emb[:, np.argsort(-w)[:m]]),
                        qids, qrels, ids)[0]
        res[f"pairs{ppq}"] = round(r, 4)
        print(f"pairs_per_q={ppq}: {r:.4f}", flush=True)
    for hk in [20, 100]:
        w = margin_importance_cfg(emb, qemb, qids, qrels, ids, calib_frac=0.7, hard_negs=True, hard_K=hk)
        r = recall_at_k(brute_rank(qemb[:, np.argsort(-w)[:m]], emb[:, np.argsort(-w)[:m]]),
                        qids, qrels, ids)[0]
        res[f"hardneg{hk}"] = round(r, 4)
        print(f"hard_negs K={hk}: {r:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
