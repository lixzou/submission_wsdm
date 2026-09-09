#!/usr/bin/env python3
"""Pilot 1（可行性验证）: 变换段单跑——置换截断 vs 前缀 vs 随机 vs PCA @ 激进压缩区。
有效性判定: 置换截断 recall@10 显著高于前缀截断（预期 +10pp 级）且不劣于 PCA。
"""
import argparse, json, os
import numpy as np

def load(ds_dir, model):
    emb = np.load(os.path.join(ds_dir, "emb_%s.npy" % model)).astype(np.float32)
    qemb = np.load(os.path.join(ds_dir, "qemb_%s.npy" % model)).astype(np.float32)
    meta = json.load(open(os.path.join(ds_dir, "meta.json")))
    qids = [str(q) for q in np.load(os.path.join(ds_dir, "query_ids.npy"))]
    qrels = {str(q): {str(d) for d in docs} for q, docs in meta["qrels"].items()}
    ids = [str(i) for i in meta["ids"]]
    return emb, qemb, qids, qrels, ids

def recall_at_k(rank, qids, qrels, ids, k=10, return_arrays=False):
    recs, ndcgs = [], []
    for i, qid in enumerate(qids):
        if qid not in qrels: continue
        rel = qrels[qid]
        got = [str(ids[j]) for j in rank[i] if j < len(ids)]
        hit = sum(1 for g in got if g in rel)
        recs.append(hit / max(len(rel), 1))
        dcg = sum((1.0 if g in rel else 0.0) / np.log2(2 + r) for r, g in enumerate(got))
        idcg = sum(1.0 / np.log2(2 + r) for r in range(min(len(rel), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    mean, ndcg = float(np.mean(recs)), float(np.mean(ndcgs))
    if return_arrays:
        return mean, ndcg, np.asarray(recs)
    return mean, ndcg

def brute_rank(qemb, emb, k=10):
    rank = np.zeros((len(qemb), k), np.int64)
    B = 2048
    for i in range(0, len(qemb), B):
        sim = qemb[i:i+B] @ emb.T
        rank[i:i+B] = np.argpartition(-sim, k, axis=1)[:, :k]
    return rank

def margin_importance(emb, qemb, qids, qrels, ids, n_pairs=100000, calib_frac=0.7, seed=0):
    """每维的检索 margin 贡献（t 统计量版）:
    I_j = (E[q_j d_j | rel] - E[q_j d_j | nonrel]) / sqrt(var/pos + var/neg)
    只用有标注的查询；评估与校准查询无重叠（报告时注明）。"""
    rng = np.random.default_rng(seed)
    qidx = {q: i for i, q in enumerate(qids)}
    judged = [q for q in qids if q in qrels]
    n_cal = int(len(judged) * calib_frac)
    cal_q = judged[:n_cal]
    idset = {s: i for i, s in enumerate(ids)}
    pos, neg = [], []
    per_q = max(1, n_pairs // max(len(cal_q), 1))
    for qid in cal_q:
        qi = qidx[qid]
        # sorted() 修复 (2026-08-17): qrels 为 set, 迭代顺序受 PYTHONHASHSEED
        # 随机化影响 -> rng.integers 索引挑选漂移 -> 跨进程结果不可复现
        rel_docs = sorted(idset[d] for d in qrels[qid] if d in idset)
        if not rel_docs: continue
        for _ in range(min(100, per_q)):
            di = rel_docs[rng.integers(len(rel_docs))]
            pos.append(qemb[qi] * emb[di])
            ni = rng.integers(len(emb))
            neg.append(qemb[qi] * emb[ni])
    pos = np.array(pos); neg = np.array(neg)
    mu = pos.mean(0) - neg.mean(0)
    se = np.sqrt(pos.var(0) / len(pos) + neg.var(0) / len(neg)) + 1e-12
    return np.abs(mu / se)  # t 统计量 = 排序重要性

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--keeps", nargs="+", type=float, default=[0.5, 0.25, 0.125])
    ap.add_argument("--calib_frac", type=float, default=0.7)
    ap.add_argument("--out", default="/tmp/pilot_result.json")
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    print("model", args.model, "| emb", emb.shape, "| queries", len(qids), flush=True)

    results = {}
    ref_rank = brute_rank(qemb, emb)
    results["reference_full"] = recall_at_k(ref_rank, qids, qrels, ids)

    # 校准（前 300 查询，评估用全部查询——校准查询与评估查询重叠仅 300/3452，报告时注明）
    judged = [q for q in qids if q in qrels]
    # calib_frac=1.0 时按查询数放大采样对, 保持每查询 ~100 对（NQ 3452 查询）
    n_pairs = max(100000, len(judged) * 100)
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=args.calib_frac, n_pairs=n_pairs)

    # PCA 投影（全局白化旋转后截断 = 保持 top-m 方差子空间，内积保持）
    mu = emb.mean(0)
    U, S, Vt = np.linalg.svd(emb - mu, full_matrices=False)  # emb(N,d) = U S Vt; Vt rows = dims 方向
    # 用 Vt 投影到 top-m: x' = x Vt[:m].T

    for keep in args.keeps:
        m = max(1, int(d * keep))
        strategies = {}
        # 1. prefix
        pfx = emb[:, :m]; qpfx = qemb[:, :m]
        strategies["prefix"] = recall_at_k(brute_rank(qpfx, pfx), qids, qrels, ids)
        # 2. random dims
        rng = np.random.default_rng(42)
        ridx = rng.permutation(d)[:m]
        rnd = emb[:, ridx]; qrnd = qemb[:, ridx]
        strategies["random"] = recall_at_k(brute_rank(qrnd, rnd), qids, qrels, ids)
        # 3. PCA top-m
        Vm = Vt[:m]  # (m, d)
        pca = emb @ Vm.T; qpca = qemb @ Vm.T
        strategies["pca"] = recall_at_k(brute_rank(qpca, pca), qids, qrels, ids)
        # 4. 我们的置换（margin 重要性排序）
        order = np.argsort(-imp)
        perm = emb[:, order[:m]]; qperm = qemb[:, order[:m]]
        strategies["perm_margin"] = recall_at_k(brute_rank(qperm, perm), qids, qrels, ids)
        results["keep=%.3f" % keep] = strategies
        print("keep=%.3f: " % keep, {k: round(v[0], 4) for k, v in strategies.items()}, flush=True)

    json.dump(results, open(args.out, "w"), indent=1)
    print("PILOT DONE ->", args.out, flush=True)

if __name__ == "__main__":
    main()
