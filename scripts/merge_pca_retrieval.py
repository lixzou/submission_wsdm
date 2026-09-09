#!/usr/bin/env python3
"""Merge retrieval-aware (ranking-importance) with PCA.

用户想法: 先把相关的信息按排序重要性找出来, 再做 PCA 加权的 PCA。
本脚本实现几种合并变体并在 scifact 上对比:
  - prefix / perm / pca           (baselines, 对齐 tab:c1)
  - imp_weighted_pca              (importance 加权协方差的 PCA: 主方向偏向检索重要维度)
  - imp_sel_pca                   (先按 importance 选 top-K 重要维度, 再在该子空间 PCA 截断到 m)
  - imp_sel_pca_perm              (imp_sel_pca 后再按重要性重排剩余维? 即选子空间内再置换)

协议: brute-force 余弦/内积, Recall@10, 与 pilot_truncation.py 一致 (uncentered 投影)。
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

def recall_at_k(rank, qids, qrels, ids, k=10):
    recs = []
    for i, qid in enumerate(qids):
        if qid not in qrels: continue
        rel = qrels[qid]
        got = [str(ids[j]) for j in rank[i] if j < len(ids)]
        recs.append(sum(1 for g in got if g in rel) / max(len(rel), 1))
    return float(np.mean(recs))

def brute_rank(qemb, emb, k=10):
    rank = np.zeros((len(qemb), k), np.int64)
    B = 2048
    for i in range(0, len(qemb), B):
        sim = qemb[i:i+B] @ emb.T
        rank[i:i+B] = np.argpartition(-sim, k, axis=1)[:, :k]
    return rank

def margin_importance(emb, qemb, qids, qrels, ids, n_pairs=100000, calib_frac=0.7, seed=0):
    """每维检索 margin 贡献 (Welch t-statistic), 与 pilot_truncation.py 一致."""
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
    return np.abs(mu / se)

def pca_directions(emb, m, w=None):
    """PCA 主方向. w 为 None 则普通 PCA; 否则 importance 加权协方差.
    返回 V (m, d) 投影矩阵, 使用 uncentered 投影协议 (与 pilot_truncation 一致).
    """
    mu = emb.mean(0)
    X = emb - mu
    if w is not None:
        # 加权: 每维乘 sqrt(w) 再做 SVD => 等价于加权协方差 E[w_j w_l x_j x_l]
        Xw = X * np.sqrt(w)[None, :]
        _, _, Vt = np.linalg.svd(Xw, full_matrices=False)
    else:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:m]  # (m, d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--keeps", nargs="+", type=float, default=[0.25, 0.125])
    ap.add_argument("--calib_frac", type=float, default=0.7)
    ap.add_argument("--n_pairs", type=int, default=0, help="0=auto max(100000, judged*100)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    print("model", args.model, "| emb", emb.shape, "| queries", len(qids), flush=True)

    results = {}
    results["reference_full"] = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)

    judged = [q for q in qids if q in qrels]
    n_pairs = args.n_pairs if args.n_pairs > 0 else max(100000, len(judged) * 100)
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=args.calib_frac, n_pairs=n_pairs)
    w = imp / (imp.max() + 1e-12) + 1e-6   # 归一化重要性权重

    for keep in args.keeps:
        m = max(1, int(d * keep))
        strategies = {}
        # --- baselines ---
        strategies["prefix"] = recall_at_k(brute_rank(qemb[:, :m], emb[:, :m]), qids, qrels, ids)
        order = np.argsort(-imp)
        strategies["perm"] = recall_at_k(brute_rank(qemb[:, order[:m]], emb[:, order[:m]]), qids, qrels, ids)
        V = pca_directions(emb, m)
        strategies["pca"] = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)

        # --- 合并方法 1: importance 加权 PCA (全局加权协方差主方向) ---
        for wp in [0.5, 1.0, 2.0]:
            ww = imp ** wp
            ww = ww / (ww.max() + 1e-12) + 1e-6
            Vw = pca_directions(emb, m, w=ww)
            strategies[f"impw_pca_p{wp}"] = recall_at_k(brute_rank(qemb @ Vw.T, emb @ Vw.T), qids, qrels, ids)

        # --- 合并方法 2: 先按重要性选 top-K 维度, 再在子空间内 PCA 截断到 m ---
        for K_mult in [2, 4, 8]:
            K = min(d, max(m, int(m * K_mult)))
            sel = np.argsort(-imp)[:K]
            emb_sel = emb[:, sel]; qemb_sel = qemb[:, sel]
            Vsub = pca_directions(emb_sel, m)  # 子空间内 PCA 到 m 维
            strategies[f"imp_sel_pca_K{K_mult}x"] = recall_at_k(
                brute_rank(qemb_sel @ Vsub.T, emb_sel @ Vsub.T), qids, qrels, ids)

        # --- 合并方法 3: 选 top-K 重要维度 + importance 加权 PCA (子空间内加权) ---
        for K_mult in [2, 4]:
            K = min(d, max(m, int(m * K_mult)))
            sel = np.argsort(-imp)[:K]
            emb_sel = emb[:, sel]; qemb_sel = qemb[:, sel]
            w_sel = w[sel]
            Vsw = pca_directions(emb_sel, m, w=w_sel)
            strategies[f"imp_sel_wpca_K{K_mult}x"] = recall_at_k(
                brute_rank(qemb_sel @ Vsw.T, emb_sel @ Vsw.T), qids, qrels, ids)

        results["keep=%.3f" % keep] = strategies
        print("keep=%.3f: " % keep, {k: round(v, 4) for k, v in strategies.items()}, flush=True)

    json.dump(results, open(args.out, "w"), indent=1)
    print("DONE ->", args.out, flush=True)

if __name__ == "__main__":
    main()
