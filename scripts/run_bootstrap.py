#!/usr/bin/env python3
"""C1 显著性检验：置换 vs 前缀 的 paired bootstrap（Q34）。
协议与 pilot_truncation.py 一致：margin 重要性用 70% 校准查询，评估全量 judged 查询。
用法: python run_bootstrap.py --ds_dir data/beir/scifact --model e5 --keep 0.125 --n_boot 1000 --out bootstrap_scifact_e5.json
"""
import argparse, json, os, numpy as np

def load(ds_dir, model):
    emb = np.load(os.path.join(ds_dir, "emb_%s.npy" % model)).astype(np.float32)
    qemb = np.load(os.path.join(ds_dir, "qemb_%s.npy" % model)).astype(np.float32)
    meta = json.load(open(os.path.join(ds_dir, "meta.json")))
    qids = [str(q) for q in np.load(os.path.join(ds_dir, "query_ids.npy"))]
    qrels = {str(q): {str(d) for d in docs} for q, docs in meta["qrels"].items()}
    ids = [str(i) for i in meta["ids"]]
    return emb, qemb, qids, qrels, ids

def margin_importance(emb, qemb, qids, qrels, ids, calib_frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    qidx = {q: i for i, q in enumerate(qids)}
    judged = [q for q in qids if q in qrels]
    n_cal = int(len(judged) * calib_frac)
    cal_q = judged[:n_cal]
    idset = {s: i for i, s in enumerate(ids)}
    pos, neg = [], []
    for qid in cal_q:
        qi = qidx[qid]
        # sorted() 修复 (2026-08-17): 同 pilot_truncation.py — set 迭代顺序
        # 受 PYTHONHASHSEED 影响, 跨进程不可复现
        rel_docs = sorted(idset[d] for d in qrels[qid] if d in idset)
        if not rel_docs: continue
        for _ in range(100):
            di = rel_docs[rng.integers(len(rel_docs))]
            pos.append(qemb[qi] * emb[di])
            ni = rng.integers(len(emb))
            neg.append(qemb[qi] * emb[ni])
    pos = np.array(pos); neg = np.array(neg)
    mu = pos.mean(0) - neg.mean(0)
    se = np.sqrt(pos.var(0)/len(pos) + neg.var(0)/len(neg)) + 1e-12
    return np.abs(mu / se)

def per_query_recall(sim, qids, qrels, ids, k=10):
    """返回 judged 查询的 per-query recall 数组（与 qrels 顺序无关，按 judged 查询列表）"""
    judged = [q for q in qids if q in qrels]
    qidx = {q: i for i, q in enumerate(qids)}
    # 每个 judged 查询的 top-k
    out = np.zeros(len(judged))
    for t, qid in enumerate(judged):
        i = qidx[qid]
        top = np.argpartition(-sim[i], k)[:k]
        got = [str(ids[j]) for j in top]
        rel = qrels[qid]
        out[t] = sum(1 for g in got if g in rel) / max(len(rel), 1)
    return judged, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--keep", type=float, default=0.125)
    ap.add_argument("--calib_frac", type=float, default=0.7)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/bootstrap.json")
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]; kd = max(1, int(d * args.keep))
    print("model", args.model, "| emb", emb.shape, "| keep dims", kd, flush=True)

    # 置换顺序（校准 70%）
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=args.calib_frac)
    order = np.argsort(-imp)
    perm_emb = emb[:, order[:kd]]
    prefix_emb = emb[:, :kd]
    perm_q = qemb[:, order[:kd]]
    prefix_q = qemb[:, :kd]

    # per-query recall（judged 查询）
    sim_perm = perm_q @ perm_emb.T
    sim_prefix = prefix_q @ prefix_emb.T
    judged, r_perm = per_query_recall(sim_perm, qids, qrels, ids)
    _, r_prefix = per_query_recall(sim_prefix, qids, qrels, ids)

    base_perm = float(r_perm.mean()); base_prefix = float(r_prefix.mean())
    print("perm recall@10 =", round(base_perm, 4), "| prefix =", round(base_prefix, 4), "| diff =", round(base_perm - base_prefix, 4), flush=True)

    # paired bootstrap
    rng = np.random.default_rng(args.seed)
    n = len(r_perm)
    diffs = np.zeros(args.n_boot)
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = r_perm[idx].mean() - r_prefix[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float(np.mean(diffs <= 0))  # 单侧: perm <= prefix 的比例
    res = {
        "keep": args.keep, "model": args.model,
        "perm_recall": base_perm, "prefix_recall": base_prefix, "diff": base_perm - base_prefix,
        "boot_mean_diff": float(diffs.mean()),
        "ci95": [float(lo), float(hi)],
        "p_value_perm_le_prefix": p,
        "n_boot": args.n_boot, "n_queries": n
    }
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
