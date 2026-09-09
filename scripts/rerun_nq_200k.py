#!/usr/bin/env python3
"""NQ-200k 诚实重跑: heldout split (无官方 train split, 从 3452 test query
随机抽 800 校准 / 2652 评估) + 同时输出 Recall@10 和 nDCG@10。

协议与 train_calib_main.py 一致: 校准查询与评估查询不相交。
用法: python rerun_nq_200k.py --model e5 --out results/runs/nq200k_e5.json
"""
import argparse, json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import faiss
faiss.omp_set_num_threads(int(os.environ.get("FAISS_THREADS", "8")))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from sota_pipeline_check import fit_calib_levels
from merge_pca_retrieval import pca_directions
from rabitq_py import random_orthogonal
from turboquant_py import lloyd_max

DS = "data/nq_sub200k"
N_CAL = 800
SEED = 0
KEEPS = [0.5, 0.75, 0.875]
B = 2  # 统一位宽, 与主表协议一致


def norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def metrics(rank, qids, qrels, ids):
    """返回 (Recall@10, nDCG@10)."""
    r, n = recall_at_k(rank, qids, qrels, ids, k=10)
    return float(r), float(n)


def run_reco(emb, qemb_test, imp, test_qids, test_qrels, ids, m, calib_qids, calib_qrels):
    """spectral 加权 PCA 截断 + 校准 Lloyd-Max 量化, 返回 (recall, ndcg)."""
    V = pca_directions(emb, m, w=(imp / (imp.max() + 1e-6)))
    emb_proj = emb @ V.T
    q_proj = qemb_test @ V.T
    P = random_orthogonal(m, seed=0)
    rot = (emb_proj @ P).astype(np.float64)
    levels = fit_calib_levels(rot, calib_qids, calib_qrels, ids, B)
    codes = np.zeros((len(rot), m), np.int32)
    for j in range(m):
        codes[:, j] = np.argmin(np.abs(rot[:, j, None] - levels[j][None, :]), axis=1)
    recon = levels[np.arange(m)[None, :], codes]
    o = (P @ recon.T).T
    o = norm_rows(o.astype(np.float32))
    return metrics(brute_rank(q_proj.astype(np.float32), o.astype(np.float32)),
                   test_qids, test_qrels, ids)


def run_base(kind, emb, qemb_test, test_qids, test_qrels, ids, m):
    """prefix 截断到 m 维 + 各量化器, 返回 (recall, ndcg) 或 None."""
    emb_t = emb[:, :m].copy()
    qemb_t = qemb_test[:, :m].copy()
    d_m = m
    if kind == "pq":
        M = d_m * B // 8
        if M < 1 or M > d_m:
            return None
        q = faiss.IndexPQ(d_m, M, 8); q.train(emb_t)
        o = q.sa_decode(q.sa_encode(emb_t)).astype(np.float32)
        return metrics(brute_rank(qemb_t, norm_rows(o)), test_qids, test_qrels, ids)
    elif kind == "opq":
        M = d_m * B // 8
        if M < 1 or M > d_m:
            return None
        opq = faiss.OPQMatrix(d_m, M); opq.train(emb_t)
        x = opq.apply(emb_t); qx = opq.apply(qemb_t)
        pq = faiss.IndexPQ(d_m, M, 8); pq.train(x)
        o = pq.sa_decode(pq.sa_encode(x)).astype(np.float32)
        return metrics(brute_rank(qx, norm_rows(o)), test_qids, test_qrels, ids)
    elif kind == "itq":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=d_m, whiten=True)
        x = pca.fit_transform(emb_t)
        R = random_orthogonal(d_m, seed=0).astype(np.float64)
        xr = x @ R
        b = (xr > 0).astype(np.float32) * 2 - 1
        qb = (pca.transform(qemb_t) @ R > 0).astype(np.float32) * 2 - 1
        return metrics(brute_rank(qb, b), test_qids, test_qrels, ids)
    elif kind == "scalar":
        mi = emb_t.min(0); ma = emb_t.max(0)
        levels = 2 ** B
        codes = np.clip(((emb_t - mi) / (ma - mi + 1e-12) * (levels - 1)), 0, levels - 1).astype(np.int32)
        recon = codes / (levels - 1) * (ma - mi) + mi
        return metrics(brute_rank(qemb_t, norm_rows(recon.astype(np.float32))),
                       test_qids, test_qrels, ids)
    elif kind == "tq":
        K = 2 ** B
        P = random_orthogonal(d_m, seed=0).astype(np.float64)
        rot = emb_t @ P
        levels = np.array([lloyd_max(d_m, K) for _ in range(d_m)])
        codes = np.argmin(np.abs(rot[:, :, None] - levels[None, :, :]), axis=2)
        rec = levels[np.arange(d_m)[None, :], codes]
        o = norm_rows((P @ rec.T).T.astype(np.float32))
        qc = np.argmin(np.abs((qemb_t @ P)[:, :, None] - levels[None, :, :]), axis=2)
        qo = norm_rows((P @ levels[np.arange(d_m)[None, :], qc].T).T.astype(np.float32))
        return metrics(brute_rank(qo, o), test_qids, test_qrels, ids)
    else:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds", default=DS)
    ap.add_argument("--n_cal", type=int, default=N_CAL)
    ap.add_argument("--keeps", type=str, default="0.5,0.75,0.875",
                    help="comma-separated retained fractions")
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds, args.model)
    d = emb.shape[1]
    # judged query 全集 (均有 qrels)
    judged = [q for q in qids if q in qrels]
    assert len(judged) == len(qids), f"有 {len(qids)-len(judged)} 条 query 无 qrels"
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(qids))
    train_idx = sorted(perm[:args.n_cal].tolist())
    test_idx = sorted(perm[args.n_cal:].tolist())
    train_qids = [qids[i] for i in train_idx]
    test_qids = [qids[i] for i in test_idx]
    train_qrels = {q: qrels[q] for q in train_qids}
    test_qrels = {q: qrels[q] for q in test_qids}
    qemb_train = qemb[np.array(train_idx)]
    qemb_test = qemb[np.array(test_idx)]

    imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)

    ref = metrics(brute_rank(qemb_test, emb), test_qids, test_qrels, ids)
    res = {"ds_dir": args.ds, "model": args.model, "d": d,
           "n_train": len(train_qids), "n_test": len(test_qids),
           "reference_recall": round(ref[0], 4), "reference_ndcg": round(ref[1], 4),
           "seed": SEED, "B": B}
    print(f"{args.model}: train {len(train_qids)} / test {len(test_qids)}, "
          f"ref recall={ref[0]:.4f} ndcg={ref[1]:.4f}", flush=True)
    keeps = [float(x) for x in args.keeps.split(",")]
    for keep in keeps:
        m = int(d * keep)
        if m % 8 != 0:
            m = (m // 8) * 8
        cell = {}
        for meth in ["pq", "opq", "itq", "scalar", "tq"]:
            try:
                r = run_base(meth, emb, qemb_test, test_qids, test_qrels, ids, m)
                if r is not None:
                    cell[meth] = {"recall": round(r[0], 4), "ndcg": round(r[1], 4)}
            except Exception as e:
                print(f"  {meth} ERR {e}", flush=True)
        try:
            r = run_reco(emb, qemb_test, imp, test_qids, test_qrels, ids, m,
                         train_qids, train_qrels)
            cell["reco"] = {"recall": round(r[0], 4), "ndcg": round(r[1], 4)}
        except Exception as e:
            print(f"  reco ERR {e}", flush=True)
        res["keep=%.3f" % keep] = cell
        print(f"  keep={keep} m={m} " +
              " ".join(f"{k}={v['recall']:.4f}/{v['ndcg']:.4f}" for k, v in cell.items()),
              flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
