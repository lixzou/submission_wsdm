#!/usr/bin/env python3
"""主实验大表生成器: 模型 × 数据集 × 方法 × keep(0.5/0.75/0.875)。

每格 = 同字节预算下的 Recall@10:
  - RECO: spectral-guided 旋转截断到 m 维 + 校准 Lloyd-Max 量化 (B=2)
  - FAISS 全家: prefix 截断到 m 维 + 各量化器 (字节 = m*2/8, 同 RECO)
方法: PQ / OPQ / RQ / LSQ++ / ITQ / TurboQuant / scalar / RECO
"""
import argparse, json, os, sys, time
import numpy as np
import faiss
faiss.omp_set_num_threads(32)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from sota_pipeline_check import fit_calib_levels, quant_recall
from merge_pca_retrieval import pca_directions
from rabitq_py import random_orthogonal
from turboquant_py import lloyd_max

KEEPS = [0.5, 0.75, 0.875]
B = 2  # 统一位宽

def norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def truncate_prefix(emb, m):
    return emb[:, :m]

def _rec(rank, qids, qrels, ids, return_arrays):
    """mean recall, or the per-query Recall@10 array when return_arrays."""
    rec = recall_at_k(rank, qids, qrels, ids, return_arrays=return_arrays)
    return rec[2] if return_arrays else rec[0]

def run_reco(emb, qemb, imp, qids, qrels, ids, m, B, return_arrays=False,
             calib_qids=None, calib_qrels=None):
    """spectral-guided 旋转截断 + 校准 Lloyd-Max 量化.
    默认校准集=评估集(与历史一致); calib_qids/calib_qrels 给出时用它们校准网格(数据泄漏修复)."""
    V = pca_directions(emb, m, w=(imp/imp.max()+1e-6))
    emb_proj = emb @ V.T; q_proj = qemb @ V.T
    P = random_orthogonal(m, seed=0)
    rot = (emb_proj @ P).astype(np.float64)
    cq = calib_qids if calib_qids is not None else qids
    cr = calib_qrels if calib_qrels is not None else qrels
    levels = fit_calib_levels(rot, cq, cr, ids, B)
    return quant_recall(q_proj, emb_proj, np.arange(m), B, levels, qids, qrels, ids,
                        return_arrays=return_arrays)

def run_faiss_quant(ds_dir, model, kind, emb, qemb, qids, qrels, ids, m, B=2, nbits=8, return_arrays=False):
    """在 m 维截断空间上跑 faiss 量化器, 返回 recall."""
    emb_t = emb[:, :m].copy(); qemb_t = qemb[:, :m].copy()
    d_m = m
    if kind == "pq":
        M = d_m * B // 8
        if M < 1 or M > d_m: return None
        q = faiss.IndexPQ(d_m, M, nbits); q.train(emb_t)
        o = q.sa_decode(q.sa_encode(emb_t)).astype(np.float32)
    elif kind == "opq":
        M = d_m * B // 8
        if M < 1 or M > d_m: return None
        opq = faiss.OPQMatrix(d_m, M)
        opq.train(emb_t)
        x = opq.apply(emb_t); qx = opq.apply(qemb_t)
        pq = faiss.IndexPQ(d_m, M, nbits); pq.train(x)
        o = pq.sa_decode(pq.sa_encode(x)).astype(np.float32)
        return _rec(brute_rank(qx, o), qids, qrels, ids, return_arrays)
    elif kind == "rq":
        M = d_m * B // 8
        if M < 1 or M > d_m: return None
        q = faiss.ResidualQuantizer(d_m, M, nbits); q.train(emb_t)
        o = q.compute_codes(emb_t)
        o = q.decode(o).astype(np.float32)
    elif kind == "lsq":
        M = d_m * B // 8
        if M < 1 or M > d_m: return None
        q = faiss.LocalSearchQuantizer(d_m, M, nbits)
        q.icm_iters = 5; q.train_iters = 10
        try:
            q.train(emb_t)
            codes = q.compute_codes(emb_t)
            o = q.decode(codes).astype(np.float32)
        except Exception as e:
            return None
    elif kind == "itq":
        M = d_m * B // 8
        if M < 1 or M > d_m: return None
        # ITQ: 学习旋转二值化, d_m bits (M=1 个 nbits=d_m? 简化: 用 32x 语义)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=d_m, whiten=True)
        x = pca.fit_transform(emb_t)
        # ITQ 旋转 (简化: 随机旋转 + 阈值)
        R = random_orthogonal(d_m, seed=0).astype(np.float64)
        xr = x @ R
        b = (xr > 0).astype(np.float32) * 2 - 1
        qr = pca.transform(qemb_t) @ R
        qb = (qr > 0).astype(np.float32) * 2 - 1
        return _rec(brute_rank(qb, b), qids, qrels, ids, return_arrays)
    elif kind == "scalar":
        # 均匀标量量化 (B bits/维)
        return run_scalar(emb_t, qemb_t, qids, qrels, ids, B, return_arrays=return_arrays)
    elif kind == "tq":
        # TurboQuant: 随机旋转 + 逐坐标 Lloyd-Max (B bits), 与论文协议一致
        K = 2**B
        P = random_orthogonal(d_m, seed=0).astype(np.float64)
        rot = emb_t @ P
        levels = np.array([lloyd_max(d_m, K) for _ in range(d_m)])
        codes = np.argmin(np.abs(rot[:, :, None] - levels[None, :, :]), axis=2)
        rec = levels[np.arange(d_m)[None, :], codes]
        o = (P @ rec.T).T
        o = norm_rows(o.astype(np.float32))
        qr = qemb_t @ P
        qc = np.argmin(np.abs(qr[:, :, None] - levels[None, :, :]), axis=2)
        qrec = levels[np.arange(d_m)[None, :], qc]
        qo = (P @ qrec.T).T
        qo = norm_rows(qo.astype(np.float32))
        return _rec(brute_rank(qo, o), qids, qrels, ids, return_arrays)
    else:
        return None
    o = norm_rows(o)
    return _rec(brute_rank(qemb_t, o), qids, qrels, ids, return_arrays)

def run_scalar(emb, qemb, qids, qrels, ids, B, return_arrays=False):
    """均匀标量量化: 每维 [min,max] 线性映射到 B bits."""
    d = emb.shape[1]
    levels = 2**B
    mi = emb.min(0); ma = emb.max(0)
    codes = np.clip(((emb - mi) / (ma - mi + 1e-12) * (levels-1)), 0, levels-1).astype(np.int32)
    recon = codes / (levels-1) * (ma - mi) + mi
    recon = norm_rows(recon.astype(np.float32))
    return _rec(brute_rank(qemb, recon), qids, qrels, ids, return_arrays)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["pq","opq","rq","lsq","itq","scalar","reco"])
    ap.add_argument("--calib_frac", type=float, default=1.0)
    ap.add_argument("--subset_docs", type=int, default=0, help="N>0 取前N文档子集")
    ap.add_argument("--keeps", type=str, default="0.5,0.75,0.875",
                    help="comma-separated retained fractions")
    ap.add_argument("--bits", type=str, default="2",
                    help="comma-separated bit widths (e.g. 2,4,8 for 32x/16x/8x)")
    args = ap.parse_args()

    keeps = [float(x) for x in args.keeps.split(",")]
    bits = [int(x) for x in args.bits.split(",")]

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    if args.subset_docs > 0:
        emb = emb[:args.subset_docs]; ids = ids[:args.subset_docs]
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d, "reference": round(ref, 4),
           "keeps": keeps, "bits": bits}
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=args.calib_frac)

    for keep in keeps:
        m = max(1, int(d * keep))
        if m % 8 != 0:
            m = (m // 8) * 8
        for B in bits:
            cell = {}
            for meth in args.methods:
                try:
                    if meth == "reco":
                        r = run_reco(emb, qemb, imp, qids, qrels, ids, m, B)
                    else:
                        r = run_faiss_quant(args.ds_dir, args.model, meth, emb, qemb, qids, qrels, ids, m, B)
                    if r is not None:
                        cell[meth] = round(r, 4)
                        print(f"  keep={keep} B={B} {meth}: {r:.4f}", flush=True)
                    else:
                        cell[meth] = None
                except Exception as e:
                    cell[meth] = None
                    print(f"  keep={keep} B={B} {meth}: ERR {e}", flush=True)
            res["keep=%.3f,B=%d" % (keep, B)] = cell
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out, flush=True)

if __name__ == "__main__":
    main()
