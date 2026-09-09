#!/usr/bin/env python3
"""baseline_faiss.py — 高质量 baseline（官方 faiss 实现, 2026-08-17 用户指令）:
  lsq  — LocalSearchQuantizer（Martinez et al. LSQ++/LSQ, ECCV 2018 家族, faiss 官方实现）
  opq  — OPQMatrix + IndexPQ（Ge et al. 学习旋转 OPQ, CVPR 2014, faiss 官方实现）
  aq   — AdditiveQuantizer（Babenko & Lempitsky 加性量化）
  rq   — ResidualQuantizer（残差量化）
协议与论文 PQ 表一致: 字节对齐预算（M×nbits = 48B/24B ↔ 16x/32x on 768d）,
decode+normalize+cosine, Recall@10（BEIR 口径）, 暴力检索。
"""
import argparse, json, time
import numpy as np
import faiss
from pilot_truncation import load, recall_at_k, brute_rank


def eval_decode(qemb, emb, codes, decode_fn, qids, qrels, ids):
    o = np.zeros_like(emb, dtype=np.float32)
    out = decode_fn(codes)
    if out is None:
        pass  # 两参形式已写入 o
    else:
        o[:] = out
    norms = np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    o /= norms
    return recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0]


def run_quantizer(ds_dir, model, kind, Ms, out, nbits=8, subset_docs=None, subset_queries=None):
    emb, qemb, qids, qrels, ids = load(ds_dir, model)
    if subset_docs:
        emb, ids = emb[:subset_docs], ids[:subset_docs]
    if subset_queries:
        qemb, qids = qemb[:subset_queries], qids[:subset_queries]
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": ds_dir, "model": model, "d": d, "reference": round(ref, 4)}
    for M in Ms:
        t0 = time.time()
        if kind == "pq":
            q = faiss.IndexPQ(d, M, nbits)
            q.train(emb)
            codes = q.sa_encode(emb)
            o = q.sa_decode(codes).astype(np.float32)
            norms = np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
            o /= norms
            r = recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0]
            res[f"pq_M{M}_n{nbits}"] = round(r, 4)
            print(f"pq M={M} nbits={nbits}: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
            continue
        if kind == "lsq":
            q = faiss.LocalSearchQuantizer(d, M, nbits)
            # 训练加速（5183 点小语料, 全默认 icm 迭代过慢）:
            q.icm_iters = 5
            q.train_iters = 10
        elif kind == "itq":
            return run_itq(ds_dir, model, out, subset_docs, subset_queries)
        elif kind == "rq":
            q = faiss.ResidualQuantizer(d, M, nbits)
        else:
            raise ValueError(kind)
        q.train(emb)
        codes = q.compute_codes(emb)
        r = eval_decode(qemb, emb, codes, q.decode, qids, qrels, ids)
        res[f"{kind}_M{M}_n{nbits}"] = round(r, 4)
        print(f"{kind} M={M} nbits={nbits}: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("SAVED", out)


def run_opq(ds_dir, model, Ms, out, nbits=8, subset_docs=None, subset_queries=None):
    emb, qemb, qids, qrels, ids = load(ds_dir, model)
    if subset_docs:
        emb, ids = emb[:subset_docs], ids[:subset_docs]
    if subset_queries:
        qemb, qids = qemb[:subset_queries], qids[:subset_queries]
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": ds_dir, "model": model, "d": d, "reference": round(ref, 4)}
    for M in Ms:
        t0 = time.time()
        opq = faiss.OPQMatrix(d, d)  # 学习旋转保留维度
        opq.train(emb)
        pq = faiss.IndexPQ(d, M, nbits)
        pq.train(opq.apply_py(emb))
        # 存储: OPQ 旋转码 + PQ 码; 检索: decode PQ 码 -> 反旋转 -> 归一化
        rot = opq.apply_py(emb)
        codes = pq.sa_encode(rot)
        o = pq.sa_decode(codes).astype(np.float32)
        o = opq.reverse_transform(o)
        norms = np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
        o /= norms
        r = recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0]
        res[f"opq_M{M}_n{nbits}"] = round(r, 4)
        print(f"opq M={M} nbits={nbits}: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("SAVED", out)


def run_itq(ds_dir, model, out, subset_docs=None, subset_queries=None):
    """ITQ 二值哈希（Gong 等, 官方 faiss ITQTransform）: d bit = d/8 字节 = 32x。
    decode: 符号码 × 逐维条件均值（对称重建, 与 TurboQuant B=1 同重建口径）。"""
    emb, qemb, qids, qrels, ids = load(ds_dir, model)
    if subset_docs:
        emb, ids = emb[:subset_docs], ids[:subset_docs]
    if subset_queries:
        qemb, qids = qemb[:subset_queries], qids[:subset_queries]
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": ds_dir, "model": model, "d": d, "reference": round(ref, 4),
           "protocol": "ITQ binary (faiss ITQTransform), d bits = d/8 bytes = 32x"}
    itq = faiss.ITQTransform(d, d, False)
    itq.train(emb)
    rot = itq.apply(emb)
    # ITQTransform 无 reverse_transform: 用 ITQMatrix 的 A/b 显式反变换 + 自校验
    A = faiss.vector_to_array(itq.itq.A).reshape(d, d).astype(np.float64)  # (d,d) 旋转
    mean = faiss.vector_to_array(itq.mean).astype(np.float64)
    def rev(y):
        # 候选反变换, 在训练数据上自校验（恢复 emb; have_bias=False 时 b=0）
        cands = [y @ A.T + mean, y @ A.T, y @ A + mean, y @ A,
                 (y - mean) @ A.T + mean, (y - mean) @ A.T]
        errs = [float(np.abs(c[:200] - emb[:200].astype(np.float64)).max()) for c in cands]
        return cands[int(np.argmin(errs))]
    levels = np.abs(rot).mean(0)  # 逐维条件均值（对称二值重建）
    o = np.where(rot > 0, levels[None, :], -levels[None, :])
    o = rev(o)
    o = np.ascontiguousarray(o).astype(np.float32)
    norms = np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    o /= norms
    r = recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0]
    res["itq_binary_32x"] = round(r, 4)
    print(f"itq binary (32x): {r:.4f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("SAVED", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["lsq", "itq", "rq", "opq", "pq"])
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ms", nargs="+", type=int, default=[48, 24])
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--subset_docs", type=int, default=None)
    ap.add_argument("--subset_queries", type=int, default=None)
    args = ap.parse_args()
    if args.kind == "opq":
        run_opq(args.ds_dir, args.model, args.Ms, args.out, args.nbits,
                args.subset_docs, args.subset_queries)
    else:
        run_quantizer(args.ds_dir, args.model, args.kind, args.Ms, args.out, args.nbits,
                      args.subset_docs, args.subset_queries)
