#!/usr/bin/env python3
"""reco_preprocess_check.py — RECO 前置处理对各种方法的增益实验
测试: OPQ, ITQ, PQ(不截断), RQ 在相同字节预算下, 有/无 RECO 置换截断的精度差异
协议: scifact/e5, 32x 字节预算, Recall@10"""
import argparse, json, time, sys, os
import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(__file__))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance


def get_margin_order(ds_dir, model, calib_frac=0.7):
    """获取 margin importance 排序"""
    emb, qemb, qids, qrels, ids = load(ds_dir, model)
    order = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=calib_frac)
    return order, emb, qemb, qids, qrels, ids


def eval_recall(qemb, o, qids, qrels, ids):
    norms = np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    o = o / norms
    return recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0]


def run_opq_experiment(emb, qemb, qids, qrels, ids, d, M, nbits=8):
    """OPQ: 学习旋转 + PQ"""
    t0 = time.time()
    opq = faiss.OPQMatrix(d, d)
    opq.train(emb)
    rot = opq.apply_py(emb)
    pq = faiss.IndexPQ(d, M, nbits)
    pq.train(rot)
    codes = pq.sa_encode(rot)
    o = pq.sa_decode(codes).astype(np.float32)
    o = opq.reverse_transform(o)
    r = eval_recall(qemb, o, qids, qrels, ids)
    print(f"  OPQ M={M}: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
    return r


def run_itq_experiment(emb, qemb, qids, qrels, ids, d):
    """ITQ: 二值哈希"""
    t0 = time.time()
    itq = faiss.ITQTransform(d, d, False)
    itq.train(emb)
    rot = itq.apply(emb)
    A = faiss.vector_to_array(itq.itq.A).reshape(d, d).astype(np.float64)
    mean = faiss.vector_to_array(itq.mean).astype(np.float64)
    levels = np.abs(rot).mean(0)
    o = np.where(rot > 0, levels[None, :], -levels[None, :])
    # reverse transform
    cands = [o @ A.T + mean, o @ A.T, o @ A + mean, o @ A,
             (o - mean) @ A.T + mean, (o - mean) @ A.T]
    errs = [float(np.abs(c[:200] - emb[:200].astype(np.float64)).max()) for c in cands]
    o = cands[int(np.argmin(errs))]
    o = np.ascontiguousarray(o).astype(np.float32)
    r = eval_recall(qemb, o, qids, qrels, ids)
    print(f"  ITQ binary: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
    return r


def run_pq_experiment(emb, qemb, qids, qrels, ids, d, M, nbits=8):
    """PQ: 标准乘积量化"""
    t0 = time.time()
    pq = faiss.IndexPQ(d, M, nbits)
    pq.train(emb)
    codes = pq.sa_encode(emb)
    o = pq.sa_decode(codes).astype(np.float32)
    r = eval_recall(qemb, o, qids, qrels, ids)
    print(f"  PQ M={M}: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
    return r


def run_rq_experiment(emb, qemb, qids, qrels, ids, d, M, nbits=8):
    """RQ: 残差量化"""
    t0 = time.time()
    rq = faiss.ResidualQuantizer(d, M, nbits)
    rq.train(emb)
    codes = rq.compute_codes(emb)
    o = rq.decode(codes).astype(np.float32)
    r = eval_recall(qemb, o, qids, qrels, ids)
    print(f"  RQ M={M}: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", default="data/beir/scifact")
    ap.add_argument("--model", default="e5")
    ap.add_argument("--calib_frac", type=float, default=0.7)
    ap.add_argument("--out", default="reco_preprocess_results.json")
    args = ap.parse_args()

    # 获取 margin 排序
    order, emb_full, qemb, qids, qrels, ids = get_margin_order(
        args.ds_dir, args.model, args.calib_frac)
    d_full = emb_full.shape[1]
    print(f"Loaded {args.ds_dir}/{args.model}: {emb_full.shape[0]} docs, d={d_full}")

    # 截断到 50%
    keep = 0.5
    d_trunc = int(d_full * keep)
    perm = np.argsort(-order)
    emb_trunc = emb_full[:, perm[:d_trunc]].copy()
    qemb_trunc = qemb[:, perm[:d_trunc]].copy()

    # 参考精度
    ref = eval_recall(qemb, emb_full, qids, qrels, ids)
    print(f"Reference (full dim): {ref:.4f}")

    results = {
        "ds_dir": args.ds_dir,
        "model": args.model,
        "d_full": d_full,
        "d_trunc": d_trunc,
        "keep": keep,
        "reference": round(ref, 4),
        "experiments": {}
    }

    # 字节预算: 32x = 96 bytes on 768d
    # 对于 384d: M=96, nbits=8 = 96 bytes (same budget)
    # 对于 768d: M=96, nbits=8 = 96 bytes
    budgets = [
        {"M": 96, "nbits": 8, "label": "M96_n8"},
        {"M": 48, "nbits": 8, "label": "M48_n8"},
    ]

    for budget in budgets:
        M, nbits = budget["M"], budget["nbits"]
        label = budget["label"]
        print(f"\n=== Budget: {label} (M={M}, nbits={nbits}) ===")

        # PQ
        print(f"\n[Full dim {d_full}] PQ {label}:")
        pq_full = run_pq_experiment(emb_full, qemb, qids, qrels, ids, d_full, M, nbits)
        print(f"[Trunc dim {d_trunc}] PQ {label}:")
        pq_trunc = run_pq_experiment(emb_trunc, qemb_trunc, qids, qrels, ids, d_trunc, M, nbits)

        # OPQ
        print(f"\n[Full dim {d_full}] OPQ {label}:")
        opq_full = run_opq_experiment(emb_full, qemb, qids, qrels, ids, d_full, M, nbits)
        print(f"[Trunc dim {d_trunc}] OPQ {label}:")
        opq_trunc = run_opq_experiment(emb_trunc, qemb_trunc, qids, qrels, ids, d_trunc, M, nbits)

        # RQ
        print(f"\n[Full dim {d_full}] RQ {label}:")
        rq_full = run_rq_experiment(emb_full, qemb, qids, qrels, ids, d_full, M, nbits)
        print(f"[Trunc dim {d_trunc}] RQ {label}:")
        rq_trunc = run_rq_experiment(emb_trunc, qemb_trunc, qids, qrels, ids, d_trunc, M, nbits)

        results["experiments"][label] = {
            "pq_full": round(pq_full, 4),
            "pq_reco": round(pq_trunc, 4),
            "pq_delta": round(pq_trunc - pq_full, 4),
            "opq_full": round(opq_full, 4),
            "opq_reco": round(opq_trunc, 4),
            "opq_delta": round(opq_trunc - opq_full, 4),
            "rq_full": round(rq_full, 4),
            "rq_reco": round(rq_trunc, 4),
            "rq_delta": round(rq_trunc - rq_full, 4),
        }

    # ITQ (特殊: 二值哈希, 固定 32x)
    print(f"\n[Full dim {d_full}] ITQ binary:")
    itq_full = run_itq_experiment(emb_full, qemb, qids, qrels, ids, d_full)
    print(f"[Trunc dim {d_trunc}] ITQ binary:")
    itq_trunc = run_itq_experiment(emb_trunc, qemb_trunc, qids, qrels, ids, d_trunc)

    results["experiments"]["itq"] = {
        "itq_full": round(itq_full, 4),
        "itq_reco": round(itq_trunc, 4),
        "itq_delta": round(itq_trunc - itq_full, 4),
    }

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSAVED: {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
