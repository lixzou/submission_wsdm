#!/usr/bin/env python3
"""reco_preprocess_quick.py — 快速验证 RECO 前置处理对 OPQ/PQ/RQ 的增益
使用 mini 模型 (384d) 加速"""
import json, time, sys, os
import numpy as np
import faiss
sys.path.insert(0, os.path.dirname(__file__))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance

def eval_recall(qemb, o, qids, qrels, ids):
    norms = np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    o = o / norms
    return recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0]

def run_pq(emb, qemb, qids, qrels, ids, d, M, nbits=8):
    t0 = time.time()
    pq = faiss.IndexPQ(d, M, nbits)
    pq.train(emb)
    codes = pq.sa_encode(emb)
    o = pq.sa_decode(codes).astype(np.float32)
    r = eval_recall(qemb, o, qids, qrels, ids)
    print(f"  PQ M={M}: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
    return r

def run_opq(emb, qemb, qids, qrels, ids, d, M, nbits=8):
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

def run_rq(emb, qemb, qids, qrels, ids, d, M, nbits=8):
    t0 = time.time()
    rq = faiss.ResidualQuantizer(d, M, nbits)
    rq.train(emb)
    codes = rq.compute_codes(emb)
    o = rq.decode(codes).astype(np.float32)
    r = eval_recall(qemb, o, qids, qrels, ids)
    print(f"  RQ M={M}: {r:.4f} [{time.time()-t0:.0f}s]", flush=True)
    return r

def main():
    ds_dir = "data/beir/scifact"
    model = "mini"  # 384d, faster
    calib_frac = 0.7

    emb_full, qemb, qids, qrels, ids = load(ds_dir, model)
    d_full = emb_full.shape[1]
    print(f"Loaded {ds_dir}/{model}: {emb_full.shape[0]} docs, d={d_full}")

    # margin importance 排序
    order = margin_importance(emb_full, qemb, qids, qrels, ids, calib_frac=calib_frac)
    perm = np.argsort(-order)

    # 截断到 50%
    keep = 0.5
    d_trunc = int(d_full * keep)
    emb_trunc = emb_full[:, perm[:d_trunc]].copy()
    qemb_trunc = qemb[:, perm[:d_trunc]].copy()
    print(f"Truncated: {d_full} -> {d_trunc}")

    # 参考精度
    ref = eval_recall(qemb, emb_full, qids, qrels, ids)
    print(f"Reference (full dim): {ref:.4f}")

    results = {"model": model, "d_full": d_full, "d_trunc": d_trunc, "keep": keep,
               "reference": round(ref, 4)}

    # 字节预算: M=48, nbits=8 = 48 bytes
    # 384d: M=48 → 48 bytes (8x compression)
    # 192d: M=48 → 48 bytes (4x compression) — same byte budget!
    M = 48
    nbits = 8

    print(f"\n=== M={M}, nbits={nbits} (48 bytes per vector) ===")

    # PQ
    print(f"\n[Full {d_full}d] PQ:")
    pq_full = run_pq(emb_full, qemb, qids, qrels, ids, d_full, M, nbits)
    print(f"[Trunc {d_trunc}d] PQ:")
    pq_trunc = run_pq(emb_trunc, qemb_trunc, qids, qrels, ids, d_trunc, M, nbits)

    # OPQ
    print(f"\n[Full {d_full}d] OPQ:")
    opq_full = run_opq(emb_full, qemb, qids, qrels, ids, d_full, M, nbits)
    print(f"[Trunc {d_trunc}d] OPQ:")
    opq_trunc = run_opq(emb_trunc, qemb_trunc, qids, qrels, ids, d_trunc, M, nbits)

    # RQ
    print(f"\n[Full {d_full}d] RQ:")
    rq_full = run_rq(emb_full, qemb, qids, qrels, ids, d_full, M, nbits)
    print(f"[Trunc {d_trunc}d] RQ:")
    rq_trunc = run_rq(emb_trunc, qemb_trunc, qids, qrels, ids, d_trunc, M, nbits)

    results["experiments"] = {
        "pq_full": round(pq_full, 4), "pq_reco": round(pq_trunc, 4),
        "pq_delta": round(pq_trunc - pq_full, 4),
        "opq_full": round(opq_full, 4), "opq_reco": round(opq_trunc, 4),
        "opq_delta": round(opq_trunc - opq_full, 4),
        "rq_full": round(rq_full, 4), "rq_reco": round(rq_trunc, 4),
        "rq_delta": round(rq_trunc - rq_full, 4),
    }

    # 总结
    print(f"\n=== SUMMARY ===")
    print(f"PQ:  {pq_full:.4f} -> {pq_trunc:.4f} (delta={pq_trunc-pq_full:+.4f})")
    print(f"OPQ: {opq_full:.4f} -> {opq_trunc:.4f} (delta={opq_trunc-opq_full:+.4f})")
    print(f"RQ:  {rq_full:.4f} -> {rq_trunc:.4f} (delta={rq_trunc-rq_full:+.4f})")

    json.dump(results, open("results/runs/reco_preprocess_mini.json", "w"), indent=2)
    print("\nSAVED results/runs/reco_preprocess_mini.json")

if __name__ == "__main__":
    main()
