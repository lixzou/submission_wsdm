#!/usr/bin/env python3
"""reco_preprocess_minimal.py — 最简验证: RECO 前置对 PQ 的增益
3 个模型 × 2 个预算, 只测 PQ (最快)"""
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

def main():
    ds_dir = "data/beir/scifact"
    calib_frac = 0.7
    keep = 0.5

    all_results = {}
    for model in ["e5", "bge", "mini"]:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        emb_full, qemb, qids, qrels, ids = load(ds_dir, model)
        d_full = emb_full.shape[1]
        print(f"Loaded: {emb_full.shape[0]} docs, d={d_full}")

        # margin importance 排序
        order = margin_importance(emb_full, qemb, qids, qrels, ids, calib_frac=calib_frac)
        perm = np.argsort(-order)

        # 截断到 50%
        d_trunc = int(d_full * keep)
        emb_trunc = emb_full[:, perm[:d_trunc]].copy()
        qemb_trunc = qemb[:, perm[:d_trunc]].copy()
        print(f"Truncated: {d_full} -> {d_trunc}")

        # 参考精度
        ref = eval_recall(qemb, emb_full, qids, qrels, ids)
        print(f"Reference: {ref:.4f}")

        model_results = {"d_full": d_full, "d_trunc": d_trunc, "reference": round(ref, 4)}

        # 测试两个预算
        for M, nbits, label in [(48, 8, "M48"), (24, 8, "M24")]:
            print(f"\n--- {label} (M={M}, nbits={nbits}) ---")

            print(f"[Full {d_full}d] PQ {label}:")
            pq_full = run_pq(emb_full, qemb, qids, qrels, ids, d_full, M, nbits)

            print(f"[Trunc {d_trunc}d] PQ {label}:")
            pq_trunc = run_pq(emb_trunc, qemb_trunc, qids, qrels, ids, d_trunc, M, nbits)

            model_results[label] = {
                "pq_full": round(pq_full, 4),
                "pq_reco": round(pq_trunc, 4),
                "pq_delta": round(pq_trunc - pq_full, 4),
            }
            print(f"  Delta: {pq_trunc - pq_full:+.4f}")

        all_results[model] = model_results

    # 总结
    print(f"\n{'='*60}")
    print("SUMMARY: PQ alone vs RECO+PQ (same byte budget)")
    print(f"{'='*60}")
    for model in ["e5", "bge", "mini"]:
        r = all_results[model]
        for label in ["M48", "M24"]:
            e = r[label]
            print(f"{model} {label}: PQ={e['pq_full']:.4f} -> RECO+PQ={e['pq_reco']:.4f} ({e['pq_delta']:+.4f})")

    json.dump(all_results, open("results/runs/reco_preprocess_pq.json", "w"), indent=2)
    print("\nSAVED results/runs/reco_preprocess_pq.json")

if __name__ == "__main__":
    main()
