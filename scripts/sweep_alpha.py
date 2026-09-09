#!/usr/bin/env python3
"""RECO 旋转 α 扫描: 在指定 (ds_dir, model) 上测不同 importance 加权强度下 RECO 的 Recall@10。
α=1 当前 / 0.5 弱化 / 0 纯 PCA。用于找 qwen3 等 Matryoshka 模型能赢的配置。
用法: python sweep_alpha.py --ds_dir data/beir_big/arguana --model qwen3
"""
import argparse, os, sys
import numpy as np
import faiss

faiss.omp_set_num_threads(8)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, margin_importance
from merge_pca_retrieval import pca_directions
from sota_pipeline_check import fit_calib_levels, quant_recall
from rabitq_py import random_orthogonal

KEEPS = [0.5, 0.75, 0.875]
B = 2
ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]


def run_reco_alpha(emb, qemb, imp, qids, qrels, ids, m, B, alpha):
    w = (imp / imp.max() + 1e-6) ** alpha
    V = pca_directions(emb, m, w=w)
    emb_proj = emb @ V.T
    q_proj = qemb @ V.T
    P = random_orthogonal(m, seed=0)
    rot = (emb_proj @ P).astype(np.float64)
    levels = fit_calib_levels(rot, qids, qrels, ids, B)
    return quant_recall(q_proj, emb_proj, np.arange(m), B, levels, qids, qrels, ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=1.0)

    # PQ 前缀基线 (best baseline 参照)
    from main_table import run_faiss_quant
    print(f"== {args.model}@{os.path.basename(args.ds_dir)} ==")
    for keep in KEEPS:
        m = max(1, int(d * keep))
        if m % 8 != 0:
            m = (m // 8) * 8
        pq = run_faiss_quant(args.ds_dir, args.model, "pq", emb, qemb, qids, qrels, ids, m)
        row = []
        for a in ALPHAS:
            r = run_reco_alpha(emb, qemb, imp, qids, qrels, ids, m, B, a)
            row.append(f"α={a}: {r:.4f}")
        print(f"keep={keep} m={m} | PQ {pq:.4f} | " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
