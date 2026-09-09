#!/usr/bin/env python3
"""正文 ablation 段精确重跑 — 加权PCA vs plain PCA, 32x 预算协议 (12.5%B8,25%B4,50%B2)。
与 train_calib_main 一致: train 校准, test 评估。验证正文 .821/.808/.822/.812/.715/.720。
"""
import argparse, json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from rabitq_py import random_orthogonal
from sota_pipeline_check import fit_calib_levels, quant_recall
from merge_pca_retrieval import pca_directions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    train_qids = json.load(open(f"{args.ds_dir}/train_qids.json"))
    train_qrels = json.load(open(f"{args.ds_dir}/train_qrels.json"))
    qemb_train = np.load(f"{args.ds_dir}/qemb_train_{args.model}.npy").astype(np.float32)
    imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d}
    for keep, B in [(0.5, 2), (0.25, 4), (0.125, 8)]:
        m = max(1, int(d * keep)); m = (m // 8) * 8
        cell = {}
        P = random_orthogonal(m, seed=0)
        # plain PCA
        V = pca_directions(emb, m)
        emb_p = emb @ V.T; q_p = qemb @ V.T
        rot = (emb_p @ P).astype(np.float64)
        levels = fit_calib_levels(rot, train_qids, train_qrels, ids, B)
        cell["pca"] = quant_recall(q_p, emb_p, np.arange(m), B, levels, qids, qrels, ids)
        # weighted PCA (alpha=1.0, RECO 口径)
        ww = imp / (imp.max() + 1e-12) + 1e-6
        Vw = pca_directions(emb, m, w=ww)
        emb_w = emb @ Vw.T; q_w = qemb @ Vw.T
        rot_w = (emb_w @ P).astype(np.float64)
        levels_w = fit_calib_levels(rot_w, train_qids, train_qrels, ids, B)
        cell["wpca"] = quant_recall(q_w, emb_w, np.arange(m), B, levels_w, qids, qrels, ids)
        res["keep=%.3f" % keep] = cell
        print(f"keep={keep} B={B}: pca={cell['pca']:.4f} wpca={cell['wpca']:.4f} diff={1000*(cell['wpca']-cell['pca']):+.1f}pp", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
