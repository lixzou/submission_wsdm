#!/usr/bin/env python3
"""完整管线对比 (train 校准, B=2 对所有 keep, 与 train_calib_main 一致):
plain PCA+量化 vs 加权PCA+量化 vs perm+量化。
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
    perm_order = np.argsort(-imp)
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d}
    for keep in [0.5, 0.25, 0.125]:
        m = max(1, int(d * keep)); m = (m // 8) * 8
        B = 2  # 与 train_calib_main 一致
        cell = {}
        # plain PCA + 量化
        V = pca_directions(emb, m)
        emb_p = emb @ V.T; q_p = qemb @ V.T
        P = random_orthogonal(m, seed=0)
        rot = (emb_p @ P).astype(np.float64)
        levels = fit_calib_levels(rot, train_qids, train_qrels, ids, B)
        cell["pca_quant"] = quant_recall(q_p, emb_p, np.arange(m), B, levels, qids, qrels, ids)
        # weighted PCA (alpha=1.0) + 量化 (RECO 口径)
        ww = imp / (imp.max() + 1e-12) + 1e-6
        Vw = pca_directions(emb, m, w=ww)
        emb_w = emb @ Vw.T; q_w = qemb @ Vw.T
        rot_w = (emb_w @ P).astype(np.float64)
        levels_w = fit_calib_levels(rot_w, train_qids, train_qrels, ids, B)
        cell["wpca_quant"] = quant_recall(q_w, emb_w, np.arange(m), B, levels_w, qids, qrels, ids)
        # perm + 量化
        sel = perm_order[:m]
        rot_p = (emb[:, sel] @ P).astype(np.float64)
        levels_p = fit_calib_levels(rot_p, train_qids, train_qrels, ids, B)
        cell["perm_quant"] = quant_recall(qemb, emb, sel, B, levels_p, qids, qrels, ids)
        res["keep=%.3f" % keep] = cell
        print(f"keep={keep} m={m}: pca={cell['pca_quant']:.4f} wpca={cell['wpca_quant']:.4f} perm={cell['perm_quant']:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
