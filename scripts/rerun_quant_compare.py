#!/usr/bin/env python3
"""完整管线对比 (train 校准): plain PCA+量化 vs 加权PCA+量化 vs RECO口径。
协议: PCA 投影 -> 随机旋转 -> fit_calib_levels(train校准) -> 量化 -> test 评估。
对照 merge_e5_v2 ablation 的未量化截断。
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
    for keep in [0.5, 0.25, 0.125]:
        m = max(1, int(d * keep))
        B = {0.5: 2, 0.25: 4, 0.125: 8}[keep]
        cell = {}
        # plain PCA + 量化
        V = pca_directions(emb, m)
        emb_p = emb @ V.T; q_p = qemb @ V.T
        P = random_orthogonal(m, seed=0)
        rot = (emb_p @ P).astype(np.float64)
        levels = fit_calib_levels(rot, train_qids, train_qrels, ids, B)
        cell["pca_quant"] = quant_recall(q_p, emb_p, np.arange(m), B, levels, qids, qrels, ids)
        # weighted PCA (alpha=1.0) + 量化
        ww = imp / (imp.max() + 1e-12) + 1e-6
        Vw = pca_directions(emb, m, w=ww)
        emb_w = emb @ Vw.T; q_w = qemb @ Vw.T
        rot_w = (emb_w @ P).astype(np.float64)
        levels_w = fit_calib_levels(rot_w, train_qids, train_qrels, ids, B)
        cell["wpca_quant"] = quant_recall(q_w, emb_w, np.arange(m), B, levels_w, qids, qrels, ids)
        res["keep=%.3f" % keep] = cell
        print(f"keep={keep}: pca_quant={cell['pca_quant']:.4f} wpca_quant={cell['wpca_quant']:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
