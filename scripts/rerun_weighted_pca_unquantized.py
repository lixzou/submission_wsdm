#!/usr/bin/env python3
"""正文 ablation 段精确重跑 — 未量化截断 (与 spectral_guided_rows 一致)。
直接 PCA 投影到 m 维, 无量化, train 校准 w, test 评估。
"""
import argparse, json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
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
    w_train = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d}
    for keep in [0.5, 0.25, 0.125]:
        m = max(1, int(d * keep)); m = (m // 8) * 8
        cell = {}
        V = pca_directions(emb, m)
        cell["pca"] = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]
        for wp in [0.5, 1.0, 2.0]:
            ww = w_train**wp; ww = ww/(ww.max()+1e-12)+1e-6
            Vw = pca_directions(emb, m, w=ww)
            cell[f"impw_pca_p{wp}"] = recall_at_k(brute_rank(qemb @ Vw.T, emb @ Vw.T), qids, qrels, ids)[0]
        res["keep=%.3f" % keep] = cell
        print(f"keep={keep}: " + " ".join(f"{k}={v:.4f}" for k,v in cell.items()), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
