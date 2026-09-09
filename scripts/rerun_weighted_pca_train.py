#!/usr/bin/env python3
"""正文 ablation 段重跑 — 加权 PCA vs 无权重 PCA (train 校准, test 评估)。
协议: margin_importance 用 train 查询校准; pca_directions 无权重 vs 加权(alpha=0.5/1.0/2.0);
评估在 test 查询 (无泄漏)。输出 results/runs/weighted_pca_train_{model}.json
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
    ap.add_argument("--keeps", nargs="+", type=float, default=[0.5, 0.25, 0.125])
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    train_qids = json.load(open(f"{args.ds_dir}/train_qids.json"))
    train_qrels = json.load(open(f"{args.ds_dir}/train_qrels.json"))
    qemb_train = np.load(f"{args.ds_dir}/qemb_train_{args.model}.npy").astype(np.float32)
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d, "reference": round(ref, 4)}
    imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)

    for keep in args.keeps:
        m = max(1, int(d * keep))
        s = {}
        V = pca_directions(emb, m)
        s["pca"] = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]
        for wp in [0.5, 1.0, 2.0]:
            ww = imp**wp
            ww = ww / (ww.max() + 1e-12) + 1e-6
            Vw = pca_directions(emb, m, w=ww)
            s[f"impw_pca_p{wp}"] = recall_at_k(brute_rank(qemb @ Vw.T, emb @ Vw.T), qids, qrels, ids)[0]
        res["keep=%.3f" % keep] = s
        print(f"keep={keep}: " + " ".join(f"{k}={v:.4f}" for k, v in s.items()), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
