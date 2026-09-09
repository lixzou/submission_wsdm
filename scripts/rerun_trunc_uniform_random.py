#!/usr/bin/env python3
"""截断消融: 随机均匀旋转 (uniform) vs 加权投影 (alpha 扫描)。
协议与 rerun_weighted_pca_train.py 一致: margin_importance 用 train 校准,
无量化, 评估在 test 查询。random_rot = 随机正交旋转后截断 m 维 (uniform baseline)。
输出 results/runs/trunc_uniform_random_{model}.json
"""
import argparse, json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from merge_pca_retrieval import pca_directions
from rabitq_py import random_orthogonal


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

    R = random_orthogonal(d, seed=0)          # 随机正交旋转 (uniform)
    emb_rot = emb @ R
    qemb_rot = qemb @ R

    for keep in args.keeps:
        m = max(1, int(d * keep))
        s = {}
        # uniform baseline: 随机旋转后取前 m 维 (坐标可交换 -> 均匀截断)
        s["random_rot"] = recall_at_k(
            brute_rank(qemb_rot[:, :m], emb_rot[:, :m]), qids, qrels, ids)[0]
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
