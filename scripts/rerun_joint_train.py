#!/usr/bin/env python3
"""tab:joint 重跑 (train 校准) — 32x 预算联合扫描, 诚实协议。
协议与 trunc_vs_fulldim_budget.py 一致, 但 margin_importance 用 train 查询校准,
评估在 test 查询。perm = margin ordering; pca = 无权重 PCA (不依赖校准);
prefix = 前 m 维; calib grid = fit_calib_levels(train 校准)。
输出 results/runs/joint_train_{model}.json
"""
import argparse, json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from rabitq_py import random_orthogonal, quantize_Bbit
from merge_pca_retrieval import pca_directions
from sota_pipeline_check import fit_calib_levels


def quant_recall(qemb, emb, sel, B, qids, qrels, ids, levels=None):
    m = len(sel)
    P = random_orthogonal(m, seed=0)
    rot = (emb[:, sel] @ P).astype(np.float32)
    if levels is None:
        codes, _ = quantize_Bbit(rot, B)
        y = codes.astype(np.float64) - ((2**B - 1) / 2.0)
        o = (P @ y.T).T
    else:
        codes = np.zeros((len(rot), m), np.int32)
        for j in range(m):
            codes[:, j] = np.argmin(np.abs(rot[:, j, None] - levels[j][None, :]), axis=1)
        rec = levels[np.arange(m)[None, :], codes]
        o = (P @ rec.T).T
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    return recall_at_k(brute_rank(qemb[:, sel].astype(np.float32), o.astype(np.float32)),
                       qids, qrels, ids)[0]


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
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d, "reference": round(ref, 4)}
    imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)
    perm_order = np.argsort(-imp)

    for keep, B in [(0.125, 8), (0.25, 4), (0.5, 2)]:  # 32x
        m = max(1, int(d * keep))
        # prefix
        r = quant_recall(qemb, emb, np.arange(m), B, qids, qrels, ids)
        res[f"32x_keep{keep}_prefix_B{B}"] = round(r, 4)
        # perm (margin ordering)
        r = quant_recall(qemb, emb, perm_order[:m], B, qids, qrels, ids)
        res[f"32x_keep{keep}_perm_B{B}"] = round(r, 4)
        # pca (无权重)
        V = pca_directions(emb, m)
        sel = np.arange(m)
        r = quant_recall(qemb @ V.T, emb @ V.T, sel, B, qids, qrels, ids)
        res[f"32x_keep{keep}_pca_B{B}"] = round(r, 4)
        print(f"32x keep={keep}: prefix={res[f'32x_keep{keep}_prefix_B{B}']:.4f} "
              f"perm={res[f'32x_keep{keep}_perm_B{B}']:.4f} "
              f"pca={res[f'32x_keep{keep}_pca_B{B}']:.4f}", flush=True)
        if keep == 0.5:
            # calib grid: perm 选维 + judged-fit levels (train 校准)
            P = random_orthogonal(m, seed=0)
            rot = (emb[:, perm_order[:m]] @ P).astype(np.float64)
            levels = fit_calib_levels(rot, train_qids, train_qrels, ids, B)
            r = quant_recall(qemb, emb, perm_order[:m], B, qids, qrels, ids, levels=levels)
            res[f"32x_keep{keep}_perm_calibgrid_B{B}"] = round(r, 4)
            print(f"  32x keep=0.5 perm calibgrid: {r:.4f}", flush=True)
    # 全维 32x = B1
    r = quant_recall(qemb, emb, np.arange(d), 1, qids, qrels, ids)
    res["32x_full_B1"] = round(r, 4)
    print(f"32x full B1: {r:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
