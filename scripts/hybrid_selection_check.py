#!/usr/bin/env python3
"""hybrid_selection_check.py — 截断选择准则优化（用户: "PCA 比我们好，要优化"）。

混合准则: score_j = w_j^β · λ_j^(1-β)（w=margin t 统计量, λ=语料方差），
β∈{0,0.25,0.5,0.75,1}; β=0 即 PCA 序, β=1 即 margin 序。
另测: 级联（先按方差取 2m 再按 margin 取 m）、w_j·λ_j^{0.5}。
"""
import argparse, json
import numpy as np
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance


def eval_order(emb, qemb, qids, qrels, ids, order, m):
    sel = order[:m]
    r = recall_at_k(brute_rank(qemb[:, sel], emb[:, sel]), qids, qrels, ids)[0]
    return round(r, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keeps", nargs="+", type=float, default=[0.25, 0.125])
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "reference": round(ref, 4)}
    w = np.abs(margin_importance(emb, qemb, qids, qrels, ids))
    lam = emb.var(0)
    # 归一化到 [eps, 1] 防数值问题
    wn = w / (w.max() + 1e-12) + 1e-12
    ln = lam / (lam.max() + 1e-12) + 1e-12
    for keep in args.keeps:
        m = max(1, int(d * keep))
        res[f"keep={keep}_pca"] = eval_order(emb, qemb, qids, qrels, ids, np.argsort(-lam), m)
        res[f"keep={keep}_margin"] = eval_order(emb, qemb, qids, qrels, ids, np.argsort(-w), m)
        for beta in [0.25, 0.5, 0.75]:
            s = (wn ** beta) * (ln ** (1 - beta))
            r = eval_order(emb, qemb, qids, qrels, ids, np.argsort(-s), m)
            res[f"keep={keep}_beta{beta}"] = r
            print(f"keep={keep} beta={beta}: {r:.4f}", flush=True)
        # 级联: 方差取 top-2m, 再 margin 取 m
        casc = np.argsort(-lam)[:2 * m]
        sub = np.argsort(-w[casc])
        r = eval_order(emb, qemb, qids, qrels, ids, casc[sub[:m]], m)
        res[f"keep={keep}_cascade"] = r
        print(f"keep={keep} cascade: {r:.4f}", flush=True)
        print(f"keep={keep} pca={res[f'keep={keep}_pca']} margin={res[f'keep={keep}_margin']}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
