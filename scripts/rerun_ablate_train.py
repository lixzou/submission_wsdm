#!/usr/bin/env python3
"""tab:ablate 重跑 — 截断顺序消融 (train 校准, test 评估), 诚实协议。
protocol: margin_importance 用 train 查询, 评估在 test 查询 (无泄漏)。
输出 results/runs/ablate_train_{model}.json
"""
import argparse, json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance


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
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d}
    imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)
    var_order = np.argsort(-emb.var(0))
    rng = np.random.default_rng(42)
    for keep in [0.5, 0.25, 0.125]:
        m = max(1, int(d * keep))
        s = {}
        s["prefix"] = recall_at_k(brute_rank(qemb[:, :m], emb[:, :m]), qids, qrels, ids)[0]
        ridx = rng.permutation(d)[:m]
        s["random"] = recall_at_k(brute_rank(qemb[:, ridx], emb[:, ridx]), qids, qrels, ids)[0]
        s["margin"] = recall_at_k(brute_rank(qemb[:, np.argsort(-imp)[:m]], emb[:, np.argsort(-imp)[:m]]), qids, qrels, ids)[0]
        s["variance"] = recall_at_k(brute_rank(qemb[:, var_order[:m]], emb[:, var_order[:m]]), qids, qrels, ids)[0]
        res["keep=%.3f" % keep] = s
        print(f"keep={keep}: " + " ".join(f"{k}={v:.4f}" for k, v in s.items()), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
