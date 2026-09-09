#!/usr/bin/env python3
"""tab:calib 重跑 — 校准规模敏感性, 诚实 train 协议 (train 查询校准, test 查询评估)。
协议与 train_calib_main.py 一致: margin_importance 用 train_qids/train_qrels/qemb_train,
按 calib_frac 取前 N% train 查询; 评估在 test qids/qrels (无泄漏)。
keep=12.5%, 纯坐标选择 (不量化), 与旧 calib_scifact_e5.json 口径一致。
用法: python train_calib_size.py --ds_dir data/beir/scifact --model e5 --out X.json
"""
import argparse, json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, brute_rank, recall_at_k, margin_importance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=0.125)
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.7, 1.0])
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    m = max(1, int(d * args.keep)); m = (m // 8) * 8
    train_qids = json.load(open(f"{args.ds_dir}/train_qids.json"))
    train_qrels = json.load(open(f"{args.ds_dir}/train_qrels.json"))
    qemb_train = np.load(f"{args.ds_dir}/qemb_train_{args.model}.npy").astype(np.float32)
    print(f"{args.model}: train {len(train_qids)} test {len(qids)} d={d} m={m}", flush=True)

    # 参考: prefix 截断 (不依赖校准, 新协议 test 评估)
    rank_p = brute_rank(qemb[:, :m].astype(np.float32), emb[:, :m].astype(np.float32))
    prefix = recall_at_k(rank_p, qids, qrels, ids)[0]
    print(f"prefix@keep={args.keep}: {prefix:.4f}", flush=True)

    res = {"ds_dir": args.ds_dir, "model": args.model, "keep": args.keep,
           "n_train": len(train_qids), "n_test": len(qids), "prefix": round(prefix, 4)}
    for frac in args.fracs:
        imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids,
                                calib_frac=frac)
        sel = np.argsort(-imp)[:m]
        rank = brute_rank(qemb[:, sel].astype(np.float32), emb[:, sel].astype(np.float32))
        r = recall_at_k(rank, qids, qrels, ids)[0]
        n_cal = int(len(train_qids) * frac)
        res[f"calib={frac:.2f}"] = {"n_cal": n_cal, "perm_recall": round(r, 4),
                                    "gain": round((r - prefix) * 100, 1)}
        print(f"  frac={frac:.2f} n_cal={n_cal} perm={r:.4f} gain={100*(r-prefix):+.1f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
