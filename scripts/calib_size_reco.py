#!/usr/bin/env python3
"""校准规模敏感性分析 — 完整 RECO 管线 (谱 + 网格都用同一批校准查询).

问题: 校准查询集大小对效果的影响有多大? 用多少 judged queries 就够?
协议: scifact e5, keep=50% B=2 (头条预算). 校准查询 = train_qids 前 N 个
(按 train 顺序取, 与 train_calib_size 一致); 评估在 test qids/qrels (零泄漏).
对每个规模:
  - reco: 完整管线 (加权 PCA 谱 + 校准网格, 都用这批查询校准)
  - perm: 仅变换 (margin 排序截断, 不量化) — 与旧 tab:calib 口径对比
  - prefix: 无校准基线 (前缀截断)
输出 JSON + 打印 per-fraction 表.
用法: python calib_size_reco.py --model e5 --ds_dir data/beir/scifact --out X.json
"""
import argparse, json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, brute_rank, recall_at_k, margin_importance
from main_table import run_reco, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5")
    ap.add_argument("--ds_dir", default="data/beir/scifact")
    ap.add_argument("--out", default="/tmp/calib_size_reco.json")
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--fracs", nargs="+", type=float,
                    default=[0.03, 0.06, 0.1, 0.25, 0.5, 0.7, 1.0])
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    m = max(1, int(d * args.keep)); m = (m // 8) * 8
    train_qids = json.load(open(f"{args.ds_dir}/train_qids.json"))
    train_qrels = json.load(open(f"{args.ds_dir}/train_qrels.json"))
    qemb_train = np.load(f"{args.ds_dir}/qemb_train_{args.model}.npy").astype(np.float32)
    n_train = len(train_qids)
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    print(f"{args.model}: train {n_train} test {len(qids)} d={d} m={m} ref={ref:.4f} B={B}", flush=True)

    # 无校准基线: prefix 截断 (同预算, 不量化)
    rank_p = brute_rank(qemb[:, :m].astype(np.float32), emb[:, :m].astype(np.float32))
    prefix = recall_at_k(rank_p, qids, qrels, ids)[0]
    print(f"prefix@keep={args.keep}: {prefix:.4f}", flush=True)

    res = {"ds_dir": args.ds_dir, "model": args.model, "keep": args.keep, "B": B,
           "n_train": n_train, "n_test": len(qids), "reference": round(ref, 4),
           "prefix": round(prefix, 4), "fracs": {}}
    for frac in args.fracs:
        n_cal = max(4, int(n_train * frac))
        cal_qids = train_qids[:n_cal]
        cal_qrels = {q: train_qrels[q] for q in cal_qids if q in train_qrels}
        imp = margin_importance(emb, qemb_train, cal_qids, cal_qrels, ids, calib_frac=1.0)

        # 完整管线 (谱 + 网格都用这批校准查询)
        try:
            r_reco = run_reco(emb, qemb, imp, qids, qrels, ids, m, B,
                              calib_qids=cal_qids, calib_qrels=cal_qrels)
        except Exception as e:
            print(f"  frac={frac} reco ERR {e}", flush=True)
            r_reco = None

        # 仅变换 (margin 排序截断, 不量化)
        sel = np.argsort(-imp)[:m]
        rank = brute_rank(qemb[:, sel].astype(np.float32), emb[:, sel].astype(np.float32))
        r_perm = recall_at_k(rank, qids, qrels, ids)[0]

        gain = (r_reco - prefix) * 100 if r_reco is not None else None
        res["fracs"][f"{frac:.2f}"] = {
            "n_cal": len(cal_qids), "reco": round(r_reco, 4) if r_reco else None,
            "perm": round(r_perm, 4), "gain_pct": round(gain, 1) if gain is not None else None}
        print(f"  frac={frac:.2f} n_cal={len(cal_qids)} reco={r_reco} perm={r_perm:.4f} "
              f"gain={gain:+}pp", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
