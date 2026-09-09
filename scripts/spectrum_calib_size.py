#!/usr/bin/env python3
"""校准规模分析 — 隔离谱: 网格固定用全部 train 查询, 只变谱 (margin_importance) 校准查询数.

问题: ranking-importance 谱需要多少 judged queries 才饱和?
协议: scifact e5, keep=50% B=2. fit_calib_levels 用全部 809 train 查询 (网格满);
margin_importance 只用前 N 个 train 查询 (谱变化). 评估 test (零泄漏).
用法: python spectrum_calib_size.py --model e5 --ds_dir data/beir/scifact --out X.json
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
    ap.add_argument("--out", default="/tmp/spectrum_calib_size.json")
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

    # 网格满: 全部 train 查询
    full_qids = train_qids
    full_qrels = train_qrels

    rank_p = brute_rank(qemb[:, :m].astype(np.float32), emb[:, :m].astype(np.float32))
    prefix = recall_at_k(rank_p, qids, qrels, ids)[0]
    print(f"prefix@keep={args.keep}: {prefix:.4f}", flush=True)

    res = {"ds_dir": args.ds_dir, "model": args.model, "keep": args.keep, "B": B,
           "n_train": n_train, "n_test": len(qids), "reference": round(ref, 4),
           "prefix": round(prefix, 4), "spectrum_only": True, "fracs": {}}
    for frac in args.fracs:
        n_cal = max(4, int(n_train * frac))
        cal_qids = train_qids[:n_cal]
        cal_qrels = {q: train_qrels[q] for q in cal_qids if q in train_qrels}
        imp = margin_importance(emb, qemb_train, cal_qids, cal_qrels, ids, calib_frac=1.0)
        # 网格用全部, 谱用子集
        r = run_reco(emb, qemb, imp, qids, qrels, ids, m, B,
                     calib_qids=full_qids, calib_qrels=full_qrels)
        gain = (r - prefix) * 100
        res["fracs"][f"{frac:.2f}"] = {"n_cal": len(cal_qids), "reco": round(r, 4),
                                       "gain_pct": round(gain, 1)}
        print(f"  frac={frac:.2f} n_cal={len(cal_qids)} reco={r:.4f} gain={gain:+.1f}pp", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
