#!/usr/bin/env python3
"""训练集校准: 用 BEIR train qrels 校准 ranking-importance + 网格, 在测试集评估.
用于有 train split 的数据集 (scifact). 用法: python train_calib_main.py --model e5 --out X.json
"""
import json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import faiss
faiss.omp_set_num_threads(int(os.environ.get("FAISS_THREADS", "4")))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, brute_rank, recall_at_k
from main_table import run_reco, run_faiss_quant

DS = os.environ.get("DS_DIR", "data/beir/scifact")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ds_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keeps", type=str, default="0.5",
                    help="comma-separated retained fractions")
    ap.add_argument("--bits", type=str, default="2",
                    help="comma-separated bit widths (e.g. 2,4,8 for 32x/16x/8x)")
    args = ap.parse_args()
    global DS
    if args.ds_dir: DS = args.ds_dir

    keeps = [float(x) for x in args.keeps.split(",")]
    bits = [int(x) for x in args.bits.split(",")]

    emb, qemb, qids, qrels, ids = load(DS, args.model)
    d = emb.shape[1]
    # 训练集: qids/qrels + 查询嵌入
    train_qids = json.load(open(f"{DS}/train_qids.json"))
    train_qrels = json.load(open(f"{DS}/train_qrels.json"))
    qemb_train = np.load(f"{DS}/qemb_train_{args.model}.npy").astype(np.float32)
    print(f"{args.model}: train {len(train_qids)} queries, test {len(qids)}", flush=True)

    # 校准 spectrum: margin_importance 用训练查询嵌入 + train qrels
    from pilot_truncation import margin_importance
    imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)

    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": DS, "model": args.model, "d": d, "reference": round(ref, 4),
           "calib": "train", "n_train": len(train_qids), "keeps": keeps, "bits": bits}
    for keep in keeps:
        m = max(1, int(d * keep))
        if m % 8 != 0:
            m = (m // 8) * 8
        for B in bits:
            cell = {}
            try:
                r = run_reco(emb, qemb, imp, qids, qrels, ids, m, B,
                             calib_qids=train_qids, calib_qrels=train_qrels)
                cell["reco"] = round(r, 4)
            except Exception as e:
                print("reco ERR", e, flush=True)
            for meth in ["pq", "opq", "itq", "scalar", "tq"]:
                try:
                    r = run_faiss_quant(DS, args.model, meth, emb, qemb, qids, qrels, ids, m, B)
                    if r is not None:
                        cell[meth] = round(r, 4)
                except Exception as e:
                    print(meth, "ERR", e, flush=True)
            res["keep=%.3f,B=%d" % (keep, B)] = cell
            print(f"  keep={keep} B={B} {cell}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
