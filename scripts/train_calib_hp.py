#!/usr/bin/env python3
"""hotpotqa20k 训练校准: train 查询校准, 719 测试查询子集评估."""
import json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import faiss
faiss.omp_set_num_threads(int(os.environ.get("FAISS_THREADS", "4")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, brute_rank, recall_at_k, margin_importance
from main_table import run_reco, run_faiss_quant, KEEPS, B

DS = "/ssd1/zoulixin/tencent_previous_compression/experiments/data/beir_big/hotpotqa20k"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = args.model

    # 载入语料 + 719 评估查询子集 + train 校准查询
    emb = np.load(f"{DS}/emb_{m}.npy").astype(np.float32)
    meta = json.load(open(f"{DS}/meta.json"))
    ids = meta["ids"]
    qids = [str(q) for q in np.load(f"{DS}/qids1000.npy")]
    qrels = {str(q): rel for q, rel in json.load(open(f"{DS}/qrels1000.json")).items()}
    qemb = np.load(f"{DS}/qemb_{m}_1000.npy").astype(np.float32)
    d = emb.shape[1]

    train_qids = json.load(open(f"{DS}/train_qids.json"))
    train_qrels = json.load(open(f"{DS}/train_qrels.json"))
    qemb_train = np.load(f"{DS}/qemb_train_{m}.npy").astype(np.float32)

    print(f"{m}: train {len(train_qids)}, eval {len(qids)}, corpus {len(ids)}", flush=True)
    imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]

    res = {"ds_dir": DS, "model": m, "d": d, "reference": round(ref, 4),
           "calib": "train", "n_train": len(train_qids), "n_eval": len(qids)}
    for keep in KEEPS:
        mk = max(1, int(d * keep))
        if mk % 8 != 0:
            mk = (mk // 8) * 8
        cell = {}
        try:
            r = run_reco(emb, qemb, imp, qids, qrels, ids, mk, B,
                         calib_qids=train_qids, calib_qrels=train_qrels)
            cell["reco"] = round(r, 4)
        except Exception as e:
            print("reco ERR", e, flush=True)
        for meth in ["pq", "opq", "itq", "scalar", "tq"]:
            try:
                r = run_faiss_quant(DS, m, meth, emb, qemb, qids, qrels, ids, mk)
                if r is not None:
                    cell[meth] = round(r, 4)
            except Exception as e:
                print(meth, "ERR", e, flush=True)
        res["keep=%.3f" % keep] = cell
        print(f"  keep={keep} {cell}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
