#!/usr/bin/env python3
"""数据泄漏修复: 校准集与评估集严格分开.

对每个 (ds_dir, model): 把 judged 查询切成 校准(前 calib_frac) + 评估(后 1-calib_frac).
在校准集上估计 ranking-importance 谱和量化网格, 只在评估集(留出)上算 Recall@10.
这避免"校准集=评估集"的测试集污染.
用法: python heldout_main.py --ds_dir X --model M --out Y.json  (每格一个进程)
"""
import argparse, json, os, sys
import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
import faiss
faiss.omp_set_num_threads(int(os.environ.get("FAISS_THREADS", "6")))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, margin_importance
from main_table import run_reco, run_faiss_quant, KEEPS, B


def split_queries(qids, qrels, calib_frac):
    """judged = 有标注的查询; 校准=前 calib_frac, 评估=其余."""
    judged = [q for q in qids if q in qrels]
    n_cal = int(len(judged) * calib_frac)
    cal_q = judged[:n_cal]
    eval_q = judged[n_cal:]
    eval_qrels = {q: qrels[q] for q in eval_q}
    return cal_q, eval_q, eval_qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib_frac", type=float, default=0.5,
                    help="校准占比 (评估=1-占比). 建议 scifact/ArguAna 0.5, NQ 0.7")
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    cal_q, eval_q, eval_qrels = split_queries(qids, qrels, args.calib_frac)
    print(f"{args.model}@{os.path.basename(args.ds_dir)}: calib {len(cal_q)} eval {len(eval_q)}", flush=True)

    # 校准集上的 ranking-importance 谱 (与网格校准用同一批校准查询)
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=args.calib_frac)
    cal_qrels = {q: qrels[q] for q in cal_q}
    # 全宽参照 (在评估集上; 传全量 qids 对齐 rank 索引, qrels 只含评估查询)
    from pilot_truncation import brute_rank, recall_at_k
    ref = recall_at_k(brute_rank(qemb, emb), qids, eval_qrels, ids)[0]

    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d,
           "calib_frac": args.calib_frac, "n_calib": len(cal_q), "n_eval": len(eval_q),
           "reference": round(ref, 4)}
    for keep in KEEPS:
        m = max(1, int(d * keep))
        if m % 8 != 0:
            m = (m // 8) * 8
        cell = {}
        # 在评估集上评估所有方法 (run 函数接受 qids/qrels 作为评估查询)
        try:
            r = run_reco(emb, qemb, imp, qids, eval_qrels, ids, m, B,
                         calib_qids=cal_q, calib_qrels=cal_qrels)
            cell["reco"] = round(r, 4)
        except Exception as e:
            print("reco ERR", e, flush=True)
        for meth in ["pq", "opq", "itq", "scalar", "tq"]:
            try:
                r = run_faiss_quant(args.ds_dir, args.model, meth, emb, qemb,
                                    qids, eval_qrels, ids, m)
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
