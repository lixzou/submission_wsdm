#!/usr/bin/env python3
"""train_calib_bootstrap.py — 诚实 train 校准下 Table 1 逐格显著性 (RECO vs best baseline)。

协议与 train_calib 主表一致:
  - margin_importance 拟合 TRAIN 查询 (tq/tr/qt), calib_frac=1.0
  - 网格 fit_calib_levels 用 TRAIN qrels
  - 评估在 TEST 查询 (零泄漏)
  - per-dataset α: scifact/nfcorpus=1.0, fiqa e5/bge=0.5, fiqa 其余=1.0
对每格: RECO 与各 baseline 的 per-query Recall@10 数组 → paired bootstrap (1000, seed=0)
输出: results/runs/train_calib_bootstrap_45.json, key="model@dataset@keep0.500"
"""
import argparse, json, os, sys
import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_table import run_faiss_quant
from pilot_truncation import load, margin_importance
from merge_pca_retrieval import pca_directions
from sota_pipeline_check import fit_calib_levels, quant_recall
from rabitq_py import random_orthogonal

faiss.omp_set_num_threads(int(os.environ.get("FAISS_THREADS", "4")))

KEEPS = [0.5, 0.75, 0.875]
B = 2
BASELINES = ["pq", "opq", "itq", "scalar", "tq"]
MODELS = ["e5", "bge", "mini", "bgem3", "qwen3"]
DSS = ["scifact", "fiqa", "nfcorpus"]
# per-dataset alpha (与 train_calib 主表一致)
def alpha_for(ds, model):
    if ds == "fiqa" and model in ("e5", "bge"):
        return 0.5
    return 1.0


def run_reco_alpha(emb, qemb, imp, qids, qrels, ids, m, B, alpha,
                   return_arrays=False, cq=None, cr=None):
    w = (imp / imp.max() + 1e-6) ** alpha
    V = pca_directions(emb, m, w=w)
    ep = emb @ V.T
    qp = qemb @ V.T
    P = random_orthogonal(m, seed=0)
    rot = (ep @ P).astype(np.float64)
    cq = cq if cq is not None else qids
    cr = cr if cr is not None else qrels
    levels = fit_calib_levels(rot, cq, cr, ids, B)
    return quant_recall(qp, ep, np.arange(m), B, levels, qids, qrels, ids,
                        return_arrays=return_arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/runs/train_calib_bootstrap_45.json")
    ap.add_argument("--cells", nargs="*", default=None, help="子串过滤, 如 e5@scifact")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out))

    for ds in DSS:
        ds_dir = f"data/beir/{ds}"
        for model in MODELS:
            keyp = f"{model}@{ds}"
            if args.cells and not any(c in keyp for c in args.cells):
                continue
            emb, qemb, qids, qrels, ids = load(ds_dir, model)
            d = emb.shape[1]
            tq = json.load(open(f"{ds_dir}/train_qids.json"))
            tr = json.load(open(f"{ds_dir}/train_qrels.json"))
            qt = np.load(f"{ds_dir}/qemb_train_{model}.npy").astype(np.float32)
            imp = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
            alpha = alpha_for(ds, model)
            for keep in KEEPS:
                mm = max(1, int(d * keep))
                if mm % 8 != 0:
                    mm = (mm // 8) * 8
                key = f"{model}@{ds}@keep{keep:.3f}"
                if key in out:
                    print("SKIP", key, flush=True)
                    continue
                perq = {}
                perq["reco"] = run_reco_alpha(emb, qemb, imp, qids, qrels, ids,
                                              mm, B, alpha, return_arrays=True,
                                              cq=tq, cr=tr)
                for meth in BASELINES:
                    r = run_faiss_quant(ds_dir, model, meth, emb, qemb, qids, qrels, ids,
                                        mm, return_arrays=True)
                    if r is not None:
                        perq[meth] = r
                means = {mm: float(np.mean(a)) for mm, a in perq.items()}
                best_base = max([m for m in means if m != "reco"], key=lambda m: means[m])
                reco_a = np.asarray(perq["reco"], dtype=np.float64)
                base_a = np.asarray(perq[best_base], dtype=np.float64)
                nq = len(reco_a)
                delta = reco_a - base_a
                boots = np.empty(args.n_boot)
                for t in range(args.n_boot):
                    idx = rng.integers(0, nq, nq)
                    boots[t] = delta[idx].mean()
                p = float(np.mean(boots <= 0))
                ci = np.percentile(boots, [2.5, 97.5])
                out[key] = {
                    "reco": round(float(means["reco"]), 4),
                    "best_base": best_base,
                    "best_base_recall": round(float(means[best_base]), 4),
                    "delta": round(float(delta.mean()), 4),
                    "p": p,
                    "ci_low": round(float(ci[0]), 4),
                    "ci_high": round(float(ci[1]), 4),
                    "n_queries": nq,
                    "alpha": alpha,
                    "sig_0001": p <= 0.001,
                    "sig_001": p <= 0.01,
                    "sig_005": p <= 0.05,
                }
                print(key, out[key], flush=True)
                json.dump(out, open(args.out, "w"), indent=1)
    json.dump(out, open(args.out, "w"), indent=1)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
