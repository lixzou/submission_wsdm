#!/usr/bin/env python3
"""Table 1 逐格显著性: RECO vs best baseline 的 paired bootstrap (主表转置 + 显著性)。

对每个 (ds_dir, model, keep) 单元:
  - 各方法返回 per-query Recall@10 (return_arrays=True)
  - best baseline = 非 RECO 中均值最高者 (runner-up)
  - paired bootstrap (1000 resamples, 固定种子): one-sided p = Pr(mean(delta)<=0)
输出 JSON: {cell_key: {reco, best_base, best_base_recall, delta, p, ci, sig_*}}
cell_key 格式: "<model>@<dataset>@keep<KEEP>"
"""
import argparse, json, os, sys
import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_table import run_reco, run_faiss_quant, KEEPS, B
from pilot_truncation import load, margin_importance

# faiss 线程数可调: 并行跑多进程时避免超订 (main_table import 后覆盖)
faiss.omp_set_num_threads(int(os.environ.get("FAISS_THREADS", "32")))

# 与 gap_results 主表一致: (ds_dir, model)
CELLS = [
    ("data/beir/scifact",     "e5"),
    ("data/beir/scifact",     "bge"),
    ("data/beir/scifact",     "mini"),
    ("data/beir/scifact",     "bgem3"),
    ("data/beir/scifact",     "qwen3"),
    ("data/beir_big/arguana", "e5"),
    ("data/beir_big/arguana", "bge"),
    ("data/beir_big/arguana", "mini"),
    ("data/beir_big/arguana", "bgem3"),
    ("data/beir_big/arguana", "qwen3"),
    ("data/nq_sub200k",       "e5"),
    ("data/nq_sub200k",       "bge"),
    ("data/nq_sub200k",       "mini"),
    ("data/nq_sub200k",       "bgem3"),
    ("data/nq_sub200k",       "qwen3"),
]

BASELINES = ["pq", "opq", "itq", "scalar", "tq"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="限定 cell_key 子串 (如 e5@scifact@keep0.500), 默认全跑")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # 断点续跑: 读已有输出, 跳过已完成的 cell
    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
    for ds_dir, model in CELLS:
        dataset = os.path.basename(ds_dir)
        emb, qemb, qids, qrels, ids = load(ds_dir, model)
        d = emb.shape[1]
        imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=1.0)
        for keep in KEEPS:
            m = max(1, int(d * keep))
            if m % 8 != 0:
                m = (m // 8) * 8
            key = f"{model}@{dataset}@keep{keep:.3f}"
            if args.cells and not any(c in key for c in args.cells):
                continue
            if key in out:
                print("SKIP", key, flush=True)
                continue
            perq = {}
            perq["reco"] = run_reco(emb, qemb, imp, qids, qrels, ids, m, B, return_arrays=True)
            for meth in BASELINES:
                r = run_faiss_quant(ds_dir, model, meth, emb, qemb, qids, qrels, ids, m,
                                    return_arrays=True)
                if r is not None:
                    perq[meth] = r
            means = {mm: float(np.mean(a)) for mm, a in perq.items()}
            best_base = max([mm for mm in means if mm != "reco"], key=lambda mm: means[mm])
            reco_a = perq["reco"]; base_a = perq[best_base]
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
                "sig_0001": p <= 0.001,
                "sig_001": p <= 0.01,
                "sig_005": p <= 0.05,
            }
            print(key, out[key], flush=True)
            json.dump(out, open(args.out, "w"), indent=1)  # 逐格保存, 防断点丢失
    json.dump(out, open(args.out, "w"), indent=1)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
