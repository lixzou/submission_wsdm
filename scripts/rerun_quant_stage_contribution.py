#!/usr/bin/env python3
"""3.3 量化阶段的贡献: 加上校准网格 (RECO) vs 去掉校准 (uniform grid, 同预算)。
协议与 main_table.run_reco 一致: weighted PCA 截断 + 旋转 seed=0 + B=2 量化 + 重归一化。
校准 (margin + grid) 用 train 查询, 评估在 test。输出 results/runs/quant_stage_{model}.json
"""
import argparse, json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from merge_pca_retrieval import pca_directions
from rabitq_py import random_orthogonal
from sota_pipeline_check import fit_calib_levels, quant_recall

B = 2

def run(ds_dir, model, keeps):
    emb, qemb, qids, qrels, ids = load(ds_dir, model)
    d = emb.shape[1]
    train_qids = json.load(open(f"{ds_dir}/train_qids.json"))
    train_qrels = json.load(open(f"{ds_dir}/train_qrels.json"))
    qemb_train = np.load(f"{ds_dir}/qemb_train_{model}.npy").astype(np.float32)
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": ds_dir, "model": model, "d": d, "reference": round(ref, 4)}
    imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)
    for keep in keeps:
        m = max(1, int(d * keep))
        if m % 8 != 0:
            m = (m // 8) * 8
        ww = imp / (imp.max() + 1e-12) + 1e-6
        V = pca_directions(emb, m, w=ww)
        emb_proj = emb @ V.T; q_proj = qemb @ V.T
        P = random_orthogonal(m, seed=0)
        rot = (emb_proj @ P).astype(np.float64)
        # 加上 3.3: 校准网格 (fit on train)
        lv_calib = fit_calib_levels(rot, train_qids, train_qrels, ids, B)
        r_calib = quant_recall(q_proj, emb_proj, np.arange(m), B, lv_calib, qids, qrels, ids)
        # 去掉 3.3: uniform grid (逐维等距, 同预算)
        lv_unif = np.array([np.linspace(rot[:, j].min(), rot[:, j].max(), 2**B)
                            for j in range(m)])
        r_unif = quant_recall(q_proj, emb_proj, np.arange(m), B, lv_unif, qids, qrels, ids)
        res[f"keep={keep:.3f}"] = {"calib_grid": round(r_calib, 4), "uniform_grid": round(r_unif, 4)}
        print(f"keep={keep} m={m}: calib={r_calib:.4f} uniform={r_unif:.4f} "
              f"delta={r_calib - r_unif:+.4f}", flush=True)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keeps", nargs="+", type=float, default=[0.5, 0.75, 0.875])
    a = ap.parse_args()
    res = run(a.ds_dir, a.model, a.keeps)
    json.dump(res, open(a.out, "w"), indent=1)
    print("SAVED", a.out)
