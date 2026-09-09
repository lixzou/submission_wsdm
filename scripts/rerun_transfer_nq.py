#!/usr/bin/env python3
"""跨域迁移实验: scifact train 校准 -> NQ-200k test 评估 (零目标域标注)。

优化: SVD 一次共享给 3 个 keep; 串行跑 5 模型, 避免 CPU 超订阅。

三列对比 (纯 Stage-1 截断, 无量化):
  prefix      : 前 m 坐标 (无校准 baseline)
  cross-domain: scifact train 校准的 ranking-importance weighted PCA
  in-domain   : NQ heldout train 校准 (上界)

用法: python rerun_transfer_nq.py
"""
import json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance

NQ_DS = "/ssd1/zoulixin/tencent_previous_compression/experiments/data/nq_sub200k"
SCIFACT_DS = "/ssd1/zoulixin/tencent_previous_compression/experiments/data/beir/scifact"
OUT_DIR = "/ssd1/zoulixin/tencent_previous_compression/experiments/results/runs"
N_CAL = 800
SEED = 0
KEEPS = [0.5, 0.75, 0.875]
MODELS = ["e5", "bge", "mini", "bgem3", "qwen3"]


def metrics(rank, qids, qrels, ids):
    r, n = recall_at_k(rank, qids, qrels, ids, k=10)
    return float(r), float(n)


def fit_proj(emb, w, keeps, d):
    """协方差矩阵特征分解 (d×d, 快): Xw.T Xw 的 eigh 给出主方向, 返回 {keep: V(m,d)}.
    eigh 返回升序列向量, 主方向在最后 -> 翻转列再转置为行向量."""
    mu = emb.mean(0)
    X = emb - mu
    Xw = X * np.sqrt(w)[None, :]
    C = Xw.T @ Xw  # d×d 协方差 (加权)
    _, evecs = np.linalg.eigh(C)  # 升序, evecs[:, i] 是第 i 特征向量
    Vt = evecs[:, ::-1].T  # 降序行向量, Vt[i] 是第 i 主方向
    return {k: Vt[:max(1, int(d * k))] for k in keeps}


def run_model(mname):
    emb_nq, qemb_nq, qids_nq, qrels_nq, ids_nq = load(NQ_DS, mname)
    d = emb_nq.shape[1]
    emb_sc, _, _, _, ids_sc = load(SCIFACT_DS, mname)
    _, _, _, _, _ = emb_sc, None, None, None, ids_sc
    train_qids_sc = json.load(open(f"{SCIFACT_DS}/train_qids.json"))
    train_qrels_sc = json.load(open(f"{SCIFACT_DS}/train_qrels.json"))
    qemb_train_sc = np.load(f"{SCIFACT_DS}/qemb_train_{mname}.npy").astype(np.float32)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(qids_nq))
    train_idx = sorted(perm[:N_CAL].tolist())
    test_idx = sorted(perm[N_CAL:].tolist())
    train_qids_nq = [qids_nq[i] for i in train_idx]
    test_qids_nq = [qids_nq[i] for i in test_idx]
    train_qrels_nq = {q: qrels_nq[q] for q in train_qids_nq}
    test_qrels_nq = {q: qrels_nq[q] for q in test_qids_nq}
    qemb_train_nq = qemb_nq[np.array(train_idx)]
    qemb_test_nq = qemb_nq[np.array(test_idx)]

    imp_cross = margin_importance(emb_sc, qemb_train_sc, train_qids_sc, train_qrels_sc,
                                  ids_sc, calib_frac=1.0)
    imp_in = margin_importance(emb_nq, qemb_train_nq, train_qids_nq, train_qrels_nq,
                               ids_nq, calib_frac=1.0)
    Vc = fit_proj(emb_nq, imp_cross / (imp_cross.max() + 1e-6), KEEPS, d)
    Vi = fit_proj(emb_nq, imp_in / (imp_in.max() + 1e-6), KEEPS, d)

    res = {"model": mname, "d": d, "n_train": N_CAL, "n_test": len(test_qids_nq),
           "source": "scifact-train", "target": "nq_sub200k", "seed": SEED}
    for keep in KEEPS:
        m = max(1, int(d * keep))
        # prefix (前 m 坐标)
        pr = metrics(brute_rank(qemb_test_nq[:, :m], emb_nq[:, :m]), test_qids_nq, test_qrels_nq, ids_nq)
        # cross
        qc = qemb_test_nq @ Vc[keep].T; ec = emb_nq @ Vc[keep].T
        cr = metrics(brute_rank(qc.astype(np.float32), ec.astype(np.float32)), test_qids_nq, test_qrels_nq, ids_nq)
        # in
        qi = qemb_test_nq @ Vi[keep].T; ei = emb_nq @ Vi[keep].T
        ir = metrics(brute_rank(qi.astype(np.float32), ei.astype(np.float32)), test_qids_nq, test_qrels_nq, ids_nq)
        res["keep=%.3f" % keep] = {
            "prefix": {"recall": round(pr[0], 4), "ndcg": round(pr[1], 4)},
            "cross": {"recall": round(cr[0], 4), "ndcg": round(cr[1], 4)},
            "in": {"recall": round(ir[0], 4), "ndcg": round(ir[1], 4)},
        }
        print(f"  {mname} keep={keep} m={m} prefix={pr[0]:.4f} cross={cr[0]:.4f} in={ir[0]:.4f}", flush=True)
    json.dump(res, open(f"{OUT_DIR}/transfer_nq_{mname}.json", "w"), indent=1)
    print("SAVED", f"{OUT_DIR}/transfer_nq_{mname}.json", flush=True)


if __name__ == "__main__":
    for m in MODELS:
        print("=== model", m, "===", flush=True)
        run_model(m)
    print("ALL DONE", flush=True)
