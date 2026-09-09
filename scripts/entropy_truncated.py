#!/usr/bin/env python3
"""截断协议下的熵编码字节节省 (新预算协议 32x/21x/18x)。

协议: RECO relevance-calibrated truncation (加权 PCA, alpha=1) 保留 m 维,
b̄=2 量化, 在保留坐标上算逐维 Shannon 熵; saving = 1 - Σ_{j<=m} H_j/(m*2).
预算: m=0.5d -> 32x, 0.75d -> 21x, 0.875d -> 18x (vs float32)。
输出 results/runs/entropy_truncated.json
"""
import json, os, sys, time
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load, margin_importance
from merge_pca_retrieval import pca_directions
from rabitq_py import quantize_Bbit

DATASETS = ["scifact", "fiqa", "nfcorpus"]
MODELS = ["e5", "bge", "mini"]
KEEPS = [0.500, 0.750, 0.875]   # -> 32x / 21x / 18x
B = 2


def _uniform_codes(proj, B):
    """均匀网格量化（quantize_Bbit 崩溃时的 fallback）: 每维按 [min,max] 线性分 2^B 格。"""
    N, m = proj.shape
    L = 2 ** B
    codes = np.zeros((N, m), np.int32)
    for j in range(m):
        lo, hi = proj[:, j].min(), proj[:, j].max()
        if hi - lo < 1e-12:
            codes[:, j] = 0
        else:
            codes[:, j] = np.clip(((proj[:, j] - lo) / (hi - lo) * L).astype(np.int32), 0, L - 1)
    return codes


def _code_entropy(codes):
    H = 0.0
    for j in range(codes.shape[1]):
        _, cnt = np.unique(codes[:, j], return_counts=True)
        p = cnt / cnt.sum()
        H += float(-(p * np.log2(p)).sum())
    return H


def truncated_saving(emb, qemb_train, train_qids, train_qrels, ids, keep):
    d = emb.shape[1]
    m = max(1, int(d * keep))
    if m % 8 != 0:
        m = (m // 8) * 8
    imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)
    ww = imp / (imp.max() + 1e-12) + 1e-6
    V = pca_directions(emb, m, w=ww)
    proj = (emb @ V.T).astype(np.float32)
    try:
        codes, _ = quantize_Bbit(proj, B)
    except Exception as e:
        print(f"    [quantize_Bbit fallback: {e}]", flush=True)
        codes = _uniform_codes(proj, B)
    H = _code_entropy(codes)
    return round(1.0 - H / (m * B), 4)


if __name__ == "__main__":
    res = {}
    for ds in DATASETS:
        res[ds] = {}
        train_qids = json.load(open(f"data/beir/{ds}/train_qids.json"))
        train_qrels = json.load(open(f"data/beir/{ds}/train_qrels.json"))
        for m_ in MODELS:
            emb_m, qemb_m, qids_m, qrels_m, ids_m = load(f"data/beir/{ds}", m_)
            qt = np.load(f"data/beir/{ds}/qemb_train_{m_}.npy").astype(np.float32)
            res[ds][m_] = {}
            for keep in KEEPS:
                t0 = time.time()
                sv = truncated_saving(emb_m, qt, train_qids, train_qrels, ids_m, keep)
                res[ds][m_][f"keep{keep}"] = sv
                print(f"{ds} {m_} keep{keep}: {sv*100:.1f}% ({time.time()-t0:.1f}s)", flush=True)
    json.dump(res, open("results/runs/entropy_truncated.json", "w"), indent=1)
    print("SAVED results/runs/entropy_truncated.json")
