#!/usr/bin/env python3
"""统一算 3 数据集 × 3 模型 × b̄∈{1,2,4} 的逐坐标熵编码字节节省（全语料）。
协议: 随机正交旋转 seed=0 + quantize_Bbit + 逐维 Shannon 熵; saving = 1 - ΣH_j/(d·b̄)。
与 paper tab:c3 数值一致 (fiqa e5 b̄=1 -> 49.0)。输出 results/runs/entropy_all.json
"""
import json, os, sys, time
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_truncation import load
from rabitq_py import random_orthogonal, quantize_Bbit

DATASETS = ["scifact", "fiqa", "nfcorpus"]
MODELS = ["e5", "bge", "mini"]
BS = [1, 2, 4]

def code_entropy_saving(emb, B):
    d = emb.shape[1]
    P = random_orthogonal(d, seed=0)
    rot = (emb @ P).astype(np.float32)
    codes, _ = quantize_Bbit(rot, B)
    H = 0.0
    for j in range(d):
        _, cnt = np.unique(codes[:, j], return_counts=True)
        p = cnt / cnt.sum()
        H += float(-(p * np.log2(p)).sum())
    return round(1.0 - H / (d * B), 4)

if __name__ == "__main__":
    res = {}
    for ds in DATASETS:
        res[ds] = {}
        for m in MODELS:
            emb, *_ = load(f"data/beir/{ds}", m)
            res[ds][m] = {}
            for B in BS:
                t0 = time.time()
                res[ds][m][f"b{B}"] = code_entropy_saving(emb, B)
                print(f"{ds} {m} b{B}: {res[ds][m][f'b{B}']*100:.1f}% ({time.time()-t0:.1f}s)", flush=True)
    json.dump(res, open("results/runs/entropy_all.json", "w"), indent=1)
    print("SAVED results/runs/entropy_all.json")
