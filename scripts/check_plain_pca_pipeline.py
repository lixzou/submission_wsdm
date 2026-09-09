"""关键: plain PCA + 校准量化 vs baselines (诚实 train 校准)。
若 plain PCA 管线赢 baselines, 说明贡献在量化; 若输, 论文无立足点。
"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, recall_at_k, brute_rank
from merge_pca_retrieval import pca_directions
from rabitq_py import random_orthogonal
from sota_pipeline_check import fit_calib_levels, quant_recall
from main_table import run_faiss_quant
import faiss; faiss.omp_set_num_threads(4)

for ds, m in [('data/beir/scifact','e5'), ('data/beir/scifact','bge'), ('data/beir/scifact','mini'),
              ('data/beir/fiqa','e5'), ('data/beir/fiqa','bge')]:
    emb, qemb, qids, qrels, ids = load(ds, m)
    d = emb.shape[1]
    tq = json.load(open(f'{ds}/train_qids.json'))
    tr = json.load(open(f'{ds}/train_qrels.json'))
    kk = 0.5; mm = (max(1,int(d*kk))//8)*8
    B = 2
    P = random_orthogonal(mm, seed=0)
    # plain PCA + 校准量化
    V = pca_directions(emb, mm)
    ep, qp = emb @ V.T, qemb @ V.T
    rot = (ep @ P).astype(np.float64)
    lv = fit_calib_levels(rot, tq, tr, ids, B)
    r_pca_q = quant_recall(qp, ep, np.arange(mm), B, lv, qids, qrels, ids)
    # baselines
    cells = {}
    for meth in ['pq','opq','scalar','tq']:
        try:
            r = run_faiss_quant(ds, m, meth, emb, qemb, qids, qrels, ids, mm)
            if r is not None: cells[meth] = round(r,4)
        except: pass
    best_base = max(cells.values())
    print(f'{ds.split("/")[-1]:9s} {m:5s}: plainPCA+量化={r_pca_q:.4f} | baselines {cells} best={best_base:.4f} {"WIN" if r_pca_q>best_base else "LOSE"}', flush=True)
