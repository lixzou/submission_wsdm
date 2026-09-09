"""诚实 train 校准下, 完整管线 vs baselines (Table 1 核心对比)。
RECO口径=谱引导旋转+量化; 对比 PQ/OPQ/ITQ/scalar/TQ (prefix截断+量化)。
看诚实协议下 RECO 是否仍赢 baselines。
"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from merge_pca_retrieval import pca_directions
from rabitq_py import random_orthogonal
from sota_pipeline_check import fit_calib_levels, quant_recall
from main_table import run_reco, run_faiss_quant
import faiss; faiss.omp_set_num_threads(4)

for ds, m in [('data/beir/scifact','e5'), ('data/beir/scifact','bge'), ('data/beir/scifact','mini'),
              ('data/beir/fiqa','e5'), ('data/beir/fiqa','bge'), ('data/beir/fiqa','mini')]:
    emb, qemb, qids, qrels, ids = load(ds, m)
    d = emb.shape[1]
    tq = json.load(open(f'{ds}/train_qids.json'))
    tr = json.load(open(f'{ds}/train_qrels.json'))
    qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)
    imp = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
    for kk in [0.5, 0.25, 0.125]:
        mm = (max(1,int(d*kk))//8)*8
        B = {0.5:2, 0.25:4, 0.125:8}[kk]
        # RECO (诚实, 谱引导)
        r_reco = run_reco(emb, qemb, imp, qids, qrels, ids, mm, B, calib_qids=tq, calib_qrels=tr)
        # baselines
        cells = {'reco': r_reco}
        for meth in ['pq','opq','scalar','tq']:
            try:
                r = run_faiss_quant(ds, m, meth, emb, qemb, qids, qrels, ids, mm)
                if r is not None: cells[meth] = r
            except: pass
        best_base = max(v for k,v in cells.items() if k!='reco')
        win = cells['reco'] > best_base
        print(f'{ds.split("/")[-1]:9s} {m:5s} k={kk}: reco={r_reco:.4f} best_base={best_base:.4f} {"WIN" if win else "LOSE"}', flush=True)
