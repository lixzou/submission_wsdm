"""真正的贡献候选: 校准量化的价值 (judged-fit Lloyd-Max vs 重建最优 vs 均匀)。
固定变换=plain PCA, 只换量化网格, 诚实 train 校准。"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, recall_at_k, brute_rank
from merge_pca_retrieval import pca_directions
from rabitq_py import random_orthogonal
from sota_pipeline_check import fit_calib_levels, quant_recall, lloyd_max_from_samples
import faiss; faiss.omp_set_num_threads(4)

def quant_recall_grid(qemb, emb, sel, B, levels, qids, qrels, ids):
    m = len(sel)
    P = random_orthogonal(m, seed=0)
    rot = (emb[:, sel] @ P).astype(np.float64)
    codes = np.zeros((len(rot), m), np.int32)
    for j in range(m):
        codes[:, j] = np.argmin(np.abs(rot[:, j, None] - levels[j][None, :]), axis=1)
    recon = levels[np.arange(m)[None, :], codes]
    o = (P @ recon.T).T
    o = o/(np.linalg.norm(o,axis=1,keepdims=True)+1e-12)
    return recall_at_k(brute_rank(qemb[:, sel].astype(np.float32), o.astype(np.float32)), qids, qrels, ids)[0]

def recon_levels(rot_emb, B):
    """重建最优网格: 全语料 Lloyd-Max (TurboQuant口径)。"""
    K = 2**B
    levels = np.zeros((rot_emb.shape[1], K))
    for j in range(rot_emb.shape[1]):
        levels[j] = lloyd_max_from_samples(rot_emb[:, j], K)
    return levels

for ds, m in [('data/beir/scifact','e5'), ('data/beir/scifact','bge'), ('data/beir/scifact','mini'),
              ('data/beir/fiqa','e5'), ('data/beir/fiqa','bge'), ('data/beir/nfcorpus','e5')]:
    emb, qemb, qids, qrels, ids = load(ds, m)
    d = emb.shape[1]
    tq = json.load(open(f'{ds}/train_qids.json'))
    tr = json.load(open(f'{ds}/train_qrels.json'))
    kk = 0.5; mm = (max(1,int(d*kk))//8)*8
    B = 2
    V = pca_directions(emb, mm)
    ep, qp = emb @ V.T, qemb @ V.T
    P = random_orthogonal(mm, seed=0)
    rot = (ep @ P).astype(np.float64)
    # 三种网格
    lv_judged = fit_calib_levels(rot, tq, tr, ids, B)  # judged-fit (RECO)
    lv_recon = recon_levels(rot, B)  # 重建最优 (TurboQuant)
    # 均匀网格
    half = (2**B-1)/2.0
    lv_uniform = np.array([np.linspace(-half, half, 2**B) for _ in range(mm)])
    r_judged = quant_recall_grid(qp, ep, np.arange(mm), B, lv_judged, qids, qrels, ids)
    r_recon = quant_recall_grid(qp, ep, np.arange(mm), B, lv_recon, qids, qrels, ids)
    r_unif = quant_recall_grid(qp, ep, np.arange(mm), B, lv_uniform, qids, qrels, ids)
    print(f'{ds.split("/")[-1]:9s} {m:5s}: judged={r_judged:.4f} recon={r_recon:.4f} uniform={r_unif:.4f} | judged-recon={(r_judged-r_recon)*1000:+.0f}pp', flush=True)
