"""决定性实验: 区分 'w精度不够(采样噪声)' vs 'train/test分布差异(泛化)'。
用 train 查询: ①算 w ②评估加权PCA ③也在 train 查询上评估。
"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from merge_pca_retrieval import pca_directions
import faiss; faiss.omp_set_num_threads(4)

ds, m = 'data/beir/scifact', 'e5'
emb, qemb, qids, qrels, ids = load(ds, m)
d = emb.shape[1]
tq = json.load(open(f'{ds}/train_qids.json'))
tr = json.load(open(f'{ds}/train_qrels.json'))
qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)
kk = 0.5; mm = (max(1,int(d*kk))//8)*8

def rec(V, qe, qids_, qrels_):
    return recall_at_k(brute_rank(qe @ V.T, qe @ V.T*0+1 if False else qe), qids_, qrels_, ids)[0] if False else \
        recall_at_k(brute_rank((qemb if qe is None else qe)[:, :d] @ V.T, emb[:, :d] @ V.T), qids, qrels, ids)[0]

# 正确实现: 用 train 查询评估
def rec_train(V):
    return recall_at_k(brute_rank(qt @ V.T, emb @ V.T), tq, tr, ids)[0]
def rec_test(V):
    return recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]

# plain PCA
V = pca_directions(emb, mm)
print(f'plain PCA: train评估={rec_train(V):.4f} test评估={rec_test(V):.4f}')

# train校准 w 加权
for seed in [0, 1, 2]:
    imp = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0, seed=seed)
    ww = imp/(imp.max()+1e-12)+1e-6
    Vw = pca_directions(emb, mm, w=ww)
    print(f'train-w(seed{seed}) 加权: train评估={rec_train(Vw):.4f} test评估={rec_test(Vw):.4f}')

# test校准 w 加权 (泄漏)
imp_t = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=1.0)
ww = imp_t/(imp_t.max()+1e-12)+1e-6
Vwt = pca_directions(emb, mm, w=ww)
print(f'test-w(泄漏) 加权: train评估={rec_train(Vwt):.4f} test评估={rec_test(Vwt):.4f}')
