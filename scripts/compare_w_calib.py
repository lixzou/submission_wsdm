"""对比: 用 test校准w 和 train校准w 做加权PCA, 都在 test 上评估。
验证: 正文的加权增益是否纯来自 test 校准泄漏。"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
MJ = '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts'
sys.path.insert(0, MJ)
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

# w: train校准 vs test校准
w_train = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
w_test = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=1.0)

V = pca_directions(emb, mm)
r_pca = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]
print(f'plain PCA: {r_pca:.4f}')

for name, w in [('w_train(诚实)', w_train), ('w_test(泄漏)', w_test)]:
    ww = w/(w.max()+1e-12)+1e-6
    Vw = pca_directions(emb, mm, w=ww)
    r = recall_at_k(brute_rank(qemb @ Vw.T, emb @ Vw.T), qids, qrels, ids)[0]
    print(f'{name}: {r:.4f} ({(r-r_pca)*1000:+.1f}pp)')
