"""测试更强的加权注入: 更高幂次 + 子空间选择。
正文 §3.2 声称加权协方差注入检索信号, 当前 α=1.0 太弱。
"""
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
imp = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
kk = 0.5; mm = (max(1,int(d*kk))//8)*8

print(f'scifact e5 keep=0.5 (d={d}, m={mm}):')
V = pca_directions(emb, mm)
r_pca = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]
print(f'  plain PCA: {r_pca:.4f}')

# 各种加权强度
for alpha in [1.0, 2.0, 3.0, 5.0, 8.0]:
    ww = (imp/(imp.max()+1e-12))**alpha + 1e-6
    Vw = pca_directions(emb, mm, w=ww)
    r = recall_at_k(brute_rank(qemb @ Vw.T, emb @ Vw.T), qids, qrels, ids)[0]
    print(f'  wpca α={alpha}: {r:.4f} ({(r-r_pca)*1000:+.1f}pp)')

# 子空间选择: 先选 top-K 高w维, 再 PCA
for Kmult in [1, 2, 3, 4]:
    K = min(d, int(mm*Kmult))
    sel = np.argsort(-imp)[:K]
    emb_sel, qemb_sel = emb[:, sel], qemb[:, sel]
    Vsub = pca_directions(emb_sel, mm)
    r = recall_at_k(brute_rank(qemb_sel @ Vsub.T, emb_sel @ Vsub.T), qids, qrels, ids)[0]
    print(f'  select-K={K} PCA: {r:.4f} ({(r-r_pca)*1000:+.1f}pp)')
