"""快速扫描可用数据集的加权PCA增益 (train校准, 未量化, keep=0.5)
找出加权PCA真正有用的数据集。"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
MJ = '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts'
sys.path.insert(0, MJ)
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from merge_pca_retrieval import pca_directions
import faiss; faiss.omp_set_num_threads(4)

datasets = ['data/beir/scifact', 'data/beir/fiqa', 'data/beir/nfcorpus']
models = ['e5', 'bge', 'mini']

for ds in datasets:
    dsname = ds.split('/')[-1]
    for m in models:
        try:
            emb, qemb, qids, qrels, ids = load(ds, m)
            d = emb.shape[1]
            tq = json.load(open(f'{ds}/train_qids.json'))
            tr = json.load(open(f'{ds}/train_qrels.json'))
            qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)
            imp = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
            kk = 0.5
            m_ = max(1, int(d*kk)); m_ = (m_//8)*8
            V = pca_directions(emb, m_)
            r_pca = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]
            ww = imp/(imp.max()+1e-12)+1e-6
            Vw = pca_directions(emb, m_, w=ww)
            r_wpca = recall_at_k(brute_rank(qemb @ Vw.T, emb @ Vw.T), qids, qrels, ids)[0]
            print(f'{dsname:12s} {m:5s}: pca={r_pca:.4f} wpca={r_wpca:.4f} diff={(r_wpca-r_pca)*1000:+.1f}pp')
        except Exception as e:
            print(f'{dsname:12s} {m:5s}: ERR {str(e)[:40]}')
