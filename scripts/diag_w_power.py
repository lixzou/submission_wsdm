"""诊断 w 预测力: perm(w排序截断) vs plain PCA vs 加权PCA, 未量化, keep=0.5
如果 perm << pca, 说明 w 在原始空间弱, 加权PCA不该有效。
"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
MJ = '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts'
sys.path.insert(0, MJ)
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from merge_pca_retrieval import pca_directions
import faiss; faiss.omp_set_num_threads(4)

for ds in ['data/beir/scifact', 'data/beir/fiqa', 'data/beir/nfcorpus']:
    for m in ['e5', 'bge', 'mini']:
        try:
            emb, qemb, qids, qrels, ids = load(ds, m)
            d = emb.shape[1]
            tq = json.load(open(f'{ds}/train_qids.json'))
            tr = json.load(open(f'{ds}/train_qrels.json'))
            qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)
            imp = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
            kk = 0.5; mm = (max(1,int(d*kk))//8)*8
            # perm: w 排序截断
            sel = np.argsort(-imp)[:mm]
            r_perm = recall_at_k(brute_rank(qemb[:,sel], emb[:,sel]), qids, qrels, ids)[0]
            # plain PCA
            V = pca_directions(emb, mm)
            r_pca = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]
            # weighted PCA
            ww = imp/(imp.max()+1e-12)+1e-6
            Vw = pca_directions(emb, mm, w=ww)
            r_wpca = recall_at_k(brute_rank(qemb @ Vw.T, emb @ Vw.T), qids, qrels, ids)[0]
            print(f'{ds.split("/")[-1]:10s} {m:5s}: perm={r_perm:.4f} pca={r_pca:.4f} wpca={r_wpca:.4f}')
        except Exception as e:
            print(f'{ds} {m}: ERR {str(e)[:40]}')
