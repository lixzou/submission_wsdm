"""测试 w 的温和变换: 问题在极端值主导, 而非 w 本身无效。
对 w 做 rank/log/裁剪变换后加权PCA, 看诚实协议下是否增益。"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
MJ = '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts'
sys.path.insert(0, MJ)
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from merge_pca_retrieval import pca_directions
import faiss; faiss.omp_set_num_threads(4)

def pca_rank(emb, mm, w):
    """用 w 的 rank 做加权 (温和, 不受极端值影响)"""
    r = np.argsort(np.argsort(w))  # rank 0..d-1
    ww = (r / r.max() + 1e-6)
    return pca_directions(emb, mm, w=ww)

for ds, m in [('data/beir/scifact','e5'), ('data/beir/scifact','bge'), ('data/beir/scifact','mini'),
              ('data/beir/fiqa','e5'), ('data/beir/nfcorpus','e5')]:
    emb, qemb, qids, qrels, ids = load(ds, m)
    d = emb.shape[1]
    tq = json.load(open(f'{ds}/train_qids.json'))
    tr = json.load(open(f'{ds}/train_qrels.json'))
    qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)
    w = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
    kk = 0.5; mm = (max(1,int(d*kk))//8)*8
    V = pca_directions(emb, mm)
    r_pca = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]
    # 原始线性加权
    ww = w/(w.max()+1e-12)+1e-6
    r_lin = recall_at_k(brute_rank(qemb @ pca_directions(emb,mm,w=ww).T, emb @ pca_directions(emb,mm,w=ww).T), qids, qrels, ids)[0]
    # rank 加权
    Vr = pca_rank(emb, mm, w)
    r_rank = recall_at_k(brute_rank(qemb @ Vr.T, emb @ Vr.T), qids, qrels, ids)[0]
    # log 加权
    wlog = np.log1p(w); wlog = wlog/(wlog.max()+1e-12)+1e-6
    r_log = recall_at_k(brute_rank(qemb @ pca_directions(emb,mm,w=wlog).T, emb @ pca_directions(emb,mm,w=wlog).T), qids, qrels, ids)[0]
    print(f'{ds.split("/")[-1]:9s} {m:5s}: pca={r_pca:.4f} | 线性={r_lin:.4f}({(r_lin-r_pca)*1000:+.0f}) | rank={r_rank:.4f}({(r_rank-r_pca)*1000:+.0f}) | log={r_log:.4f}({(r_log-r_pca)*1000:+.0f})')
