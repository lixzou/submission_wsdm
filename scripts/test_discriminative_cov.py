"""真正的检索监督协方差: 相关对 vs 不相关对 的二阶统计差异。
Σ_disc = E[相关对 x_q x_d^T] - E[不相关对 x_q x_d^T]
取主方向 = 真正编码检索信号的变换。替代 w 缩放协方差。
"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
MJ = '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts'
sys.path.insert(0, MJ)
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from merge_pca_retrieval import pca_directions
import faiss; faiss.omp_set_num_threads(4)

def discriminative_directions(emb, qemb_train, train_qids, train_qrels, ids, m, n_pairs=100000, seed=0):
    """构造判别协方差并取主方向。"""
    rng = np.random.default_rng(seed)
    qidx = {q: i for i, q in enumerate(train_qids)}
    idset = {s: i for i, s in enumerate(ids)}
    pos_outer = np.zeros((emb.shape[1], emb.shape[1]))
    neg_outer = np.zeros((emb.shape[1], emb.shape[1]))
    n_pos = n_neg = 0
    for qid in train_qids:
        qi = qidx[qid]
        rel_docs = sorted(idset[d] for d in train_qrels[qid] if d in idset)
        if not rel_docs: continue
        per_q = max(1, n_pairs // len(train_qids))
        for _ in range(min(100, per_q)):
            di = rel_docs[rng.integers(len(rel_docs))]
            x = qemb_train[qi][:,None] * emb[di][None,:]  # 外积 q*d
            pos_outer += x; n_pos += 1
            ni = rng.integers(len(emb))
            x = qemb_train[qi][:,None] * emb[ni][None,:]
            neg_outer += x; n_neg += 1
    if n_pos: pos_outer /= n_pos
    if n_neg: neg_outer /= n_neg
    Sigma_disc = pos_outer - neg_outer  # 判别矩阵 (非对称)
    # 用对称部分: (S + S^T)/2
    S = (Sigma_disc + Sigma_disc.T) / 2
    # 取前 m 个特征向量
    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(-np.abs(evals))[:m]
    return evecs[:, order].T  # (m, d)

for ds, m in [('data/beir/scifact','e5'), ('data/beir/scifact','bge'), ('data/beir/scifact','mini'),
              ('data/beir/fiqa','e5'), ('data/beir/nfcorpus','e5')]:
    emb, qemb, qids, qrels, ids = load(ds, m)
    d = emb.shape[1]
    tq = json.load(open(f'{ds}/train_qids.json'))
    tr = json.load(open(f'{ds}/train_qrels.json'))
    qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)
    kk = 0.5; mm = (max(1,int(d*kk))//8)*8
    V = pca_directions(emb, mm)
    r_pca = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]
    Vd = discriminative_directions(emb, qt, tq, tr, ids, mm)
    r_disc = recall_at_k(brute_rank(qemb @ Vd.T, emb @ Vd.T), qids, qrels, ids)[0]
    print(f'{ds.split("/")[-1]:9s} {m:5s}: plainPCA={r_pca:.4f} | 判别协方差={r_disc:.4f} ({(r_disc-r_pca)*1000:+.0f}pp)')
