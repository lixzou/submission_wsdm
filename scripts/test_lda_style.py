"""LDA风格检索引导: 最大化 类间/类内 距离比。
对每维构造 相关文档均值 vs 不相关文档均值 的判别方向, 再做标准PCA/投影。
这是比"缩放协方差"更接近检索目标的方法。"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, recall_at_k, brute_rank
from merge_pca_retrieval import pca_directions
import faiss; faiss.omp_set_num_threads(4)

def lda_directions(emb, qemb_train, train_qids, train_qrels, ids, m, seed=0):
    """构造 LDA 判别矩阵: Sw^{-1} Sb 的主方向.
    Sb = 相关对均值外积 - 全均值外积 (类间)
    近似: 用 相关-不相关 均值向量构造判别方向。"""
    rng = np.random.default_rng(seed)
    qidx = {q: i for i, q in enumerate(train_qids)}
    idset = {s: i for i, s in enumerate(ids)}
    d = emb.shape[1]
    # 收集相关对和不相关对
    pos_q, pos_d, neg_q, neg_d = [], [], [], []
    for qid in train_qids:
        qi = qidx[qid]
        rel = sorted(idset[dd] for dd in train_qrels[qid] if dd in idset)
        if not rel: continue
        for _ in range(20):
            pos_q.append(qi); pos_d.append(rel[rng.integers(len(rel))])
            neg_q.append(qi); neg_d.append(rng.integers(len(emb)))
    pos_q=np.array(pos_q); pos_d=np.array(pos_d)
    neg_q=np.array(neg_q); neg_d=np.array(neg_d)
    # 类间散度: 相关对的均值外积 - 不相关对的均值外积
    Sb = qemb_train[pos_q].T @ emb[pos_d]/len(pos_q) - qemb_train[neg_q].T @ emb[neg_d]/len(neg_q)
    Sb = (Sb+Sb.T)/2
    # 类内散度: 全语料协方差 (Sw)
    mu = emb.mean(0); Xc = emb - mu
    Sw = Xc.T @ Xc / len(emb)
    Sw += np.eye(d)*1e-6  # 正则
    # 广义特征分解 Sw^{-1} Sb
    try:
        import scipy.linalg as sla
        evals, evecs = sla.eig(Sb, Sw)
        evecs = evecs.real
        order = np.argsort(-np.abs(evals.real))[:m]
        return evecs[:, order].T
    except Exception as e:
        # 退化为 Sb 特征分解
        evals, evecs = np.linalg.eigh(Sb)
        order = np.argsort(-np.abs(evals))[:m]
        return evecs[:, order].T

for ds, m in [('data/beir/scifact','e5'), ('data/beir/scifact','bge'), ('data/beir/scifact','mini'),
              ('data/beir/fiqa','e5'), ('data/beir/fiqa','bge'), ('data/beir/fiqa','mini'),
              ('data/beir/nfcorpus','e5')]:
    emb, qemb, qids, qrels, ids = load(ds, m)
    d = emb.shape[1]
    tq = json.load(open(f'{ds}/train_qids.json'))
    tr = json.load(open(f'{ds}/train_qrels.json'))
    qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)
    kk = 0.5; mm = (max(1,int(d*kk))//8)*8
    V = pca_directions(emb, mm)
    r_pca = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]
    Vl = lda_directions(emb, qt, tq, tr, ids, mm)
    r_lda = recall_at_k(brute_rank(qemb @ Vl.T, emb @ Vl.T), qids, qrels, ids)[0]
    print(f'{ds.split("/")[-1]:9s} {m:5s}: plainPCA={r_pca:.4f} | LDA={r_lda:.4f} ({(r_lda-r_pca)*1000:+.0f}pp)', flush=True)
