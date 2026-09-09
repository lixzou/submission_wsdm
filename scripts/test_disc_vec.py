"""向量化判别协方差: 相关对 vs 不相关对的均值外积差。快速。"""
import json, numpy as np, sys, os
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, recall_at_k, brute_rank
from merge_pca_retrieval import pca_directions
import faiss; faiss.omp_set_num_threads(4)

def disc_directions(emb, qemb_train, train_qids, train_qrels, ids, m, n_pos=20000, seed=0):
    rng = np.random.default_rng(seed)
    qidx = {q: i for i, q in enumerate(train_qids)}
    idset = {s: i for i, s in enumerate(ids)}
    pos_q, pos_d, neg_q, neg_d = [], [], [], []
    for qid in train_qids:
        qi = qidx[qid]
        rel = sorted(idset[d] for d in train_qrels[qid] if d in idset)
        if not rel: continue
        per = max(1, n_pos // len(train_qids))
        for _ in range(min(30, per)):
            pos_q.append(qi); pos_d.append(rel[rng.integers(len(rel))])
            neg_q.append(qi); neg_d.append(rng.integers(len(emb)))
    pos_q = np.array(pos_q); pos_d = np.array(pos_d)
    neg_q = np.array(neg_q); neg_d = np.array(neg_d)
    P_pos = qemb_train[pos_q].T @ emb[pos_d] / len(pos_q)
    P_neg = qemb_train[neg_q].T @ emb[neg_d] / len(neg_q)
    S = (P_pos - P_neg); S = (S + S.T) / 2
    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(-np.abs(evals))[:m]
    return evecs[:, order].T

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
    Vd = disc_directions(emb, qt, tq, tr, ids, mm)
    r_disc = recall_at_k(brute_rank(qemb @ Vd.T, emb @ Vd.T), qids, qrels, ids)[0]
    print(f'{ds.split("/")[-1]:9s} {m:5s}: plainPCA={r_pca:.4f} | 判别协方差={r_disc:.4f} ({(r_disc-r_pca)*1000:+.0f}pp)', flush=True)
