"""判别协方差完整管线: 判别方向 -> 随机旋转 -> 校准量化 -> test评估。
对比 plain PCA 完整管线, 诚实 train 校准。"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, recall_at_k, brute_rank
from merge_pca_retrieval import pca_directions
from rabitq_py import random_orthogonal
from sota_pipeline_check import fit_calib_levels, quant_recall
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
              ('data/beir/scifact','bgem3'), ('data/beir/scifact','qwen3'),
              ('data/beir/fiqa','e5'), ('data/beir/fiqa','bge'), ('data/beir/fiqa','mini'),
              ('data/beir/nfcorpus','e5')]:
    emb, qemb, qids, qrels, ids = load(ds, m)
    d = emb.shape[1]
    tq = json.load(open(f'{ds}/train_qids.json'))
    tr = json.load(open(f'{ds}/train_qrels.json'))
    qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)
    kk = 0.5; mm = (max(1,int(d*kk))//8)*8
    B = 2
    P = random_orthogonal(mm, seed=0)
    # plain PCA 完整管线
    V = pca_directions(emb, mm)
    ep, qp = emb @ V.T, qemb @ V.T
    rot = (ep @ P).astype(np.float64)
    lv = fit_calib_levels(rot, tq, tr, ids, B)
    r_pca = quant_recall(qp, ep, np.arange(mm), B, lv, qids, qrels, ids)
    # 判别协方差完整管线
    Vd = disc_directions(emb, qt, tq, tr, ids, mm)
    ed, qd = emb @ Vd.T, qemb @ Vd.T
    rotd = (ed @ P).astype(np.float64)
    lvd = fit_calib_levels(rotd, tq, tr, ids, B)
    r_disc = quant_recall(qd, ed, np.arange(mm), B, lvd, qids, qrels, ids)
    print(f'{ds.split("/")[-1]:9s} {m:5s}: PCA量化={r_pca:.4f} | 判别量化={r_disc:.4f} ({(r_disc-r_pca)*1000:+.0f}pp)', flush=True)
