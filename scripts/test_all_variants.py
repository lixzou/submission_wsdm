"""全面测检索引导变换变体 (诚实 train 校准, scifact e5, 未量化 keep=0.5/0.25/0.125)。
目标: 找出能稳定超过 plain PCA 的变体。
"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, recall_at_k, brute_rank
from merge_pca_retrieval import pca_directions
import faiss; faiss.omp_set_num_threads(4)

ds, m = 'data/beir/scifact', 'e5'
emb, qemb, qids, qrels, ids = load(ds, m)
d = emb.shape[1]
tq = json.load(open(f'{ds}/train_qids.json'))
tr = json.load(open(f'{ds}/train_qrels.json'))
qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)

def margin_importance_hard(emb, qemb, qids, qrels, ids, n_pairs=100000, hard_K=50, seed=0):
    """难负例 margin: 负例从 top-hard_K 相似文档中采样 (排除相关)。"""
    rng = np.random.default_rng(seed)
    qidx = {q: i for i, q in enumerate(qids)}
    judged = [q for q in qids if q in qrels]
    idset = {s: i for i, s in enumerate(ids)}
    pos, neg = [], []
    per_q = max(1, n_pairs // max(len(judged),1))
    for qid in judged:
        qi = qidx[qid]
        rel_docs = sorted(idset[dd] for dd in qrels[qid] if dd in idset)
        if not rel_docs: continue
        # 难负例池: top-hard_K 相似文档, 排除相关
        sims = qemb[qi] @ emb.T
        order = np.argsort(-sims)
        relset = set(rel_docs)
        neg_pool = [j for j in order if j not in relset][:hard_K]
        if not neg_pool: continue
        for _ in range(min(100, per_q)):
            di = rel_docs[rng.integers(len(rel_docs))]
            pos.append(qemb[qi]*emb[di])
            ni = neg_pool[rng.integers(len(neg_pool))]
            neg.append(qemb[qi]*emb[ni])
    pos=np.array(pos); neg=np.array(neg)
    mu = pos.mean(0)-neg.mean(0)
    se = np.sqrt(pos.var(0)/len(pos)+neg.var(0)/len(neg))+1e-12
    return np.abs(mu/se)

# 各种 w
w_uniform = None
# 难负例 margin
from pilot_truncation import margin_importance
w_hard = margin_importance_hard(emb, qt, tq, tr, ids, hard_K=50)
w_unif = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)

def recall_keep(V):
    return recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)[0]

for kk in [0.5, 0.25, 0.125]:
    mm = (max(1,int(d*kk))//8)*8
    V = pca_directions(emb, mm)
    r_pca = recall_keep(V)
    row = [f'keep={kk} pca={r_pca:.4f}']
    # 1. 均匀 margin 加权
    for name, w in [('unif', w_unif), ('hard', w_hard)]:
        ww = w/(w.max()+1e-12)+1e-6
        row.append(f'{name}w={recall_keep(pca_directions(emb,mm,w=ww)):.4f}')
    # 2. 子空间选择 top-K + 子空间PCA
    for Kx in [1.5, 2, 3]:
        K = min(d, int(mm*Kx))
        sel = np.argsort(-w_unif)[:K]
        Vsub = pca_directions(emb[:,sel], mm)
        # 需要把子空间方向映射回原空间
        # Vsub: (mm, K), 投影 = emb[:,sel] @ Vsub.T
        r = recall_at_k(brute_rank(qemb[:,sel] @ Vsub.T, emb[:,sel] @ Vsub.T), qids, qrels, ids)[0]
        row.append(f'sel{Kx}x={r:.4f}')
    print(' | '.join(row), flush=True)
