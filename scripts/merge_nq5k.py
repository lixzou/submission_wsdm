#!/usr/bin/env python3
"""spectral-guided PCA on NQ 5k subset (1024d models), aligned with c1_nq5k protocol."""
import json, numpy as np, sys
sys.path.insert(0, '.')
from merge_pca_retrieval import recall_at_k, brute_rank, margin_importance, pca_directions

DS = 'data/beir_big/nq'

def load_nq5k(model):
    emb = np.load(f'{DS}/emb5k_{model}.npy').astype(np.float32)
    qemb = np.load(f'{DS}/qemb5k_{model}.npy').astype(np.float32)
    meta = json.load(open(f'{DS}/meta.json'))
    all_qids = [str(q) for q in np.load(f'{DS}/query_ids.npy')]
    ids = [str(i) for i in meta['ids']]  # 全量 ids (2M)
    # c1_nq5k 协议: 文档取前 5000 (emb5k), 查询取前 500 (qemb5k),
    # judged-only: 只保留相关文档落在 5k 子集内的查询 (对齐 reference .9609/.846)
    ids5k = ids[:len(emb)]
    idset5k = set(ids5k)
    qids_sub = all_qids[:len(qemb)]
    judged = [q for q in qids_sub if q in meta['qrels'] and any(d in idset5k for d in meta['qrels'][q])]
    qrels5k = {q: {d for d in meta['qrels'][q] if d in idset5k} for q in judged}
    return emb, qemb, judged, qrels5k, ids5k

for model in ['bgem3', 'qwen3']:
    emb, qemb, qids, qrels, ids = load_nq5k(model)
    d = emb.shape[1]
    print(f"=== NQ5k {model}: emb {emb.shape} qemb {qemb.shape} judged {len(qids)} ===", flush=True)
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=0.7, n_pairs=100000)
    w = imp / (imp.max()+1e-12) + 1e-6
    res = {'reference_full': recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)}
    for keep in [0.25, 0.125]:
        m = max(1, int(d*keep))
        strat = {}
        strat['prefix'] = recall_at_k(brute_rank(qemb[:, :m], emb[:, :m]), qids, qrels, ids)
        order = np.argsort(-imp)
        strat['perm'] = recall_at_k(brute_rank(qemb[:, order[:m]], emb[:, order[:m]]), qids, qrels, ids)
        V = pca_directions(emb, m)
        strat['pca'] = recall_at_k(brute_rank(qemb @ V.T, emb @ V.T), qids, qrels, ids)
        for wp in [0.5, 1.0, 2.0]:
            ww = imp**wp; ww = ww/(ww.max()+1e-12)+1e-6
            Vw = pca_directions(emb, m, w=ww)
            strat[f'impw_pca_p{wp}'] = recall_at_k(brute_rank(qemb @ Vw.T, emb @ Vw.T), qids, qrels, ids)
        for K_mult in [2, 4]:
            K = min(d, max(m, int(m*K_mult)))
            sel = np.argsort(-imp)[:K]
            emb_sel, qemb_sel = emb[:, sel], qemb[:, sel]
            Vsub = pca_directions(emb_sel, m)
            strat[f'imp_sel_pca_K{K_mult}x'] = recall_at_k(brute_rank(qemb_sel @ Vsub.T, emb_sel @ Vsub.T), qids, qrels, ids)
        res[f'keep={keep:.3f}'] = strat
        print(f"  keep={keep}: " + json.dumps({k: round(v,4) for k,v in strat.items()}), flush=True)
    json.dump(res, open(f'results/runs/merge_nq5k_{model}.json','w'), indent=1)
    print(f"  saved results/runs/merge_nq5k_{model}.json", flush=True)
