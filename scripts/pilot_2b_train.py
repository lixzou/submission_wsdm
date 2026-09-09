#!/usr/bin/env python3
"""pilot_2b_train.py — tab:c2 诚实 train 校准重跑 (scifact E5)。

与 pilot_2b.py 相同的分配比较, 但 margin 统计量拟合 train 查询 (tq/tr/qt),
评估仍在 test 查询。uniform/var 与 qrels 无关, 不变化。
"""
import json, os, sys, time
import numpy as np
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from rabitq_py import random_orthogonal
from pilot_2b import block_quantize, waterfill


def run_one(m):
    ds = 'data/beir/scifact'
    emb, qemb, qids, qrels, ids = load(ds, m)
    N, d = emb.shape
    tq = json.load(open(f'{ds}/train_qids.json'))
    tr = json.load(open(f'{ds}/train_qrels.json'))
    qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)
    imp = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
    P = random_orthogonal(d, seed=7)
    rot = (P.T @ emb.T).T.astype(np.float64)

    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    print("reference:", round(ref, 4), flush=True)

    BS = 64
    nblocks = d // BS
    margin_order = np.argsort(-imp)
    var_order = np.argsort(-np.var(emb, 0))
    from sklearn.cluster import SpectralClustering
    X = emb - emb.mean(0)
    cov = X.T @ X / len(X)
    std = np.sqrt(np.diag(cov)) + 1e-8
    rho = np.abs(cov) / np.outer(std, std)
    np.fill_diagonal(rho, 0)
    labels = SpectralClustering(n_clusters=nblocks, affinity="precomputed",
                                assign_labels="kmeans", random_state=0, n_init=10).fit_predict(rho)
    buckets = [np.where(labels == b)[0] for b in range(nblocks)]
    bucket_imp = np.array([imp[b].mean() for b in buckets])
    bucket_order = np.argsort(-bucket_imp)

    contig_blocks = [np.arange(b * BS, (b + 1) * BS) for b in range(nblocks)]
    margin_blocks = [margin_order[b * BS:(b + 1) * BS] for b in range(nblocks)]
    var_blocks = [var_order[b * BS:(b + 1) * BS] for b in range(nblocks)]
    ratc_blocks = [buckets[bi] for bi in bucket_order]

    results = {"reference": ref, "calib": "train", "n_cal": len(tq)}
    for avg_bits in [4, 2, 1]:
        t0 = time.time()
        res = {}
        Bu = np.full(nblocks, avg_bits)
        o = block_quantize(rot, P, emb, contig_blocks, Bu)
        res["uniform"] = round(recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0], 4)
        Bm = waterfill(margin_order, d, avg_bits, BS)
        Bm_blocks = [int(np.median(Bm[bl])) for bl in margin_blocks]
        o = block_quantize(rot, P, emb, margin_blocks, Bm_blocks)
        res["margin_block"] = round(recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0], 4)
        Bv = waterfill(var_order, d, avg_bits, BS)
        Bv_blocks = [int(np.median(Bv[bl])) for bl in var_blocks]
        o = block_quantize(rot, P, emb, var_blocks, Bv_blocks)
        res["var_block"] = round(recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0], 4)
        nb = len(ratc_blocks)
        block_bits = np.zeros(nb, int)
        budget = avg_bits * d
        for bi in range(nb):
            for b in [8, 4, 2, 1]:
                sz = len(ratc_blocks[bi])
                if budget >= b * sz:
                    block_bits[bi] = b; budget -= b * sz; break
        o = block_quantize(rot, P, emb, ratc_blocks, block_bits)
        res["ratc_bucket"] = round(recall_at_k(brute_rank(qemb, o), qids, qrels, ids)[0], 4)
        results["avg_bits=%d" % avg_bits] = res
        print(f"avg_bits={avg_bits}: {res} [{time.time()-t0:.0f}s]", flush=True)
    out = f'results/runs/pilot_2b_train_scifact_{m}.json'
    json.dump(results, open(out, 'w'), indent=1)
    print('SAVED', out)


if __name__ == '__main__':
    for mm in ['e5', 'bge', 'mini', 'bgem3', 'qwen3']:
        run_one(mm)
