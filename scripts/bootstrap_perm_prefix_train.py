#!/usr/bin/env python3
"""bootstrap_perm_prefix_train.py — 诚实 train 校准下 Perm vs Prefix 的 paired bootstrap。

scifact E5/BGE/Mini, keep=12.5%/25%, 未量化截断。
Perm = margin 排序取前 m 维; Prefix = 前 m 维。每 query 算 Recall@10 数组。
paired bootstrap (1000 resamples, seed=0): one-sided p = Pr(mean(delta)<=0)。
重写附录 tab:bootstrap (旧版是 test 泄漏的 c1 数据)。
"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, brute_rank, margin_importance

N_BOOT = 1000
SEED = 0
KEEPS = [0.25, 0.125]


def per_query_recall(qemb, o, qids, qrels, ids, k=10):
    rank = brute_rank(qemb.astype(np.float32), o.astype(np.float32), k)
    idset = ids
    recs = []
    for i, qid in enumerate(qids):
        if qid not in qrels:
            continue
        rel = qrels[qid]
        got = [str(idset[j]) for j in rank[i] if j < len(idset)]
        hit = sum(1 for g in got if g in rel)
        recs.append(hit / max(len(rel), 1))
    return np.array(recs, dtype=np.float64)


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for m in ['e5', 'bge', 'mini']:
        emb, qemb, qids, qrels, ids = load('data/beir/scifact', m)
        d = emb.shape[1]
        tq = json.load(open('data/beir/scifact/train_qids.json'))
        tr = json.load(open('data/beir/scifact/train_qrels.json'))
        qt = np.load(f'data/beir/scifact/qemb_train_{m}.npy').astype(np.float32)
        imp = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
        order = np.argsort(-imp)
        for k in KEEPS:
            mm = max(1, int(d * k))
            r_perm = per_query_recall(qemb[:, order[:mm]], emb[:, order[:mm]], qids, qrels, ids)
            r_prefix = per_query_recall(qemb[:, :mm], emb[:, :mm], qids, qrels, ids)
            delta = r_perm - r_prefix
            mean_delta = delta.mean()
            # paired bootstrap
            boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()
                             for _ in range(N_BOOT)])
            p = float((boot <= 0).mean())
            lo, hi = np.percentile(boot, [2.5, 97.5])
            cell = {f'keep={k}': {
                'perm': round(float(r_perm.mean()), 4),
                'prefix': round(float(r_prefix.mean()), 4),
                'delta': round(float(mean_delta), 4),
                'p': p,
                'ci': [round(lo, 4), round(hi, 4)],
            }}
            out.setdefault(m, {}).update(cell)
            print(f'{m} k={k}: perm={r_perm.mean():.4f} prefix={r_prefix.mean():.4f} '
                  f'delta={mean_delta:+.4f} p={p:.3f} CI=[{lo:.4f},{hi:.4f}]', flush=True)
    json.dump(out, open('results/runs/bootstrap_perm_prefix_train.json', 'w'), indent=1)
    print('SAVED results/runs/bootstrap_perm_prefix_train.json')


if __name__ == '__main__':
    main()
