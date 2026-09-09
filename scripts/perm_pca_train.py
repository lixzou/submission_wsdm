#!/usr/bin/env python3
"""perm_pca_train.py — 诚实 train 校准下, 未量化 perm vs PCA 截断 Recall@10。

对 scifact 5 模型在 keep=0.5/0.25/0.125 计算:
  - perm : 按 margin 排序选前 m 维 (原始空间, 不旋转)
  - pca  : 前 m 主方向投影
校准: margin_importance 拟合 train 查询; 评估 test 查询。
输出 perm-pca 差值 (points), 用于核对 sec:analysis 的 -14.0/-2.2/... 声明。
"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from merge_pca_retrieval import pca_directions

DS = 'data/beir/scifact'
MODELS = ['e5', 'bge', 'mini', 'bgem3', 'qwen3']
KEEPS = [0.5, 1.0 / 3.0, 0.25, 1.0 / 6.0, 0.125]

res = {}
for m in MODELS:
    emb, qemb, qids, qrels, ids = load(DS, m)
    d = emb.shape[1]
    tq = json.load(open(f'{DS}/train_qids.json'))
    tr = json.load(open(f'{DS}/train_qrels.json'))
    qt = np.load(f'{DS}/qemb_train_{m}.npy').astype(np.float32)
    imp = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
    order = np.argsort(-imp)
    res[m] = {'d': d}
    for k in KEEPS:
        mm = max(1, int(d * k))
        # perm: 原始空间取前 m 维 (按 margin 排序)
        o_perm = emb[:, order[:mm]]
        r_perm = recall_at_k(brute_rank(qemb[:, order[:mm]], o_perm), qids, qrels, ids)[0]
        # pca: 前 m 主方向
        V = pca_directions(emb, mm)
        o_pca = emb @ V.T
        r_pca = recall_at_k(brute_rank(qemb @ V.T, o_pca), qids, qrels, ids)[0]
        res[m][f'keep={k}'] = {'perm': round(r_perm, 4), 'pca': round(r_pca, 4),
                               'gap_pts': round(100 * (r_perm - r_pca), 1)}
        print(f'{m:6s} k={k}: perm={r_perm:.4f} pca={r_pca:.4f} gap={100*(r_perm-r_pca):+6.1f} pts', flush=True)
    # effective rank of covariance
    emb_c = emb - emb.mean(0)
    ev = np.linalg.svd(emb_c, compute_uv=False) ** 2
    ev /= ev.sum()
    cum = np.cumsum(ev)
    eff = (cum < 0.9).sum()
    res[m]['effective_rank90'] = int(eff)
    res[m]['top1_share'] = round(float(ev[0]) * 100, 1)
    print(f'{m:6s} effective_rank90={eff} top1_var={ev[0]*100:.1f}%', flush=True)
json.dump(res, open('results/runs/perm_pca_train.json', 'w'), indent=1)
print('SAVED results/runs/perm_pca_train.json')
