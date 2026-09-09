#!/usr/bin/env python3
"""核心诊断: train 校准的 w 与 test 校准的 w 是否相关?
如果相关 -> 加权在 train 校准下应该也有用 (说明我的重跑有bug)
如果不相关 -> train 校准的 w 质量差, 是数据/校准问题
"""
import json, numpy as np
from pilot_truncation import load, margin_importance

emb, qemb, qids, qrels, ids = load('data/beir/scifact', 'e5')
train_qids = json.load(open('data/beir/scifact/train_qids.json'))
train_qrels = json.load(open('data/beir/scifact/train_qrels.json'))
qemb_train = np.load('data/beir/scifact/qemb_train_e5.npy').astype(np.float32)

# test 校准的 w (正文用的协议, 泄漏)
w_test = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=1.0)
# train 校准的 w (诚实协议)
w_train = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)

print('w_test shape:', w_test.shape)
print('w_train shape:', w_train.shape)
print('Spearman corr:', np.corrcoef(w_test, w_train)[0,1].round(4))
# 前50个最重要的维度的重叠率
topk = 50
top_test = set(np.argsort(-w_test)[:topk])
top_train = set(np.argsort(-w_train)[:topk])
print(f'top{topk} 重叠:', len(top_test & top_train), '/', topk)
# 各自 top 维度的值对比
print('w_test top10:', np.sort(w_test)[-10:].round(2))
print('w_train top10:', np.sort(w_train)[-10:].round(2))
