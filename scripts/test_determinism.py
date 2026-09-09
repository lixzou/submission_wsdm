"""测试 run_reco 在同一进程内两次调用是否一致 + 检查 margin_importance 确定性"""
import json, numpy as np
import os, sys
sys.path.insert(0, '.')
from pilot_truncation import load, margin_importance
from main_table import run_reco
import faiss
faiss.omp_set_num_threads(4)

emb, qemb, qids, qrels, ids = load('data/beir/scifact', 'e5')
train_qids = json.load(open('data/beir/scifact/train_qids.json'))
train_qrels = json.load(open('data/beir/scifact/train_qrels.json'))
qemb_train = np.load('data/beir/scifact/qemb_train_e5.npy').astype(np.float32)
imp = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)

m = 384
# 同一进程两次调用
r1 = run_reco(emb, qemb, imp, qids, qrels, ids, m, 2, calib_qids=train_qids, calib_qrels=train_qrels)
r2 = run_reco(emb, qemb, imp, qids, qrels, ids, m, 2, calib_qids=train_qids, calib_qrels=train_qrels)
print(f'run1={r1:.4f} run2={r2:.4f} 一致={r1==r2}')

# 检查 PYTHONHASHSEED 影响: 重算 imp 两次
imp_a = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)
imp_b = margin_importance(emb, qemb_train, train_qids, train_qrels, ids, calib_frac=1.0)
print(f'imp 两次一致: {np.array_equal(imp_a, imp_b)}')
