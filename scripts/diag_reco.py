"""诊断 run_reco 非确定性的唯一可能源"""
import json, numpy as np, sys
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts')
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments')
from pilot_truncation import load, margin_importance
from main_table import run_reco
import faiss; faiss.omp_set_num_threads(4)

emb, qemb, qids, qrels, ids = load('data/beir/scifact', 'e5')
tq = json.load(open('data/beir/scifact/train_qids.json'))
tr = json.load(open('data/beir/scifact/train_qrels.json'))
qt = np.load('data/beir/scifact/qemb_train_e5.npy').astype(np.float32)

# 1. imp 是否确定
imp1 = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
imp2 = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
print('imp 两次一致:', np.array_equal(imp1, imp2))

# 2. run_reco 两次 (同进程)
m = 384
r1 = run_reco(emb, qemb, imp1, qids, qrels, ids, m, 2, calib_qids=tq, calib_qrels=tr)
r2 = run_reco(emb, qemb, imp1, qids, qrels, ids, m, 2, calib_qids=tq, calib_qrels=tr)
print(f'run_reco 同进程两次: {r1:.4f} vs {r2:.4f} 一致={r1==r2}')

# 3. 直接对比: 现有 json 的 .8186 从哪来? 尝试 test 校准 (calib_qids=qids)
r_test = run_reco(emb, qemb, imp1, qids, qrels, ids, m, 2)  # 无 calib -> 默认 test 校准
print(f'run_reco 默认(test校准): {r_test:.4f}')

# 4. 旧版 main_table (baseline) 的 run_reco
sys.path.insert(0, '/ssd1/zoulixin/tencent_previous_compression/experiments')
import importlib
# 备份
import main_table as mt_old
# 强制重新加载 baseline 版本
import sys as s
for mod in list(s.modules):
    if 'main_table' in mod: del s.modules[mod]
sys.path = ['/ssd1/zoulixin/tencent_previous_compression/experiments'] + sys.path
try:
    from main_table import run_reco as rr_old
    print('baseline run_reco sig:', __import__('inspect').signature(rr_old))
except Exception as e:
    print('baseline main_table import err:', e)
