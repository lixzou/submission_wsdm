"""验证: 现有 train_calib RECO=.8186 是否来自 test 校准(泄漏)的旧版行为"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
MJ = '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts'
sys.path.insert(0, MJ)
from pilot_truncation import load, margin_importance
from main_table import run_reco
import faiss; faiss.omp_set_num_threads(4)

emb, qemb, qids, qrels, ids = load('data/beir/scifact', 'e5')
tq = json.load(open('data/beir/scifact/train_qids.json'))
tr = json.load(open('data/beir/scifact/train_qrels.json'))
qt = np.load('data/beir/scifact/qemb_train_e5.npy').astype(np.float32)
imp = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)

old = json.load(open('results/runs/train_calib/scifact/e5.json'))
m = 384
# (a) train 校准 (诚实, 新代码)
r_train = run_reco(emb, qemb, imp, qids, qrels, ids, m, 2, calib_qids=tq, calib_qrels=tr)
# (b) test 校准 (泄漏, 旧版默认行为)
r_test = run_reco(emb, qemb, imp, qids, qrels, ids, m, 2)  # 无 calib -> 默认用 qids(评估集)
print(f'50%: train校准={r_train:.4f}  test校准={r_test:.4f}  现有json={old["keep=0.500"]["reco"]:.4f}')
