"""诊断: train校准w 的top维度 与 test校准w 的top维度 的对齐。
若 train w top 维度与 test w 差异大 -> 加权伤害; 若一致 -> 加权应该有用。
找出需要数据集的特征。"""
import json, numpy as np, sys, os
os.chdir('/ssd1/zoulixin/tencent_previous_compression/experiments')
MJ = '/ssd1/zoulixin/tencent_previous_compression/experiments/scripts'
sys.path.insert(0, MJ)
from pilot_truncation import load, margin_importance

for ds in ['data/beir/scifact', 'data/beir/fiqa', 'data/beir/nfcorpus']:
    for m in ['e5', 'bge', 'mini']:
        try:
            emb, qemb, qids, qrels, ids = load(ds, m)
            tq = json.load(open(f'{ds}/train_qids.json'))
            tr = json.load(open(f'{ds}/train_qrels.json'))
            qt = np.load(f'{ds}/qemb_train_{m}.npy').astype(np.float32)
            w_train = margin_importance(emb, qt, tq, tr, ids, calib_frac=1.0)
            w_test = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=1.0)
            # Spearman corr + top-50 重叠
            corr = np.corrcoef(w_train, w_test)[0,1]
            top = 50
            inter = len(set(np.argsort(-w_train)[:top]) & set(np.argsort(-w_test)[:top]))
            print(f'{ds.split("/")[-1]:10s} {m:5s}: corr={corr:.3f} top{top}重叠={inter}/{top}')
        except Exception as e:
            print(f'{ds} {m}: ERR {str(e)[:40]}')
