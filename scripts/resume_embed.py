#!/usr/bin/env python3
"""续跑缺失嵌入 (fiqa bgem3/qwen3 + nfcorpus qwen3), 单进程顺序, 跳过已有."""
import os, json, numpy as np
os.environ['HF_HOME'] = '/ssd1/zoulixin/tencent_previous_compression/experiments/caches/hf_cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from sentence_transformers import SentenceTransformer
import torch

MODELS = {'bgem3': 'BAAI/bge-m3', 'qwen3': 'Qwen/Qwen3-Embedding-0.6B'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = '/ssd1/zoulixin/tencent_previous_compression/experiments/data/beir'

JOBS = [
    ('fiqa', 'bgem3'), ('fiqa', 'qwen3'), ('nfcorpus', 'qwen3'),
]


def embed(x, m, out, batch=64):
    if os.path.exists(out):
        print(f'{os.path.basename(out)} exists', flush=True)
        return
    model = SentenceTransformer(MODELS[m], device=device)
    e = model.encode(x, batch_size=batch, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
    np.save(out, e.astype(np.float32))
    print(f'{os.path.basename(out)} done {e.shape}', flush=True)
    del model
    torch.cuda.empty_cache()


for ds, m in JOBS:
    ddir = f'{ROOT}/{ds}'
    texts = json.load(open(f'{ddir}/texts.json'))
    qtexts = list(np.load(f'{ddir}/query_texts.npy'))
    train_texts = json.load(open(f'{ddir}/train_qtexts.json'))
    print(f'== {ds} {m} ==', flush=True)
    embed(texts, m, f'{ddir}/emb_{m}.npy', 32)
    embed(qtexts, m, f'{ddir}/qemb_{m}.npy', 32)
    embed(train_texts, m, f'{ddir}/qemb_train_{m}.npy', 32)
print('ALL DONE', flush=True)
