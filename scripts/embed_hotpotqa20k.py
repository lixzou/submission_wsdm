#!/usr/bin/env python3
"""hotpotqa20k 补 bgem3/qwen3 嵌入 (缺啥补啥)."""
import os, json, numpy as np
os.environ['HF_HOME'] = '/ssd1/zoulixin/tencent_previous_compression/experiments/caches/hf_cache'
from sentence_transformers import SentenceTransformer
import torch

MODELS = {'bgem3': 'BAAI/bge-m3', 'qwen3': 'Qwen/Qwen3-Embedding-0.6B'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
D = '/ssd1/zoulixin/tencent_previous_compression/experiments/data/beir_big/hotpotqa20k'
texts = json.load(open(f'{D}/texts.json'))
qtexts = list(np.load(f'{D}/query_texts.npy'))
train_texts = json.load(open(f'{D}/train_qtexts.json'))


def embed(x, m, out):
    if os.path.exists(out):
        print(f'{m} {os.path.basename(out)} exists', flush=True)
        return
    model = SentenceTransformer(MODELS[m], device=device)
    e = model.encode(x, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
    np.save(out, e.astype(np.float32))
    print(f'{m} {os.path.basename(out)} done {e.shape}', flush=True)
    del model
    torch.cuda.empty_cache()


for m in MODELS:
    embed(texts, m, f'{D}/emb_{m}.npy')
    embed(qtexts, m, f'{D}/qemb_{m}.npy')
    embed(train_texts, m, f'{D}/qemb_train_{m}.npy')
print('DONE', flush=True)
