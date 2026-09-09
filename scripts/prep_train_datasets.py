#!/usr/bin/env python3
"""为有 train 的数据集 (nfcorpus, hotpotqa) 准备: 语料+查询嵌入 + train 查询嵌入 + qrels.
用法: python prep_train_datasets.py [nfcorpus|hotpotqa]
"""
import os, json, sys, numpy as np
os.environ['HF_HOME'] = '/ssd1/zoulixin/tencent_previous_compression/experiments/caches/hf_cache'
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch

MODELS = {'e5': 'intfloat/e5-base-v2', 'bge': 'BAAI/bge-base-en-v1.5',
          'mini': 'sentence-transformers/all-MiniLM-L6-v2',
          'bgem3': 'BAAI/bge-m3', 'qwen3': 'Qwen/Qwen3-Embedding-0.6B'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = '/ssd1/zoulixin/tencent_previous_compression/experiments/data'


def embed(texts, model_name, out_path, batch=64):
    if os.path.exists(out_path):
        print(f"  {model_name} exists", flush=True)
        return
    model = SentenceTransformer(MODELS[model_name], device=device)
    embs = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    np.save(out_path, embs.astype(np.float32))
    print(f"  {model_name} done {embs.shape}", flush=True)
    del model; torch.cuda.empty_cache()


def prep_nfcorpus():
    ds = 'nfcorpus'
    ddir = f'{ROOT}/beir/{ds}'
    os.makedirs(ddir, exist_ok=True)
    corpus = load_dataset(f'BeIR/{ds}', 'corpus', split='corpus')
    queries = load_dataset(f'BeIR/{ds}', 'queries', split='queries')
    qr = load_dataset(f'BeIR/{ds}-qrels')
    ids = [r['_id'] for r in corpus]
    texts = [r['text'] for r in corpus]
    qids = [r['_id'] for r in queries]
    qtexts = [r['text'] for r in queries]
    json.dump({'ids': ids, 'qrels': {str(r['query-id']): [str(x['corpus-id']) for x in qr['test'] if str(x['query-id'])==str(r['query-id'])] for r in qr['test']}}, open(f'{ddir}/meta.json', 'w'))
    json.dump(texts, open(f'{ddir}/texts.json', 'w'))
    np.save(f'{ddir}/query_ids.npy', np.array(qids))
    np.save(f'{ddir}/query_texts.npy', np.array(qtexts))
    # train qrels/queries
    train_q = sorted(set(str(r['query-id']) for r in qr['train']))
    train_texts = [dict((r['_id'], r['text']) for r in queries)[q] for q in train_q]
    json.dump(train_q, open(f'{ddir}/train_qids.json', 'w'))
    json.dump(train_texts, open(f'{ddir}/train_qtexts.json', 'w'))
    tr_qrels = {}
    for r in qr['train']:
        tr_qrels.setdefault(str(r['query-id']), []).append(str(r['corpus-id']))
    json.dump(tr_qrels, open(f'{ddir}/train_qrels.json', 'w'))
    print(f'{ds}: corpus {len(texts)}, test queries {len(qids)}, train queries {len(train_q)}', flush=True)
    for m in MODELS:
        embed(texts, m, f'{ddir}/emb_{m}.npy', 32)
        embed(qtexts, m, f'{ddir}/qemb_{m}.npy', 32)
        embed(train_texts, m, f'{ddir}/qemb_train_{m}.npy', 32)


def prep_hotpotqa():
    ds = 'hotpotqa'
    ddir = f'{ROOT}/beir_big/{ds}'
    os.makedirs(ddir, exist_ok=True)
    qr = load_dataset(f'BeIR/{ds}-qrels')
    queries = load_dataset(f'BeIR/{ds}', 'queries', split='queries')
    qid2text = {str(r['_id']): r['text'] for r in queries}
    # train 查询子集 (前 2000) 用于校准
    train_q = sorted(set(str(r['query-id']) for r in qr['train']))[:2000]
    train_texts = [qid2text[q] for q in train_q]
    json.dump(train_q, open(f'{ddir}/train_qids.json', 'w'))
    json.dump(train_texts, open(f'{ddir}/train_qtexts.json', 'w'))
    tr_qrels = {}
    for r in qr['train']:
        q = str(r['query-id'])
        if q in train_q:
            tr_qrels.setdefault(q, []).append(str(r['corpus-id']))
    json.dump(tr_qrels, open(f'{ddir}/train_qrels.json', 'w'))
    print(f'{ds}: train subset {len(train_q)} queries', flush=True)
    for m in MODELS:
        embed(train_texts, m, f'{ddir}/qemb_train_{m}.npy', 32)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'nfcorpus'
    if which == 'nfcorpus':
        prep_nfcorpus()
    else:
        prep_hotpotqa()
    print('DONE', which, flush=True)
