#!/usr/bin/env python3
"""为已有数据目录补生成指定模型的 emb/qemb (缺啥补啥)。GPU 优先。
用法: python embed_extra.py --ds_dir data/beir_big/arguana --models bgem3 qwen3
"""
import argparse, json, os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

MODELS = {"bgem3": "BAAI/bge-m3", "qwen3": "Qwen/Qwen3-Embedding-0.6B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--models", nargs="+", default=["bgem3", "qwen3"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--subset", type=int, default=0, help="N>0 只编码前 N 文本 (如 NQ 200k)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    texts = json.load(open(os.path.join(args.ds_dir, "texts.json")))
    qtexts = list(np.load(os.path.join(args.ds_dir, "query_texts.npy")))
    if args.subset > 0:
        texts = texts[:args.subset]
    print(args.ds_dir, "corpus", len(texts), "queries", len(qtexts), flush=True)

    for mname in args.models:
        out = os.path.join(args.ds_dir, "emb_%s.npy" % mname)
        if os.path.exists(out):
            print(mname, "already exists, skip", flush=True)
            continue
        model = SentenceTransformer(MODELS[mname], device=device)
        embs = model.encode(texts, batch_size=args.batch, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        np.save(out, embs.astype(np.float32))
        qembs = model.encode(qtexts, batch_size=args.batch, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
        np.save(os.path.join(args.ds_dir, "qemb_%s.npy" % mname), qembs.astype(np.float32))
        print(mname, "done", embs.shape, flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
