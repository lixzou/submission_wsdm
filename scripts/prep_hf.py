#!/usr/bin/env python3
"""BEIR 数据经 HuggingFace datasets 下载（走 hf-mirror），embedding 预计算。
用法: python prep_hf.py --datasets scifact nq --models e5 bge mini --subset 200000
"""
import argparse, json, os
import numpy as np
from datasets import load_dataset

ROOT = "/ssd1/zoulixin/tencent_previous_compression/experiments/data/beir"
os.makedirs(ROOT, exist_ok=True)

MODELS = {
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "mini": "sentence-transformers/all-MiniLM-L6-v2",
    "bgem3": "BAAI/bge-m3",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact"])
    ap.add_argument("--models", nargs="+", default=["e5", "bge", "mini"])
    ap.add_argument("--subset", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    for ds in args.datasets:
        ds_dir = os.path.join(ROOT, ds)
        os.makedirs(ds_dir, exist_ok=True)
        meta_path = os.path.join(ds_dir, "meta.json")
        if not os.path.exists(meta_path):
            print("loading", ds, flush=True)
            corpus_ds = load_dataset("BeIR/%s" % ds, "corpus", split="corpus")
            queries_ds = load_dataset("BeIR/%s" % ds, "queries", split="queries")
            qrels_ds = load_dataset("BeIR/%s-qrels" % ds, split="test")
            corpus_items = [(r["_id"], r["text"]) for r in corpus_ds.select(range(min(args.subset, len(corpus_ds))))]
            ids = [c[0] for c in corpus_items]
            texts = [c[1] for c in corpus_items]
            qids = [r["_id"] for r in queries_ds]
            qtexts = [r["text"] for r in queries_ds]
            qrels = {}
            for r in qrels_ds:
                q, d = str(r["query-id"]), str(r["corpus-id"])
                qrels.setdefault(q, []).append(d)
            json.dump({"ids": ids, "qrels": qrels}, open(meta_path, "w"))
            np.save(os.path.join(ds_dir, "query_ids.npy"), np.array(qids))
            np.save(os.path.join(ds_dir, "query_texts.npy"), np.array(qtexts))
            json.dump(texts, open(os.path.join(ds_dir, "texts.json"), "w"))
        else:
            meta = json.load(open(meta_path))
            ids = meta["ids"]
            texts = json.load(open(os.path.join(ds_dir, "texts.json")))
            qids = list(np.load(os.path.join(ds_dir, "query_ids.npy")))
            qtexts = list(np.load(os.path.join(ds_dir, "query_texts.npy")))
        print(ds, "corpus", len(texts), "queries", len(qtexts), flush=True)

        for mname in args.models:
            out = os.path.join(ds_dir, "emb_%s.npy" % mname)
            if os.path.exists(out):
                print(ds, mname, "done", flush=True); continue
            model = SentenceTransformer(MODELS[mname], device=device)
            embs = model.encode(texts, batch_size=args.batch, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
            np.save(out, embs.astype(np.float32))
            qembs = model.encode(qtexts, batch_size=args.batch, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
            np.save(os.path.join(ds_dir, "qemb_%s.npy" % mname), qembs.astype(np.float32))
            print(ds, mname, "done:", embs.shape, flush=True)
            del model
            if device == "cuda": torch.cuda.empty_cache()
    print("PREP DONE", flush=True)

if __name__ == "__main__":
    main()
