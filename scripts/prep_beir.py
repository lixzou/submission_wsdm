#!/usr/bin/env python3
"""BEIR 数据集下载 + 冻结模型 embedding 预计算（pilot 规模可控）。
用法: python prep_beir.py --datasets nq msmarco hotpotqa --models e5 bge mini --subset 200000
"""
import argparse, json, os, sys
import numpy as np

ROOT = "/ssd1/zoulixin/tencent_previous_compression/experiments/data/beir"

MODELS = {
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "mini": "sentence-transformers/all-MiniLM-L6-v2",
    "bgem3": "BAAI/bge-m3",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nq", "msmarco", "hotpotqa"])
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
        # 通过 beir 库下载
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(ds)
        zip_path = os.path.join(ROOT, ds + ".zip")
        if not os.path.exists(zip_path):
            util.download_url(url, zip_path)
        util.unzip(zip_path, ds_dir)
        loader = GenericDataLoader(data_folder=ds_dir)
        corpus, queries, qrels = loader.load(split="test")
        corpus_items = list(corpus.items())[:args.subset]
        corpus_ids = [cid for cid, _ in corpus_items]
        corpus_texts = [c["text"] for _, c in corpus_items]
        json.dump({"ids": corpus_ids, "qrels": {str(k): v for k, v in qrels.items()}},
                  open(os.path.join(ds_dir, "meta.json"), "w"))
        np.save(os.path.join(ds_dir, "query_ids.npy"), np.array(list(queries.keys())))
        np.save(os.path.join(ds_dir, "query_texts.npy"), np.array(list(queries.values())))

        for mname in args.models:
            mpath = MODELS[mname]
            out = os.path.join(ds_dir, "emb_%s.npy" % mname)
            if os.path.exists(out):
                print(ds, mname, "already done", flush=True); continue
            model = SentenceTransformer(mpath, device=device)
            embs = model.encode(corpus_texts, batch_size=args.batch, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
            np.save(out, embs.astype(np.float32))
            # 查询 embedding
            qtexts = list(queries.values())
            qembs = model.encode(qtexts, batch_size=args.batch, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
            np.save(os.path.join(ds_dir, "qemb_%s.npy" % mname), qembs.astype(np.float32))
            print(ds, mname, "done:", embs.shape, flush=True)
            del model
            if device == "cuda": torch.cuda.empty_cache()
    print("PREP DONE", flush=True)

if __name__ == "__main__":
    main()
