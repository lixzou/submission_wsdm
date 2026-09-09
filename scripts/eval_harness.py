#!/usr/bin/env python3
"""BEIR 检索评测 harness: 精确/量化 recall@10 + nDCG@10。
协议: brute-force 余弦检索（BEIR 口径），量化域近似内积可选 rerank。
用法: python eval_harness.py --emb data/beir/nq/emb_e5.npy --qemb ... --meta meta.json
     --quant sq8|bq|pq|none --rerank 100
"""
import argparse, json, os
import numpy as np
import faiss

def load(args):
    emb = np.load(args.emb).astype(np.float32)
    qemb = np.load(args.qemb).astype(np.float32)
    meta = json.load(open(args.meta))
    qrels = {str(q): {str(d): 1 for d in docs} for q, docs in meta["qrels"].items()}
    return emb, qemb, qrels, meta["ids"]

def quantize(x, mode, pq_m=8, pq_ks=256):
    if mode == "none": return x, lambda q: q
    if mode == "sq8":
        qx = np.zeros(x.shape, np.uint8)
        # 全局标量量化 int8 (faiss 式)
        d = x.shape[1]
        mi, ma = float(x.min()), float(x.max())
        qx = np.clip(((x - mi) / (ma - mi) * 255), 0, 255).astype(np.uint8)
        def fq(q): return ((q - mi) / (ma - mi) * 255).astype(np.float32)
        return qx, fq
    if mode == "bq":
        return np.packbits((x > 0).astype(np.uint8), axis=1), lambda q: (q > 0).astype(np.float32)
    if mode == "pq":
        d = x.shape[1]; dsub = d // pq_m
        xsub = x[:, :dsub * pq_m].copy()
        pq = faiss.ProductQuantizer(dsub * pq_m, pq_m, 8)
        pq.train(xsub)
        codes = pq.compute_codes(xsub)
        def fq(q): return pq.decode(pq.compute_codes(q[:, :dsub * pq_m]))
        return codes, fq
    raise ValueError(mode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True); ap.add_argument("--qemb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--quant", default="none", choices=["none", "sq8", "bq", "pq"])
    ap.add_argument("--rerank", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="/tmp/eval_result.json")
    args = ap.parse_args()

    emb, qemb, qrels, ids = load(args)
    N, d = emb.shape
    # 参考精确检索（分批避免内存爆）
    qids = [str(i) for i in range(len(qemb))]  # meta 存了 query_ids
    qids = list(np.load(os.path.join(os.path.dirname(args.meta), "query_ids.npy")))
    exact_rank = np.zeros((len(qemb), args.k), np.int64)
    B = 4096
    for i in range(0, len(qemb), B):
        sim = qemb[i:i+B] @ emb.T
        exact_rank[i:i+B] = np.argpartition(-sim, args.k, axis=1)[:, :args.k]

    # 量化域近似检索
    codes, fq = quantize(emb, args.quant)
    qq = fq(qemb)
    if args.quant == "bq":
        codes_f = np.unpackbits(codes, axis=1).astype(np.float32) * 2 - 1
        qq_f = (qemb > 0).astype(np.float32) * 2 - 1
        qrank = np.zeros((len(qemb), args.k), np.int64)
        for i in range(0, len(qemb), B):
            sim = qq_f[i:i+B] @ codes_f.T
            qrank[i:i+B] = np.argpartition(-sim, args.k, axis=1)[:, :args.k]
    elif args.quant in ("sq8", "pq"):
        codes_f = codes.astype(np.float32) if args.quant == "sq8" else codes
        if args.quant == "pq":
            import faiss
            pq = faiss.ProductQuantizer(codes_f.shape[1] * 8, codes_f.shape[1], 8)
            pass  # 简化: PQ 用码字质心近似（自实现阶段再优化）
        qrank = np.zeros((len(qemb), args.k), np.int64)
        for i in range(0, len(qemb), B):
            sim = qq[i:i+B] @ codes_f.T
            qrank[i:i+B] = np.argpartition(-sim, args.k, axis=1)[:, :args.k]
    else:
        qrank = exact_rank

    # rerank: 精确重排 top-r
    if args.rerank > 0:
        rr = min(args.rerank, N)
        cand = np.zeros((len(qemb), rr), np.int64)
        for i in range(0, len(qemb), B):
            sim = qq[i:i+B] @ codes_f.T if args.quant != "none" else qemb[i:i+B] @ emb.T
            cand[i:i+B] = np.argpartition(-sim, rr, axis=1)[:, :rr]
        final = np.zeros((len(qemb), args.k), np.int64)
        for i in range(0, len(qemb), B):
            exact = qemb[i:i+B] @ emb[cand[i:i+B]].transpose(0, 2, 1)
            top = np.argpartition(-exact, args.k, axis=2)[:, :, :args.k]
            # 简化: 按精确分排序
            order = np.argsort(-exact, axis=2)[:, :, :args.k]
            final[i:i+B] = np.take_along_axis(cand[i:i+B], order, axis=1)
        qrank = final

    # recall@k / nDCG@k
    recalls, ndcgs = [], []
    for i, qid in enumerate(qids):
        if qid not in qrels: continue
        rel = qrels[qid]
        got = [str(ids[j]) for j in qrank[i] if j < len(ids)]
        hit = sum(1 for g in got if g in rel)
        recalls.append(hit / max(len(rel), 1))
        dcg = sum((1.0 if g in rel else 0.0) / np.log2(2 + r) for r, g in enumerate(got))
        idcg = sum(1.0 / np.log2(2 + r) for r in range(min(len(rel), args.k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    res = {"recall@%d" % args.k: float(np.mean(recalls)), "ndcg@%d" % args.k: float(np.mean(ndcgs)),
           "n_queries": len(recalls), "quant": args.quant, "rerank": args.rerank}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
