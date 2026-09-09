#!/usr/bin/env python3
"""PQ/OPQ baseline v3（正确协议: decode→归一化→cosine，与参考一致）。
faiss ProductQuantizer / OPQMatrix，decode 后 L2 归一化，余弦检索（与 pilot 参考同口径）。
"""
import argparse, json, os, numpy as np, faiss, warnings
warnings.filterwarnings("ignore")

def load(ds_dir, model):
    emb = np.load(os.path.join(ds_dir, "emb_%s.npy" % model)).astype(np.float32)
    qemb = np.load(os.path.join(ds_dir, "qemb_%s.npy" % model)).astype(np.float32)
    meta = json.load(open(os.path.join(ds_dir, "meta.json")))
    qids = [str(q) for q in np.load(os.path.join(ds_dir, "query_ids.npy"))]
    qrels = {str(q): {str(d) for d in docs} for q, docs in meta["qrels"].items()}
    ids = [str(i) for i in meta["ids"]]
    return emb, qemb, qids, qrels, ids

def recall_from_sim(sim, qids, qrels, ids, k=10):
    recs = []
    B = 2048
    for i in range(0, len(qids), B):
        top = np.argpartition(-sim[i:i+B], k, axis=1)[:, :k]
        for t, j in enumerate(range(i, min(i+B, len(qids)))):
            qid = qids[j]
            if qid not in qrels: continue
            rel = qrels[qid]
            got = [str(ids[g]) for g in top[t] if g < len(ids)]
            recs.append(sum(1 for g in got if g in rel) / max(len(rel), 1))
    return float(np.mean(recs)) if recs else 0.0

def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--out", default="/tmp/pq_opq.json")
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    if args.subset > 0 and args.subset < len(emb):
        emb = emb[:args.subset]; ids = ids[:args.subset]
    d = emb.shape[1]
    print("model", args.model, "| emb", emb.shape, "| queries", len(qids), flush=True)

    res = {}
    ref_sim = qemb @ emb.T
    res["reference"] = recall_from_sim(ref_sim, qids, qrels, ids)
    print("reference recall@10 =", round(res["reference"], 4), flush=True)

    def eval_pq(m, rot=None):
        # rot: 可选 OPQ 旋转矩阵（d×d）。对 emb/qemb 先旋转，PQ 在旋转域训练。
        x = emb if rot is None else emb @ rot
        qx = qemb if rot is None else qemb @ rot
        pq = faiss.ProductQuantizer(d, m, 8)
        try: pq.cp.niter = 10
        except: pass
        pq.train(x)
        codes = pq.compute_codes(x)
        recon = norm(pq.decode(codes))
        sim = qx @ recon.T
        return recall_from_sim(sim, qids, qrels, ids)

    for m in [8, 16, 32]:
        res["PQ_m%d" % m] = eval_pq(m)
        print("PQ_m%d recall@10 =" % m, round(res["PQ_m%d" % m], 4), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)

    # 字节对齐 PQ（32x/16x = m96/m192, dsub=8/4）
    for m in [48, 96, 192]:
        if d % m == 0:
            res["PQ_align_m%d" % m] = eval_pq(m)
            print("PQ_align_m%d recall@10 =" % m, round(res["PQ_align_m%d" % m], 4), flush=True)
            json.dump(res, open(args.out, "w"), indent=1)

    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)

if __name__ == "__main__":
    main()
