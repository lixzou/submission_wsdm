#!/usr/bin/env python3
"""缺口实验 v1: 补齐 Phase 4/5 剩余实验（消融 / 压力 / ArguAna / NQ uniform 对照 / OPQ 近似 / Thm2 谱）。
协议与既有运行一致:
- C1 消融: 复用 pilot_truncation 的 load/recall_at_k/brute_rank/margin_importance
- 熵: rabitq_py 半整数网格（随机正交旋转 seed=0）+ 逐维码字 Shannon 熵
- uniform SQ: 随机正交旋转 + 逐维扫掠步长半整数网格（与 method Eq.(1) 一致）
- 评估: Recall@10（BEIR 暴力余弦协议）
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
import rabitq_py

def save(out, d):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(d, open(out, "w"), indent=1)
    print("SAVED", out, flush=True)

# ---------------- C1 on any dataset (perm/prefix/random/pca) ----------------
def run_c1(ds_dir, model, keeps, out):
    emb, qemb, qids, qrels, ids = load(ds_dir, model)
    d = emb.shape[1]
    print("C1", model, emb.shape, len(qids), flush=True)
    res = {"ds_dir": ds_dir, "model": model, "n_docs": len(emb), "d": d}
    ref = brute_rank(qemb, emb)
    res["reference_full"] = recall_at_k(ref, qids, qrels, ids)[0]
    imp = margin_importance(emb, qemb, qids, qrels, ids)
    mu = emb.mean(0)
    U, S, Vt = np.linalg.svd(emb - mu, full_matrices=False)
    for keep in keeps:
        m = max(1, int(d * keep))
        s = {}
        pfx = emb[:, :m]; qpfx = qemb[:, :m]
        s["prefix"] = recall_at_k(brute_rank(qpfx, pfx), qids, qrels, ids)[0]
        rng = np.random.default_rng(42)
        ridx = rng.permutation(d)[:m]
        s["random"] = recall_at_k(brute_rank(qemb[:, ridx], emb[:, ridx]), qids, qrels, ids)[0]
        Vm = Vt[:m]
        s["pca"] = recall_at_k(brute_rank(qemb @ Vm.T, emb @ Vm.T), qids, qrels, ids)[0]
        order = np.argsort(-imp)
        s["perm_margin"] = recall_at_k(brute_rank(qemb[:, order[:m]], emb[:, order[:m]]), qids, qrels, ids)[0]
        res["keep=%.3f" % keep] = s
        print("  keep=%.3f" % keep, {k: round(v, 4) for k, v in s.items()}, flush=True)
    save(out, res)

# ---------------- entropy (C3) ----------------
def code_entropy(emb_sub, B):
    """rabitq_py 协议: 随机正交旋转 + 半整数网格 → 逐维码字 Shannon 熵（bits）"""
    P = rabitq_py.random_orthogonal(emb_sub.shape[1], seed=0)
    rot = (emb_sub @ P).astype(np.float32)
    codes, _ = rabitq_py.quantize_Bbit(rot, B)
    Hs = []
    for j in range(codes.shape[1]):
        _, counts = np.unique(codes[:, j], return_counts=True)
        p = counts / counts.sum()
        Hs.append(float(-(p * np.log2(p)).sum()))
    return float(np.sum(Hs)), codes.shape[1] * B

def run_c3(ds_dir, model, Bs, out, subset=1000, seed=0):
    emb, *_ = load(ds_dir, model)
    if subset and subset < len(emb):
        rng = np.random.default_rng(seed)
        emb = emb[rng.choice(len(emb), subset, replace=False)]
    d = emb.shape[1]
    print("C3", model, "subset", len(emb), flush=True)
    res = {"ds_dir": ds_dir, "model": model, "subset": subset, "d": d}
    for B in Bs:
        h, fixed = code_entropy(emb, B)
        res["B=%d" % B] = {"fixed_bits": fixed, "entropy_bits": round(h, 2),
                           "byte_saving": round(1.0 - h / fixed, 4)}
        print("  B=%d saving=%.4f" % (B, res["B=%d" % B]["byte_saving"]), flush=True)
    save(out, res)

# ---------------- ablations: order criterion / block whitening ----------------
def block_whiten_margin(emb, qemb, qids, qrels, ids, k, calib_frac=0.7):
    """按相关性聚类把维度分 k 桶 → 桶内正交去相关 → 旋转域 margin 排序 → 截断"""
    N, d = emb.shape
    # 相关性嵌入: 每维 j → corr 向量（用子集估计 C 以省时）
    sub = emb[: min(3000, N)]
    C = np.corrcoef(sub.T)
    C = np.nan_to_num(C)
    # 简单谱聚类: 对 |C| 做 k-means（相关性嵌入空间）
    from scipy.cluster.vq import kmeans2
    cen, lab = kmeans2(np.abs(C), k, seed=0, minit="points")
    # 桶内正交去相关（SVD 基）→ 保持内积
    rot = np.zeros((d, d), np.float32)  # rot[m, j] = 新坐标 m 在旧维度 j 的系数
    new_dims = []
    for b in range(k):
        bidx = np.where(lab == b)[0]
        if len(bidx) == 0:
            continue
        Xb = emb[:, bidx]
        Ub, Sb, _ = np.linalg.svd(Xb - Xb.mean(0), full_matrices=False)
        for r, jj in enumerate(bidx):
            rot[bidx[0] + r, :] = 0
            rot[bidx[0] + r, bidx] = Ub[r]  # 桶内旋转: 新坐标 r = Σ_j Ub[r,j]·x_bidx[j]
    emb_r = emb @ rot.T
    qemb_r = qemb @ rot.T
    imp = margin_importance(emb_r, qemb_r, qids, qrels, ids, calib_frac=calib_frac)
    return imp, rot

def run_ablate(ds_dir, model, keeps, out, k_blocks=(2, 4)):
    emb, qemb, qids, qrels, ids = load(ds_dir, model)
    d = emb.shape[1]
    print("ABLATE", model, emb.shape, flush=True)
    res = {"ds_dir": ds_dir, "model": model, "d": d}
    ref = brute_rank(qemb, emb)
    res["reference_full"] = recall_at_k(ref, qids, qrels, ids)[0]
    # 基准: margin 置换（复算，保证同 seed 可对比）
    imp_margin = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=a.calib_frac)
    # 方差序
    var_order = np.argsort(-emb.var(0))
    res["orders"] = {}
    for keep in keeps:
        m = max(1, int(d * keep))
        s = {}
        s["prefix"] = recall_at_k(brute_rank(qemb[:, :m], emb[:, :m]), qids, qrels, ids)[0]
        rng = np.random.default_rng(42)
        ridx = rng.permutation(d)[:m]
        s["random"] = recall_at_k(brute_rank(qemb[:, ridx], emb[:, ridx]), qids, qrels, ids)[0]
        s["margin"] = recall_at_k(brute_rank(qemb[:, np.argsort(-imp_margin)[:m]], emb[:, np.argsort(-imp_margin)[:m]]), qids, qrels, ids)[0]
        s["variance"] = recall_at_k(brute_rank(qemb[:, var_order[:m]], emb[:, var_order[:m]]), qids, qrels, ids)[0]
        res["keep=%.3f" % keep] = s
        print("  keep=%.3f" % keep, {k: round(v, 4) for k, v in s.items()}, flush=True)
    # 块白化 + margin（k 桶）
    for k in k_blocks:
        imp_b, _ = block_whiten_margin(emb, qemb, qids, qrels, ids, k, calib_frac=a.calib_frac)
        for keep in keeps:
            m = max(1, int(d * keep))
            order = np.argsort(-imp_b)
            r = recall_at_k(brute_rank(qemb[:, order[:m]], emb[:, order[:m]]), qids, qrels, ids)[0]
            res.setdefault("keep=%.3f" % keep, {})["blockk%d_margin" % k] = r
            print("  keep=%.3f k=%d: %.4f" % (keep, k, r), flush=True)
    save(out, res)

# ---------------- calibration size sensitivity ----------------
def run_calib(ds_dir, model, keep, out, fracs=(0.1, 0.25, 0.5)):
    emb, qemb, qids, qrels, ids = load(ds_dir, model)
    d = emb.shape[1]
    res = {"ds_dir": ds_dir, "model": model, "keep": keep}
    m = max(1, int(d * keep))
    res["n_judged"] = sum(1 for q in qids if q in qrels)
    for f in fracs:
        imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=f)
        order = np.argsort(-imp)
        r = recall_at_k(brute_rank(qemb[:, order[:m]], emb[:, order[:m]]), qids, qrels, ids)[0]
        res["calib=%.2f" % f] = r
        print("  calib=%.2f (n=%d): %.4f" % (f, int(res["n_judged"] * f), r), flush=True)
    save(out, res)

# ---------------- stress: domain drift (scifact-calib order -> NQ eval) ----------------
def run_drift(calib_ds, eval_ds, model, keeps, out):
    emb_c, qemb_c, qids_c, qrels_c, ids_c = load(calib_ds, model)
    emb_e, qemb_e, qids_e, qrels_e, ids_e = load(eval_ds, model)
    assert emb_c.shape[1] == emb_e.shape[1]
    d = emb_c.shape[1]
    print("DRIFT", model, "calib", emb_c.shape, "eval", emb_e.shape, flush=True)
    res = {"calib_ds": calib_ds, "eval_ds": eval_ds, "model": model}
    imp = margin_importance(emb_c, qemb_c, qids_c, qrels_c, ids_c)  # 在 calib 集上估计
    # 对照: 在 eval 集自身上估计（上界）
    imp_e = margin_importance(emb_e, qemb_e, qids_e, qrels_e, ids_e)
    for keep in keeps:
        m = max(1, int(d * keep))
        s = {}
        s["prefix"] = recall_at_k(brute_rank(qemb_e[:, :m], emb_e[:, :m]), qids_e, qrels_e, ids_e)[0]
        order_c = np.argsort(-imp)
        s["perm_calib"] = recall_at_k(brute_rank(qemb_e[:, order_c[:m]], emb_e[:, order_c[:m]]), qids_e, qrels_e, ids_e)[0]
        order_e = np.argsort(-imp_e)
        s["perm_eval"] = recall_at_k(brute_rank(qemb_e[:, order_e[:m]], emb_e[:, order_e[:m]]), qids_e, qrels_e, ids_e)[0]
        res["keep=%.3f" % keep] = s
        print("  keep=%.3f" % keep, {k: round(v, 4) for k, v in s.items()}, flush=True)
    save(out, res)

# ---------------- NQ 2M uniform SQ 对照（最后一格） ----------------
def quantize_uniform_fast(emb, B, seed=0, block=262144):
    """随机正交旋转 + 逐维扫掠步长半整数网格（快速版，method Eq.(1) 协议）。
    返回重建向量（已 L2 归一化）。"""
    N, d = emb.shape
    P = rabitq_py.random_orthogonal(d, seed=seed)
    out = np.zeros_like(emb)
    # 旋转分批做（N 可能 2M，d=768: 一次 matmul 6GB 可行，仍分批稳妥）
    rot_blocks = []
    for i in range(0, N, block):
        rot_blocks.append((emb[i:i+block] @ P))
    L = 2 ** B
    half = (L - 1) / 2.0
    # 逐维扫掠步长: t_j 使该维语料 |v_j| 的量化 cos² 最大
    for j in range(d):
        col = np.concatenate([rb[:, j] for rb in rot_blocks])
        a = np.abs(col).astype(np.float64)
        best_t, best_cos2 = None, -1.0
        # 临界点 = k/|v|, k=1..L/2-1；逐维对全部语料统一扫掠（d 次扫掠，N·d 总开销）
        if B == 1:
            y = np.sign(col)
            dot = float((y * col).sum()); norm2 = float((y * y).sum())
            best_t = 1.0
        else:
            for k in range(1, L // 2):
                # 候选 t 的临界值在 k/max|a| 与 k/min|a| 之间二分即可；直接采样分位数
                for frac in (0.02, 0.1, 0.5, 0.9, 0.98):
                    t = k / (np.quantile(a, frac) + 1e-12)
                    y = np.floor(t * col) + 0.5
                    y = np.clip(y, -half, half)
                    dot = float((y * col).sum()); n2 = float((y * y).sum())
                    cos2 = dot * dot / (n2 + 1e-30)
                    if cos2 > best_cos2:
                        best_cos2, best_t = cos2, t
        # 用 best_t 编码该维
        pos = 0
        for rb in rot_blocks:
            nb = rb.shape[0]
            y = np.clip(np.floor(best_t * rb[:, j]) + 0.5, -half, half)
            out[pos:pos+nb, j] = y
            pos += nb
        if j % 64 == 0:
            print("    dim %d/%d t=%.4g cos2=%.4f" % (j, d, best_t, best_cos2), flush=True)
    out = out @ P.T
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out.astype(np.float32)

def run_nq_uniform(ds_dir, model, Bs, out, block=262144):
    emb, qemb, qids, qrels, ids = load(ds_dir, model)
    print("NQ-UNIFORM", model, emb.shape, len(qids), flush=True)
    res = {"ds_dir": ds_dir, "model": model, "n_docs": len(emb)}
    for B in Bs:
        recon = quantize_uniform_fast(emb, B, block=block)
        recs = []
        for i in range(0, len(qemb), 512):
            sim = qemb[i:i+512] @ recon.T
            top = np.argpartition(-sim, 10, axis=1)[:, :10]
            for t, j in enumerate(range(i, min(i+512, len(qemb)))):
                qid = qids[j]
                if qid not in qrels:
                    continue
                rel = qrels[qid]
                got = [str(ids[g]) for g in top[t] if g < len(ids)]
                recs.append(sum(1 for g in got if g in rel) / max(len(rel), 1))
        res["B=%d" % B] = float(np.mean(recs))
        print("  B=%d uniform recall@10 = %.4f" % (B, res["B=%d" % B]), flush=True)
        save(out, res)
    save(out, res)

# ---------------- OPQ 近似: 随机 Hadamard 旋转 + PQ ----------------
def hadamard_rotation(d):
    """随机构造 d×d 正交 Hadamard（d 补齐到 2 的幂）"""
    def hmat(n):
        if n == 1:
            return np.array([[1.0]])
        h = hmat(n // 2)
        return np.block([[h, h], [h, -h]])
    p = 1
    while p < d:
        p *= 2
    H = hmat(p) / np.sqrt(p)
    rng = np.random.default_rng(0)
    D = np.diag(rng.choice([-1.0, 1.0], p))
    s = rng.permutation(p)
    R = (D @ H)[s][:d, :d].astype(np.float32)
    return R

def run_opq_hadamard(ds_dir, model, ms, out):
    import faiss
    emb, qemb, qids, qrels, ids = load(ds_dir, model)
    d = emb.shape[1]
    print("OPQ-HADAMARD", model, emb.shape, flush=True)
    R = hadamard_rotation(d)
    x = (emb @ R).astype(np.float32)
    qx = (qemb @ R).astype(np.float32)
    res = {"ds_dir": ds_dir, "model": model, "note": "random Hadamard rotation + PQ (OPQ approximation)"}
    for m in ms:
        if d % m != 0:
            continue
        pq = faiss.ProductQuantizer(d, m, 8)
        pq.train(x)
        codes = pq.compute_codes(x)
        recon = pq.decode(codes)
        recon /= (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-12)
        recs = []
        for i in range(0, len(qemb), 1024):
            sim = qx[i:i+1024] @ recon.T
            top = np.argpartition(-sim, 10, axis=1)[:, :10]
            for t, j in enumerate(range(i, min(i+1024, len(qemb)))):
                qid = qids[j]
                if qid not in qrels:
                    continue
                rel = qrels[qid]
                got = [str(ids[g]) for g in top[t] if g < len(ids)]
                recs.append(sum(1 for g in got if g in rel) / max(len(rel), 1))
        res["OPQh_m%d" % m] = float(np.mean(recs))
        print("  OPQh_m%d = %.4f" % (m, res["OPQh_m%d" % m]), flush=True)
        save(out, res)

# ---------------- Thm2 谱拟合 ----------------
def run_thm2(ds_dir, models, keeps, out):
    res = {"ds_dir": ds_dir}
    for model in models:
        emb, qemb, qids, qrels, ids = load(ds_dir, model)
        d = emb.shape[1]
        C = np.cov(emb.T)
        w = np.linalg.eigvalsh(C)[::-1]
        w = w[w > 0]
        eff = float((w.sum() ** 2) / (w ** 2).sum())
        res[model] = {"d": d, "eff_rank": round(eff, 1),
                      "lambda1_ratio": round(float(w[0] / w.sum()), 4),
                      "top10_ratio": round(float(w[:10].sum() / w.sum()), 4),
                      "spectrum": [round(float(x), 8) for x in w]}
        print(model, "eff_rank=%.1f lambda1=%.4f top10=%.4f" % (eff, w[0]/w.sum(), w[:10].sum()/w.sum()), flush=True)
    save(out, res)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("c1"); p.add_argument("--ds_dir", required=True); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--keeps", nargs="+", type=float, default=[0.5, 0.25, 0.125])
    p = sub.add_parser("c3"); p.add_argument("--ds_dir", required=True); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--Bs", nargs="+", type=int, default=[1, 2, 4]); p.add_argument("--subset", type=int, default=1000)
    p = sub.add_parser("ablate"); p.add_argument("--ds_dir", required=True); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--keeps", nargs="+", type=float, default=[0.5, 0.25, 0.125]); p.add_argument("--k", nargs="+", type=int, default=[2, 4]); p.add_argument("--calib_frac", type=float, default=0.7)
    p = sub.add_parser("calib"); p.add_argument("--ds_dir", required=True); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--keep", type=float, default=0.125); p.add_argument("--fracs", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    p = sub.add_parser("drift"); p.add_argument("--calib_ds", required=True); p.add_argument("--eval_ds", required=True); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--keeps", nargs="+", type=float, default=[0.5, 0.25, 0.125])
    p = sub.add_parser("nq_uniform"); p.add_argument("--ds_dir", required=True); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--Bs", nargs="+", type=int, default=[1, 2])
    p = sub.add_parser("opq_had"); p.add_argument("--ds_dir", required=True); p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--ms", nargs="+", type=int, default=[48, 96, 192])
    p = sub.add_parser("thm2"); p.add_argument("--ds_dir", required=True); p.add_argument("--out", required=True); p.add_argument("--models", nargs="+", default=["e5", "bge", "mini", "bgem3", "qwen3"])
    a = ap.parse_args()

    if a.cmd == "c1":
        run_c1(a.ds_dir, a.model, a.keeps, a.out)
    elif a.cmd == "c3":
        run_c3(a.ds_dir, a.model, a.Bs, a.out, subset=a.subset)
    elif a.cmd == "ablate":
        run_ablate(a.ds_dir, a.model, a.keeps, a.out, k_blocks=tuple(a.k))
    elif a.cmd == "calib":
        run_calib(a.ds_dir, a.model, a.keep, a.out, fracs=tuple(a.fracs))
    elif a.cmd == "drift":
        run_drift(a.calib_ds, a.eval_ds, a.model, a.keeps, a.out)
    elif a.cmd == "nq_uniform":
        run_nq_uniform(a.ds_dir, a.model, a.Bs, a.out)
    elif a.cmd == "opq_had":
        run_opq_hadamard(a.ds_dir, a.model, a.ms, a.out)
    elif a.cmd == "thm2":
        run_thm2(a.ds_dir, a.models, None, a.out)
