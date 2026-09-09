#!/usr/bin/env python3
"""perm_pca_quant_check.py — 截断+量化全管线: Perm vs PCA 的差距是否在量化后缩小/反超。

协议: 截断到 keep 维（prefix/perm/pca 三种选维）→ 该 m 维子空间 B=1 量化（随机旋转
+半整数符号, rabitq 协议）→ 归一化重建 → Recall@10。
对比: 纯截断（不量化）vs 截断+量化——看 PCA 的量化误差累积是否吃掉其混合优势。
"""
import argparse, json
import numpy as np
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance
from rabitq_py import random_orthogonal, quantize_Bbit


def recall_of(qemb, o, qids, qrels, ids):
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    return recall_at_k(brute_rank(qemb.astype(np.float32), o.astype(np.float32)), qids, qrels, ids)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib_frac", type=float, default=1.0)
    args = ap.parse_args()
    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    ref = recall_at_k(brute_rank(qemb, emb), qids, qrels, ids)[0]
    res = {"ds_dir": args.ds_dir, "model": args.model, "d": d, "reference": round(ref, 4)}
    imp = margin_importance(emb, qemb, qids, qrels, ids, calib_frac=args.calib_frac)
    mu = emb.mean(0)
    U, S, Vt = np.linalg.svd(emb - mu, full_matrices=False)
    perm_order = np.argsort(-imp)
    for keep in [0.25, 0.125]:
        m = max(1, int(d * keep))
        # 三种选维
        orders = {
            "prefix": np.arange(d)[:m],
            "perm": perm_order[:m],
            "pca": None,  # PCA 用投影
        }
        P = random_orthogonal(m, seed=0)
        for name in ["prefix", "perm", "pca"]:
            if name == "pca":
                Vm = Vt[:m]
                sel_emb = emb @ Vm.T
                sel_q = qemb @ Vm.T
            else:
                sel = orders[name]
                sel_emb = emb[:, sel]
                sel_q = qemb[:, sel]
            # 纯截断
            r_raw = recall_of(sel_q.astype(np.float32), sel_emb, qids, qrels, ids)
            # 截断+量化 B=1
            rot = (sel_emb @ P).astype(np.float32)
            codes, _ = quantize_Bbit(rot, 1)
            y = codes.astype(np.float64) - 0.5
            o = (P @ y.T).T
            r_quant = recall_of(sel_q.astype(np.float32), o, qids, qrels, ids)
            res[f"keep={keep}_{name}_raw"] = round(r_raw, 4)
            res[f"keep={keep}_{name}_quant"] = round(r_quant, 4)
            print(f"keep={keep} {name}: raw {r_raw:.4f} -> quant {r_quant:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
