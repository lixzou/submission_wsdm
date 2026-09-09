#!/usr/bin/env python3
"""heldout_check.py — 校准/评估查询不重叠的 held-out 验证（Issue 05 关闭实验, 2026-08-17 重建）。

协议: judged 查询前 70% 校准 margin 重要性, 剩余 30% 上评估 perm vs prefix @12.5% keep。
2026-08-17 确定性修复后重建（原脚本未留存, 原 JSON 数字受 PYTHONHASHSEED 影响不可复现）。
"""
import argparse, json, os
import numpy as np
from pilot_truncation import load, recall_at_k, brute_rank, margin_importance

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=0.125)
    ap.add_argument("--calib_frac", type=float, default=0.7)
    args = ap.parse_args()

    emb, qemb, qids, qrels, ids = load(args.ds_dir, args.model)
    d = emb.shape[1]
    judged = [q for q in qids if q in qrels]
    n_cal = int(len(judged) * args.calib_frac)
    cal_q = judged[:n_cal]
    held_q = judged[n_cal:]
    assert held_q, "held-out set empty"

    # 校准集 = 前 n_cal 个 judged 查询（与 margin_importance 的 judged[:n_cal] 一致）
    import pilot_truncation as pt
    imp = pt.margin_importance(emb, qemb, qids, qrels, ids, calib_frac=args.calib_frac)
    order = np.argsort(-imp)
    m = max(1, int(d * args.keep))

    # 评估集 = held_q（其余查询屏蔽）
    qmask = np.array([q in set(held_q) for q in qids])
    hqids = [q for q in qids if q in set(held_q)]
    hqemb = qemb[qmask]
    ref = recall_at_k(brute_rank(hqemb, emb), hqids, qrels, ids)[0]
    pfx = recall_at_k(brute_rank(hqemb[:, :m], emb[:, :m]), hqids, qrels, ids)[0]
    perm = recall_at_k(brute_rank(hqemb[:, order[:m]], emb[:, order[:m]]), hqids, qrels, ids)[0]
    res = {
        "heldout_30pct": {
            "n_queries": len(held_q), "ref": round(ref, 4),
            "prefix": round(pfx, 4), "perm": round(perm, 4),
            "delta_pp": round((perm - pfx) * 100, 1),
        }
    }
    json.dump(res, open(args.out, "w"), indent=1)
    print("HELDOUT", args.model, res, flush=True)

if __name__ == "__main__":
    main()
