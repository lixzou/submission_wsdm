#!/usr/bin/env python3
"""update_canonical.py — 从源结果文件重建 canonical_numbers.json（源文件 = 唯一真相）。

2026-08-17 确定性修复后: 所有 margin 依赖实验已用 sorted() 修复重跑（rerun_deterministic.sh）,
合成实验已用 synthetic_experiments.py 重建（各向同性噪声 0.2）。
本脚本把 34-key canonical 结构从源文件重新推导, 保证 canonical 永远是源文件的投影。
"""
import json, os

B = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(B)

def r4(x):
    return round(x[0] if isinstance(x, list) else x, 4)

def load(name):
    for d in (os.path.join(ROOT, "results", "canonical"),
              os.path.join(ROOT, "results", "exploratory"),
              os.path.join(ROOT, "results", "runs"), B):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(name)

canon = {}

# ---- C1: scifact 5 + nq 3 + arguana 3 ----
for ds, models in [("scifact", ["e5", "bge", "mini", "bgem3", "qwen3"]),
                   ("nq", ["e5", "bge", "mini"])]:
    for m in models:
        srcname = f"c1_{ds}_{m}_f1.json" if ds == "scifact" else f"c1_nq_{m}.json"
        d = load(srcname)
        canon[f"c1_{ds}_{m}"] = {
            k: {
                "prefix": r4(d[k]["prefix"]),
                "random": r4(d[k]["random"]),
                "pca": r4(d[k]["pca"]),
                "perm_margin": r4(d[k]["perm_margin"]),
            } for k in d if k.startswith("keep=")
        }
for m in ["e5", "bge", "mini"]:
    d = load(f"results/runs/c1_arguana_{m}.json")
    entry = {"reference_full": r4(d["reference_full"])}
    for k in d:
        if k.startswith("keep="):
            entry[k] = {
                "prefix": r4(d[k]["prefix"]), "random": r4(d[k]["random"]),
                "pca": r4(d[k]["pca"]), "perm_margin": r4(d[k]["perm_margin"]),
            }
    canon[f"c1_arguana_{m}"] = entry

# ---- C1 NQ 5k 1024d（新增, judged-only 评测）----
for m in ["bgem3", "qwen3"]:
    d = load(f"results/runs/c1_nq5k_{m}.json")
    entry = {"reference_full": r4(d["reference_full"])}
    for k in d:
        if k.startswith("keep="):
            entry[k] = {"prefix": r4(d[k]["prefix"]), "pca": r4(d[k]["pca"]),
                        "perm_margin": r4(d[k]["perm_margin"])}
    canon[f"c1_nq5k_{m}"] = entry

# ---- C2: uniform vs best tilt（pilot_2b 新格式, best = 三变体最大）----
def c2_from_2b(d, budgets):
    out = {}
    for cx, ab in budgets:
        if ab not in d:
            continue
        out[cx] = {
            "uniform": r4(d[ab]["uniform"]),
            "best_tilt": r4(max(d[ab]["ratc_bucket"], d[ab]["margin_block"], d[ab]["var_block"])),
        }
    return out

for m in ["e5", "bge", "mini", "bgem3", "qwen3"]:
    d = load(f"pilot_2b_scifact_{m}.json")
    budgets = [("8x", "avg_bits=4"), ("16x", "avg_bits=2"), ("32x", "avg_bits=1")] if m == "e5" \
        else [("8x", "avg_bits=4"), ("16x", "avg_bits=2")]
    canon[f"c2_{m}_scifact"] = {**c2_from_2b(d, budgets),
                                "source": f"pilot_2b_scifact_{m}.json"}
for m in ["e5", "bge"]:
    d = load(f"pilot_2b_nq5k_{m}.json")
    canon[f"c2_{m}_nq5k"] = {**c2_from_2b(d, [("8x", "avg_bits=4"), ("16x", "avg_bits=2")]),
                             "source": f"pilot_2b_nq5k_{m}.json"}

# ---- c2_summary: 12 合成 + 15 真实 ----
syn = load("theory_eta_scan.json")
syn_wins = sum(1 for s in syn.values() if s["uniform"] >= s["tilt"])
real_settings = []
real_settings += [(f"c2_e5_scifact", cx) for cx in ["8x", "16x", "32x"]]
for m in ["bge", "mini", "bgem3", "qwen3"]:
    real_settings += [(f"c2_{m}_scifact", cx) for cx in ["8x", "16x"]]
for m in ["e5", "bge"]:
    real_settings += [(f"c2_{m}_nq5k", cx) for cx in ["8x", "16x"]]
wins = ties = losses = 0
exceptions = []
for key, cx in real_settings:
    u, t = canon[key][cx]["uniform"], canon[key][cx]["best_tilt"]
    if u > t + 1e-4:
        wins += 1
    elif abs(u - t) <= 1e-4:
        ties += 1
        exceptions.append(f"{key} {cx}: tie {u} vs {t}")
    else:
        losses += 1
        exceptions.append(f"{key} {cx}: tilt {t} vs uniform {u} (+{round((t-u)*1000)/1000})")
canon["c2_summary"] = {
    "settings_total": len(syn) + len(real_settings),
    "synthetic": len(syn), "real": len(real_settings),
    "uniform_wins": syn_wins + wins, "ties": ties, "losses": losses,
    "exceptions": exceptions,
    "synthetic_gaps_pp": f"{min(s['uniform']-s['tilt'] for s in syn.values())*100:.1f} to {max(s['uniform']-s['tilt'] for s in syn.values())*100:.1f}",
}

# ---- C3 熵（确定性, 未受 margin 影响: 直接复制旧 canonical）----
old = load("canonical_numbers.json")
for k in ["c3_scifact", "c3_big", "c3_arguana_e5", "c3_arguana_bge", "c3_arguana_mini",
          "pq_scifact", "pq_scifact_e5_bytealigned", "pq_scifact_bge_bytealigned",
          "pq_scifact_mini_bytealigned", "pq_nq2m_e5_bytealigned", "pq_nq2m_bge_bytealigned",
          "pq_nq2m_mini_bytealigned", "drift_scifact_nq_e5", "nq_uniform_e5",
          "opq_had_scifact"]:
    canon[k] = old[k]

# ---- drift: 确定性重跑后更新 ----
d = load("results/runs/drift_scifact_nq_e5.json")
canon["drift_scifact_nq_e5"] = {
    k: {"prefix": r4(d[k]["prefix"]), "perm_scifact_calib": r4(d[k]["perm_calib"]),
        "perm_nq_calib": r4(d[k]["perm_eval"])}
    for k in d if k.startswith("keep=")
}
canon["drift_scifact_nq_e5"]["source"] = "results/runs/drift_scifact_nq_e5.json (deterministic rerun)"

# ---- 元数据 ----
canon["_meta"] = {
    "updated": "2026-08-17 deterministic-fix rerun",
    "margin_fix": "pilot_truncation.py/run_bootstrap.py rel_docs sorted() (PYTHONHASHSEED bug)",
    "synthetic_rebuilt": "synthetic_experiments.py (isotropic query noise 0.2, d=256, n_docs=5000, n_queries=200)",
    "heldout": "heldout_check.py rerun: +33.6pp",
}

json.dump(canon, open(os.path.join(ROOT, "results", "canonical", "canonical_numbers.json"), "w"), indent=1)
print(f"canonical_numbers.json rebuilt: {len(canon)} keys")
print("c2_summary:", json.dumps(canon["c2_summary"], indent=1))
