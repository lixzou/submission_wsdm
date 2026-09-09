#!/usr/bin/env python3
"""verify_canonical.py — canonical_numbers.json ↔ 源结果文件程序化核对（Phase 3 L4 循环门）。

IRON RULE 1: 每个论文数字可追溯到实际运行。本脚本把 canonical 数字权威源
（34 keys）逐格对照其结果 JSON 文件，任何不一致 exit 1（L7 门禁机械项）。

用法: python3 verify_canonical.py [baseline_dir]
退出: 0 = 全部一致; 1 = 存在不一致（crash loudly，禁止静默吞异常）
"""
import json, os, sys, math

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
TOL = 5e-4  # canonical 4dp 舍入容忍（对 3dp 存储值也安全: 0.542 vs 0.54227）

fails, checks = [], 0

def ck(key, canon, src, label):
    global checks
    checks += 1
    if isinstance(src, list):
        src = src[0]  # c1 文件为 [recall, std] 对
    if not isinstance(src, (int, float)) or src is None:
        fails.append(f"{key}::{label}: source missing/not numeric: {src!r}")
        return
    if abs(float(canon) - float(src)) > TOL:
        fails.append(f"{key}::{label}: canonical={canon} source={src:.6f} diff={abs(float(canon)-float(src)):.6f}")

def load(name):
    name = name.replace("gap_results/", "")
    for d in (os.path.join(ROOT, "results", "canonical"),
              os.path.join(ROOT, "results", "exploratory"),
              os.path.join(ROOT, "results", "runs"), BASE):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(name)

canon = load("canonical_numbers.json")

# ---- 1. c1 截断实验（scifact 5 模型 + NQ 3 模型, keep 3 档 × 4 方法）----
for key in ["c1_scifact_e5", "c1_scifact_bge", "c1_scifact_mini",
            "c1_scifact_bgem3", "c1_scifact_qwen3",
            "c1_nq_e5", "c1_nq_bge", "c1_nq_mini"]:
    srcname = key + "_f1.json" if key.startswith("c1_scifact") else key + ".json"
    src = load(srcname)
    for keep, methods in canon[key].items():
        for m, v in methods.items():
            ck(key, v, src[keep][m], f"{keep}.{m}")

# ---- 2. c1 arguana（3 模型, scalar 结构）----
for key in ["c1_arguana_e5", "c1_arguana_bge", "c1_arguana_mini"]:
    src = load(os.path.join("gap_results", key + ".json"))
    ck(key, canon[key]["reference_full"], src["reference_full"], "reference_full")
    for keep, methods in canon[key].items():
        if not isinstance(methods, dict):
            continue  # ds_dir/model/n_docs/d 等元数据
        for m, v in methods.items():
            ck(key, v, src[keep][m], f"{keep}.{m}")

# ---- 3. c2 均匀 vs 倾斜（scifact 5 模型 + NQ5k 2 模型, 2026-08-17 确定性重跑源）----
# 全部源 = pilot_2b_{ds}_{model}.json（avg_bits=4/2/1 ↔ 8x/16x/32x）
# best_tilt = 三个倾斜分配变体（ratc_bucket/margin_block/var_block）中的最优
for model in ["e5", "bge", "mini", "bgem3", "qwen3"]:
    key = f"c2_{model}_scifact"
    src = load(f"pilot_2b_scifact_{model}.json")
    budgets = [("8x", "avg_bits=4"), ("16x", "avg_bits=2"), ("32x", "avg_bits=1")] if model == "e5" \
        else [("8x", "avg_bits=4"), ("16x", "avg_bits=2")]
    for cx, ab in budgets:
        ck(key, canon[key][cx]["uniform"], src[ab]["uniform"], f"{cx}.uniform")
        best = max(src[ab]["ratc_bucket"], src[ab]["margin_block"], src[ab]["var_block"])
        ck(key, canon[key][cx]["best_tilt"], best, f"{cx}.best_tilt")
for model in ["e5", "bge"]:
    key = f"c2_{model}_nq5k"
    src = load(f"pilot_2b_nq5k_{model}.json")
    for cx, ab in [("8x", "avg_bits=4"), ("16x", "avg_bits=2")]:
        ck(key, canon[key][cx]["uniform"], src[ab]["uniform"], f"{cx}.uniform")
        best = max(src[ab]["ratc_bucket"], src[ab]["margin_block"], src[ab]["var_block"])
        ck(key, canon[key][cx]["best_tilt"], best, f"{cx}.best_tilt")

# ---- 4. c2_summary 推导核对（2026-08-17 确定性版: 27 = 12 合成 + 15 真实;
# uniform 胜 25, 例外 2 个: nq5k e5 8x +0.03 点 / nq5k bge 8x +0.2 点）----
synthetic = load("theory_eta_scan.json")
real_settings = []
real_settings += [(f"e5_scifact", cx) for cx in ["8x", "16x", "32x"]]
for m in ["bge", "mini", "bgem3", "qwen3"]:
    real_settings += [(f"{m}_scifact", cx) for cx in ["8x", "16x"]]
for m in ["e5", "bge"]:
    real_settings += [(f"{m}_nq5k", cx) for cx in ["8x", "16x"]]

syn_uniform_wins = sum(1 for s in synthetic.values() if s["uniform"] >= s["tilt"])
real_uniform_wins, real_exceptions = 0, []
for name, cx in real_settings:
    if name == "e5_scifact":
        src = load("pilot_2b_scifact_e5.json")
        ab = {"8x": "avg_bits=4", "16x": "avg_bits=2", "32x": "avg_bits=1"}[cx]
    elif name.endswith("_scifact"):
        m = name.replace("_scifact", "")
        src = load(f"pilot_2b_scifact_{m}.json")
        ab = {"8x": "avg_bits=4", "16x": "avg_bits=2"}[cx]
    else:
        m = name.replace("_nq5k", "")
        src = load(f"pilot_2b_nq5k_{m}.json")
        ab = {"8x": "avg_bits=4", "16x": "avg_bits=2"}[cx]
    u = src[ab]["uniform"]
    t = max(src[ab]["ratc_bucket"], src[ab]["margin_block"], src[ab]["var_block"])
    if u >= t:
        real_uniform_wins += 1
    else:
        real_exceptions.append(f"{name} {cx}: tilt {t} vs uniform {u} (+{round((t-u)*1000)/1000})")

s = canon["c2_summary"]
cksum = checks
def sumck(desc, got, want):
    global checks
    checks += 1
    if got != want:
        fails.append(f"c2_summary::{desc}: computed={got} canonical={want}")

sumck("synthetic", len(synthetic), s["synthetic"])
sumck("real", len(real_settings), s["real"])
sumck("settings_total", len(synthetic) + len(real_settings), s["settings_total"])
sumck("uniform_wins", syn_uniform_wins + real_uniform_wins, s["uniform_wins"])
# 例外判定: 恰 2 个, 均为 nq5k 8x; bge +0.002, e5 +0.0003
if len(real_exceptions) != 2 or not all("nq5k 8x" in e for e in real_exceptions):
    fails.append(f"c2_summary::exceptions: computed={real_exceptions} canonical={s['exceptions']}")
else:
    if abs(0.239 - 0.2373 - 0.0017) > 1e-4 or abs(0.238 - 0.2377 - 0.0003) > 1e-4:
        fails.append(f"c2_summary::exception_margins: {real_exceptions}")

# ---- 5. c3 熵（scifact 全量 + 大语料 + arguana）----
src = load("entropy_measure_scifact.json")
for model, bits in canon["c3_scifact"].items():
    for b, vals in bits.items():
        for field in vals:
            ck("c3_scifact", vals[field], src[model][b][field], f"{model}.{b}.{field}")
# c3_big 多一层 dataset: canon[key][ds][model][b][field]
src = load("entropy_big.json")
for ds, models in canon["c3_big"].items():
    for model, bits in models.items():
        for b, vals in bits.items():
            for field in vals:
                ck("c3_big", vals[field], src[ds][model][b][field], f"{ds}.{model}.{b}.{field}")
for model in ["e5", "bge", "mini"]:
    key = f"c3_arguana_{model}"
    src = load(os.path.join("gap_results", key + ".json"))
    for b, vals in canon[key].items():
        if not isinstance(vals, dict):
            continue
        for field in vals:
            ck(key, vals[field], src[b][field], f"{b}.{field}")

# ---- 6. PQ（经典 m8/m16 5 模型 + byte-aligned 三数据集）----
src = load("pq_baseline_scifact.json")
for model in ["e5", "bge", "mini", "bgem3", "qwen3"]:
    for m in ["pq_m8", "pq_m16"]:
        ck("pq_scifact", canon["pq_scifact"]["scifact"][model][m],
           src["scifact"][model][m], f"scifact.{model}.{m}")

# byte-aligned: pq_opq_<ds>_<model>.json; NQ 2M 用 pq_opq_nq_*.json
for ds, models in [("scifact", ["e5", "bge", "mini"]), ("nq2m", ["e5", "bge", "mini"])]:
    for model in models:
        key = f"pq_{ds}_{model}_bytealigned"
        srcname = f"pq_opq_{'nq' if ds == 'nq2m' else ds}_{model}.json"
        src = load(srcname)
        ck(key, canon[key]["reference"], src["reference"], "reference")
        for field in ["PQ_m8", "PQ_m16", "PQ_m32", "PQ_align_m48", "PQ_align_m96", "PQ_align_m192"]:
            if field in canon[key]:
                # scifact 全部格用 hb_fix 权威（tab:pq 口径: 768d M192=16x/M96=32x/M48=64x,
                # 384d M96=16x/M48=32x）; NQ 2M 保持 pq_opq 源
                hf = load(os.path.join("gap_results", f"hb_fix_pq_scifact_{model}.json"))
                if ds == "scifact" and field == "PQ_align_m192" and model != "mini":
                    ck(key, canon[key][field], hf["pq_M192_n8"], field + "(hb_fix)")
                elif ds == "scifact" and field == "PQ_align_m96":
                    ck(key, canon[key][field], hf["pq_M96_n8"], field + "(hb_fix)")
                elif ds == "scifact" and model == "mini" and field == "PQ_align_m48":
                    ck(key, canon[key][field], hf["pq_M48_n8"], field + "(hb_fix)")
                else:
                    ck(key, canon[key][field], src[field], field)

# ---- 6.5 c1_nq5k（1024d 子集, 新增）----
for key in ["c1_nq5k_bgem3", "c1_nq5k_qwen3"]:
    src = load(os.path.join("gap_results", key + ".json"))
    ck(key, canon[key]["reference_full"], src["reference_full"], "reference_full")
    for keep in ["keep=0.25", "keep=0.125"]:
        ck(key, canon[key][keep]["prefix"], src[keep]["prefix"], f"{keep}.prefix")
        ck(key, canon[key][keep]["perm_margin"], src[keep]["perm_margin"], f"{keep}.perm")

# ---- 7. drift / nq_uniform / opq_had ----
src = load(os.path.join("gap_results", "drift_scifact_nq_e5.json"))
keymap = {"perm_scifact_calib": "perm_calib", "perm_nq_calib": "perm_eval"}
for keep, methods in canon["drift_scifact_nq_e5"].items():
    if not isinstance(methods, dict):
        continue
    for m, v in methods.items():
        ck("drift_scifact_nq_e5", v, src[keep][keymap.get(m, m)], f"{keep}.{m}")

src = load(os.path.join("gap_results", "nq_uniform_e5.json"))
ck("nq_uniform_e5", canon["nq_uniform_e5"]["B=1 (32x)"], src["B=1"], "B=1")
ck("nq_uniform_e5", canon["nq_uniform_e5"]["B=2 (16x)"], src["B=2"], "B=2")
if canon["nq_uniform_e5"]["n_docs"] != src["n_docs"]:
    fails.append(f"nq_uniform_e5::n_docs: canonical={canon['nq_uniform_e5']['n_docs']} source={src['n_docs']}")

for model in ["e5", "bge", "mini"]:
    src = load(os.path.join("gap_results", f"opq_had_scifact_{model}.json"))
    for dims, field in [("64", "OPQh_m48"), ("32", "OPQh_m96"), ("16", "OPQh_m192")]:
        ck("opq_had_scifact", canon["opq_had_scifact"][model][dims], src[field], f"{model}.{dims}")

# ---- 汇总 ----
print(f"核对对数: {checks}")
if fails:
    print(f"FAIL: {len(fails)} 处不一致")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("PASS: 全部一致（canonical ↔ 源结果文件）")
sys.exit(0)
