#!/usr/bin/env python3
"""verify_paper_numbers.py — 论文正文每个数字 ↔ canonical/源文件逐格核对（Q52/L7 5.11）。

方法: 从 tex 提取全部数字 token → 与"已知数字池"（canonical_numbers.json +
全部源结果 JSON 的 1-4dp 舍入形式 + 手工登记常量）比对 → 池外数字列出手工
复核清单。exit 1 = 存在池外数字且未在白名单（白名单必须是逐项审过的）。

用法: python3 verify_paper_numbers.py paper_dir baseline_dir
"""
import json, os, re, sys

PAPER = sys.argv[1] if len(sys.argv) > 1 else "/ssd1/zoulixin/tencent_previous_compression/paper/wsdm"
BASE = sys.argv[2] if len(sys.argv) > 2 else "/ssd1/zoulixin/tencent_previous_compression/experiments"

tex = open(os.path.join(PAPER, "main.tex")).read()
# 剔除注释
tex = re.sub(r"(?<!\\)%.*", "", tex)
# 剔除 tikz / tabular 环境（坐标与列格式不是正文数字）
for env in ["tikzpicture", "tabular", "algorithm", "table", "figure"]:
    tex = re.sub(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", " ", tex, flags=re.S)
# 剔除 \cite/\citep/\citet/\ref/\label/\eqref 命令（引用键年份/label 冒号不是正文数字）
tex = re.sub(r"\\(cite[a-z]*|ref|label|eqref|includegraphics)\*?(\[[^\]]*\])?\{[^}]*\}", " ", tex)
# preamble 剔除（\documentclass 到 \begin{document} 之间: 惩罚值/字体等非 claim 数字）
tex = re.sub(r"\\documentclass.*?\\begin\{document\}", " ", tex, flags=re.S)
# 千分位 {\,} 与 {,} 去除: 5{,}183 -> 5183
tex = re.sub(r"\{[,]\}", "", tex)
tex = re.sub(r"\{[,]\s*", "", tex)

# ---- 数字池 ----
pool = set()

def add_forms(v):
    for f in [v, round(v, 1), round(v, 2), round(v, 3), round(v, 4)]:
        s = f"{f:.4f}".rstrip("0").rstrip(".")
        pool.add(s)
        if 0 < f < 1:
            pool.add(s.lstrip("0"))  # .151 形式
        if 0 < f <= 1:
            pool.add(f"{f*100:.2f}".rstrip("0").rstrip("."))  # 百分比形式 49.6
    if 0 <= v <= 1:
        # 3dp 固定形式（.180/.260 尾零保留, 表格同款）
        pool.add(f"{v:.3f}")
        pool.add(f"{v:.3f}".lstrip("0"))

def collect(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            collect(v, path + "/" + str(k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            collect(v, path + f"[{i}]")
    elif isinstance(d, (int, float)):
        if not isinstance(d, bool):
            add_forms(float(d))
    elif isinstance(d, str):
        for m in re.findall(r"\d+\.\d+", d):
            add_forms(float(m))  # paper_reported_values 字符串里的数值（SAQ 曲线等）

SKIP_DIRS = {"hf_cache", "data", "env", "__pycache__"}
for root, dirs, files in os.walk(BASE):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for fn in files:
        if fn.endswith(".json"):
            try:
                with open(os.path.join(root, fn)) as f:
                    collect(json.load(f))
            except Exception as e:
                print(f"  [warn] {fn}: {e}")

# 手工登记常量（逐项审计过的: 维度/文档数/算术推导/论文协议数字）
manual = [
    "768", "384", "1024", "512", "128", "96", "300", "3452", "1406", "5000", "2000000",
    "5183", "8674", "24576", "12288", "6144", "4096", "2048", "3072", "1536", "500",
    "1000", "250", "150", "30", "100", "200", "2", "3", "4", "5", "8", "10", "16", "25", "32", "50",
    "64", "12", "27", "26", "1", "0", "5.2", "50.3", "33", "9", "63",
    # 推导/组合量
    "35.8", ".151", ".508", ".837", "57.1", "1.78", "49.6", "44.0", "6.9",
    "4.1", "2.7", "2.4", "0.2", "8.4", "1.0", "3.0",
]
for m in manual:
    pool.add(m)
# 常见百分比/倍数后缀形式（数字+符号）
for base in list(pool):
    pool.add(base + "%")
    pool.add(base + "x")

# 白名单（逐项审计: 数字 + 来源 + 理由）
WHITELIST = {
    "256": "设计算术: per-corpus 存储 $4d+8d+256d$ bytes（256 质心码本）",
    "110": "模型事实: E5-base-v2 110M 参数（官方）",
    "568": "模型事实: BGE-M3 dense 568M 参数（官方）",
    "503": "硬件事实: 本机 503GB RAM",
    "200": "SAQ 官方协议参数 nprobe=200",
    ".137": "SAQ 官方锚点（GPU06 官方二进制, baseline_anchors.md）",
    ".195": "SAQ 官方锚点（同上）",
    ".311": "SAQ 官方锚点（同上）",
    ".534": "SAQ 官方锚点（同上）",
    "5.75": "RaBitQ 引理常数 c_ε=5.75（paper_data_map.md 原文核对）",
    "0.999": "Extended-RaBitQ Remark1 >99.9% 概率（issue I25 原文核对）",
    "2.0": "算术: 熵乘数 1/(1-0.496)=1.98≈2.0（scifact, 与 57x=32×1.78 同式）",
    # 2026-08-17 确定性重跑后的推导量（逐项手算核对）
    "33.1": "算术: ArguAna e5 12.5% 0.5484-0.2176=0.3307（c1_arguana_e5.json）",
    "20.5": "算术: spectrum α=2.0 quant 0.305-0.100=0.205（spectrum_sweep.json）",
    "19.6": "算术: calib 10% 0.3462-0.1506=0.1956（calib_scifact_e5.json）",
    "6.2": "算术: c2 e5 8x 0.8378-0.7759=0.0619（pilot_2b_scifact_e5.json）",
    "24.8": "算术: c2 e5 32x 0.7307-0.4822=0.2485（pilot_2b_scifact_e5.json）",
    "9.0": "算术: 兼容性最大增益 mini 25% LSQ 0.6426-0.5524=0.0902（hb_compat_mini.json）",
    "8.9": "算术: calib 网格 B=1 0.8142-0.7251=0.0891（beat_tq_e5.json）",
    "7.9": "算术: 中心化 B=1 0.8040-0.7251=0.0789（beat_tq_e5.json）",
    "14.0": "算术: e5 50% perm-PCA 差 0.6843-0.8247=-0.1404（2026-08-19 f1 校准+uncentered PCA 协议重算）",
    "24.0": "算术: NQ in-domain 增益 0.3188-0.0784=0.2404（drift_scifact_nq_e5.json, R32-6 修复）",
    "4.3": "算术: qwen3 50% perm-PCA 差 0.8526-0.8099=0.0427（2026-08-19 f1 重算, R32-19 修复）",
    "0.0": "实测: 中心化后熵节省归零（entropy_centering_*.json, 3 模型）",
    "9.7": "算术: Qwen3 100% 校准 0.7412-0.6439=0.0973（c1_scifact_f1_qwen3.json）",
    "35.3": "算术: E5 100% 校准 0.5038-0.1506=0.3532（c1_scifact_f1_e5.json）",
    "15.6": "算术: e5 100% 校准 perm-PCA 差 @50%（f1 派生）",
    "118": "实测: bgem3 有效秩 118（thm2_spectra.json）",
    "102": "实测: e5 有效秩 102（thm2_spectra.json）",
}

# ---- 提取论文数字 token ----
tokens = re.findall(r"(?<![\w.])\.\d+|\d+(?:\.\d+)?(?:e[+-]?\d+)?", tex)
outside = []
seen = {}
for t in tokens:
    # 归一化: 去前导零 / 保留原形
    cands = {t, t.lstrip("0") if t.startswith("0") else t, t.lstrip("."), t + "%", t + "x"}
    norm = t.replace(".", "")
    if not any(c in pool for c in cands) and not (norm.isdigit() and any(
            (c in pool) or (c.lstrip("0") in pool) for c in cands)):
        outside.append(t)
        seen[t] = seen.get(t, 0) + 1

# 上下文（复核用）: 白名单内 → 通过; 池外 → 列出待修
unlisted = {}
for t in set(outside):
    if t in WHITELIST:
        print(f"  [whitelist] {t!r}: {WHITELIST[t]}")
    else:
        unlisted[t] = seen[t]
print(f"论文数字 token 总数: {len(tokens)}; 池外且未白名单: {len(unlisted)} 种")
for t in sorted(unlisted, key=lambda x: -unlisted[x]):
    idxs = [m.start() for m in re.finditer(re.escape(t), tex)]
    ctx = tex[max(0, idxs[0]-60):idxs[0]+60].replace("\n", " ")
    print(f"  {t!r} ×{unlisted[t]}: ...{ctx}...")
if unlisted:
    sys.exit(1)
print("PASS: 所有数字 token 落在数字池或已审计白名单内")
