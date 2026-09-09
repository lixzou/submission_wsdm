#!/usr/bin/env python3
"""从 gap_results + bootstrap JSON 生成 Table 1 的转置 LaTeX 表体。

行 = 方法(PQ/OPQ/ITQ/Scalar/TQ/RECO) × keep(50/75/87%), 列 = 三数据集模型并排。
自动加: 每格 best 加粗、RECO 显著胜 best baseline 加 † (p<=0.05)。
用法: python make_tab1.py <boot_json>  (cwd = experiments)
"""
import json, sys

PANELS = [
    ("scifact", [("e5", "E5-base-v2"), ("bge", "BGE-base-en-v1.5"),
                 ("mini", "All-MiniLM-L6-v2"), ("bgem3", "BGE-M3 dense"),
                 ("qwen3", "Qwen3-Embedding-0.6B")]),
    ("fiqa", [("e5", "E5-base-v2"), ("bge", "BGE-base-en-v1.5"),
             ("mini", "All-MiniLM-L6-v2"), ("bgem3", "BGE-M3 dense"),
             ("qwen3", "Qwen3-Embedding-0.6B")]),
    ("nfcorpus", [("e5", "E5-base-v2"), ("bge", "BGE-base-en-v1.5"),
                  ("mini", "All-MiniLM-L6-v2"), ("bgem3", "BGE-M3 dense"),
                  ("qwen3", "Qwen3-Embedding-0.6B")]),
]
METHODS = ["pq", "opq", "itq", "scalar", "tq", "reco"]
KEEPS = [0.5, 0.75, 0.875]
DIMS = {"e5": 768, "bge": 768, "mini": 384, "bgem3": 1024, "qwen3": 1024}
RES = {
    "scifact": {m: "results/runs/train_calib/scifact/%s.json" % m for m in ["e5","bge","mini","bgem3","qwen3"]},
    "fiqa": {m: "results/runs/train_calib/fiqa/%s.json" % m for m in ["e5","bge","mini","bgem3","qwen3"]},
    "nfcorpus": {m: "results/runs/train_calib/nfcorpus/%s.json" % m for m in ["e5","bge","mini","bgem3","qwen3"]},

}
KEEP_KEY = {0.5: "keep=0.500", 0.75: "keep=0.750", 0.875: "keep=0.875"}
METHOD_LABEL = {"pq": "PQ", "opq": "OPQ", "itq": "ITQ", "scalar": "Scalar",
                "tq": "TQ", "reco": r"\reco"}
KEEP_LABEL = {0.5: "50\\%", 0.75: "75\\%", 0.875: "87\\%"}
# boot JSON key 用 ds_dir basename (小写), 面板标签 → 数据集名
PANEL2DS = {"scifact": "scifact", "fiqa": "fiqa", "nfcorpus": "nfcorpus"}


def load_values():
    """val[panel][mkey][keep] -> {method: float}"""
    val = {}
    for panel, models in PANELS:
        val[panel] = {}
        for mkey, _ in models:
            d = json.load(open(RES[panel][mkey]))
            val[panel][mkey] = {}
            for keep, kk in KEEP_KEY.items():
                cell = d.get(kk, {})
                val[panel][mkey][keep] = {m: cell[m] for m in METHODS
                                          if cell.get(m) is not None}
    return val


def main():
    boot = json.load(open(sys.argv[1]))
    val = load_values()
    cols = [(panel, mkey) for panel, models in PANELS for mkey, _ in models]

    # 每格 best (含并列)
    best_cell = {}
    for panel, mkey in cols:
        for keep in KEEPS:
            cells = val[panel][mkey][keep]
            if not cells:
                continue
            mx = max(cells.values())
            best_cell[(panel, mkey, keep)] = [m for m, v in cells.items()
                                              if abs(v - mx) < 1e-9]

    # 显著性: RECO 胜 best baseline 且 p<=0.05
    sig = {}
    for key, pk in boot.items():
        if pk["reco"] > pk["best_base_recall"] and pk["p"] <= 0.05:
            sig[key] = pk["p"]

    def full_name(panel, mkey):
        return [fn for p, models in PANELS if p == panel for mk, fn in models
                if mk == mkey][0]

    n_meth = len(METHODS)
    lines = []
    # 三数据集并排: 行 = 模型×keep(三数据集共用), 列 = 每数据集一个方法组(6列)
    lines.append(r"\begin{tabular}{ll" + "c" * (n_meth * len(PANELS)) + "}")
    lines.append(r"\toprule")
    # 数据集面板标题
    hdr = ["", ""]
    for panel, _ in PANELS:
        hdr.append(r"\multicolumn{%d}{c}{\textbf{%s}}" % (n_meth, panel))
    lines.append(" & ".join(hdr) + r" \\")
    # 方法表头 (每组重复)
    hdr = [r"\textbf{Model}", r"\textbf{Keep}"]
    for _ in PANELS:
        hdr.append(r"\textbf{PQ} & \textbf{OPQ} & \textbf{ITQ} & \textbf{Scl} & \textbf{TQ} & \textbf{\reco}")
    lines.append(" & ".join(hdr) + r" \\")
    lines.append(r"\midrule")
    # 行: 模型 × keep (模型名/keep 共用, 每数据集给方法值)
    for mi, (mkey, mname) in enumerate(PANELS[0][1]):
        for ki, keep in enumerate(KEEPS):
            row = []
            if ki == 0:
                row.append(r"\multirow{3}{*}{\shortstack{%s\\(%dd)}}" % (mname, DIMS[mkey]))
            else:
                row.append("")
            row.append(KEEP_LABEL[keep])
            for panel, _ in PANELS:
                cells = val[panel][mkey][keep]
                for meth in METHODS:
                    if meth not in cells:
                        row.append("--")
                        continue
                    s = "%.3f" % cells[meth]
                    if s.startswith("0."):
                        s = s[1:]
                    is_best = meth in best_cell.get((panel, mkey, keep), [])
                    key = f"{mkey}@{PANEL2DS[panel]}@keep{keep:.3f}"
                    mark = r"$^{\dagger}$" if (meth == "reco" and key in sig) else ""
                    if is_best:
                        row.append(r"\textbf{%s}%s" % (s, mark))
                    elif mark:
                        row.append(r"%s%s" % (s, mark))
                    else:
                        row.append(s)
            lines.append(" & ".join(row) + r" \\")
            if ki < len(KEEPS) - 1:
                lines.append(r"\cmidrule(lr){3-%d}" % (2 + n_meth * len(PANELS)))
        if mi < len(PANELS[0][1]) - 1:
            lines.append(r"\cmidrule(lr){2-%d}" % (2 + n_meth * len(PANELS)))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(main())
