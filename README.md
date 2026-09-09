# experiments — 实验代码

论文提交版说明（详尽的工作区文档不随附）。

- `scripts/`    实验代码（同一目录扁平存放，互相 import）
- `results/`    实验结果：`canonical/` 论文数字权威源；`runs/` 各实验原始结果；
                `exploratory/` 早期结果；`legacy/` 历史产物（不用于核对）
- `MANIFEST.md` 论文表/图 ↔ 脚本/结果 的对应
- `docs/`       环境依赖清单 + 复现说明

运行：在 `experiments/` 根目录执行 `python3 scripts/<脚本>`，数据路径用
`--ds_dir` 传入。数据与第三方基线源码体积较大，未随附；重建入口见
`MANIFEST.md`。
