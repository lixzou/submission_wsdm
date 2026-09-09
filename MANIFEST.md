# MANIFEST — 论文 ↔ 实验/结果对应（提交版）

| 论文部分 | 脚本 | 结果 |
|---|---|---|
| Table 1 主结果 | `scripts/main_table.py` | `results/runs/train_calib/<数据集>/<模型>.json` |
| Table 2 NQ200k | `scripts/rerun_nq_200k.py` | `results/runs/nq200k_*.json` |
| Table 3 跨域迁移 | `scripts/rerun_transfer_nq.py` | `results/runs/transfer_nq_*.json` |
| 截断消融 | `scripts/rerun_trunc_uniform_random.py` | `results/runs/trunc_uniform_random_*.json` |
| 量化消融 | `scripts/rerun_quant_stage_contribution.py` | `results/runs/quant_stage_*.json` |
| 熵编码 | `scripts/entropy_truncated.py` | `results/runs/entropy_truncated.json` |
| 显著性检验 | `scripts/train_calib_bootstrap.py` | `results/runs/train_calib_bootstrap_45.json` |
| 效率 | `scripts/eff_latency_multibudget.py` | 运行输出 |
| 论文数字权威源 | `scripts/update_canonical.py` | `results/canonical/canonical_numbers.json` |

模型：`e5` `bge` `mini` `bgem3` `qwen3`；数据集：`scifact` `fiqa` `nfcorpus` `nq_sub200k`。
校验：`python3 scripts/verify_canonical.py`（300 项，期望 PASS）。
