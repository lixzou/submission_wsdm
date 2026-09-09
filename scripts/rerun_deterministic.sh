#!/bin/bash
# rerun_deterministic.sh — 2026-08-17 确定性修复后全量重跑（margin 依赖实验）
# 修复: margin_importance 的 rel_docs sorted()（set 迭代顺序 → PYTHONHASHSEED 随机）
# 每个 job 独立进程、独立日志；exit 码汇总写入 rerun_status.tsv
cd /ssd1/zoulixin/tencent_previous_compression/experiments || exit 1
PY=./venv/bin/python
LOG=/tmp/rerun_logs
mkdir -p $LOG
STATUS=$LOG/rerun_status.tsv
: > $STATUS

run() {
  local name="$1"; shift
  "$@" > "$LOG/$name.log" 2>&1
  echo -e "$name\t$?" >> $STATUS
  echo "[done] $name exit=$? $(tail -1 $LOG/$name.log)"
}

# ---- SLOW: NQ 2M（先后台启动, 单线程大内存）----
run c1_nq_e5   $PY scripts/pilot_truncation.py --ds_dir data/beir_big/nq --model e5   --out results/exploratory/c1_nq_e5.json &
run c1_nq_bge  $PY scripts/pilot_truncation.py --ds_dir data/beir_big/nq --model bge  --out results/exploratory/c1_nq_bge.json &
run c1_nq_mini $PY scripts/pilot_truncation.py --ds_dir data/beir_big/nq --model mini --out results/exploratory/c1_nq_mini.json &
sleep 2
# ---- FAST: scifact ×5 ----
run c1_scifact_e5    $PY scripts/pilot_truncation.py --ds_dir data/beir/scifact --model e5    --out results/exploratory/c1_scifact_e5.json &
run c1_scifact_bge   $PY scripts/pilot_truncation.py --ds_dir data/beir/scifact --model bge   --out results/exploratory/c1_scifact_bge.json &
run c1_scifact_mini  $PY scripts/pilot_truncation.py --ds_dir data/beir/scifact --model mini  --out results/exploratory/c1_scifact_mini.json &
run c1_scifact_bgem3 $PY scripts/pilot_truncation.py --ds_dir data/beir/scifact --model bgem3 --out results/exploratory/c1_scifact_bgem3.json &
run c1_scifact_qwen3 $PY scripts/pilot_truncation.py --ds_dir data/beir/scifact --model qwen3 --out results/exploratory/c1_scifact_qwen3.json &
# ---- FAST: arguana ×3 ----
run c1_arguana_e5   $PY scripts/pilot_truncation.py --ds_dir data/beir_big/arguana --model e5   --out results/runs/c1_arguana_e5.json &
run c1_arguana_bge  $PY scripts/pilot_truncation.py --ds_dir data/beir_big/arguana --model bge  --out results/runs/c1_arguana_bge.json &
run c1_arguana_mini $PY scripts/pilot_truncation.py --ds_dir data/beir_big/arguana --model mini --out results/runs/c1_arguana_mini.json &
# ---- FAST: ablation / calib / drift(NQ 慢) ----
run ablate_e5 $PY scripts/gap_experiments.py ablate --ds_dir data/beir/scifact --model e5  --out results/runs/ablate2_scifact_e5.json  --k 2 4 &
run ablate_bge $PY scripts/gap_experiments.py ablate --ds_dir data/beir/scifact --model bge --out results/runs/ablate2_scifact_bge.json --k 2 4 &
run calib_e5 $PY scripts/gap_experiments.py calib --ds_dir data/beir/scifact --model e5 --out results/runs/calib_scifact_e5.json --keep 0.125 --fracs 0.1 0.25 0.5 0.7 &
run drift_e5 $PY scripts/gap_experiments.py drift --calib_ds data/beir/scifact --eval_ds data/beir_big/nq --model e5 --out results/runs/drift_scifact_nq_e5.json &
# ---- FAST: bootstrap ×6 ----
run boot_e5_125  $PY scripts/run_bootstrap.py --ds_dir data/beir/scifact --model e5   --keep 0.125 --out bootstrap_scifact_e5_k0.125.json &
run boot_e5_25   $PY scripts/run_bootstrap.py --ds_dir data/beir/scifact --model e5   --keep 0.25  --out bootstrap_scifact_e5_k0.25.json &
run boot_bge_125 $PY scripts/run_bootstrap.py --ds_dir data/beir/scifact --model bge  --keep 0.125 --out bootstrap_scifact_bge_k0.125.json &
run boot_bge_25  $PY scripts/run_bootstrap.py --ds_dir data/beir/scifact --model bge  --keep 0.25  --out bootstrap_scifact_bge_k0.25.json &
run boot_mini_125 $PY scripts/run_bootstrap.py --ds_dir data/beir/scifact --model mini --keep 0.125 --out bootstrap_scifact_mini_k0.125.json &
run boot_mini_25  $PY scripts/run_bootstrap.py --ds_dir data/beir/scifact --model mini --keep 0.25  --out bootstrap_scifact_mini_k0.25.json &
# ---- FAST: c2 (uniform vs tilt) scifact ×5 + nq 5k ×2 ----
run c2_scifact_e5    $PY scripts/pilot_2b.py --ds_dir data/beir/scifact --model e5    --out results/exploratory/pilot_2b_scifact_e5.json &
run c2_scifact_bge   $PY scripts/pilot_2b.py --ds_dir data/beir/scifact --model bge   --out results/exploratory/pilot_2b_scifact_bge.json &
run c2_scifact_mini  $PY scripts/pilot_2b.py --ds_dir data/beir/scifact --model mini  --out results/exploratory/pilot_2b_scifact_mini.json &
run c2_scifact_bgem3 $PY scripts/pilot_2b.py --ds_dir data/beir/scifact --model bgem3 --out results/exploratory/pilot_2b_scifact_bgem3.json &
run c2_scifact_qwen3 $PY scripts/pilot_2b.py --ds_dir data/beir/scifact --model qwen3 --out results/exploratory/pilot_2b_scifact_qwen3.json &
run c2_nq5k_e5  $PY scripts/pilot_2b.py --ds_dir data/beir_big/nq --model e5  --subset_docs 5000 --subset_queries 500 --out results/exploratory/pilot_2b_nq5k_e5.json &
run c2_nq5k_bge $PY scripts/pilot_2b.py --ds_dir data/beir_big/nq --model bge --subset_docs 5000 --subset_queries 500 --out results/exploratory/pilot_2b_nq5k_bge.json &

wait
echo "=== ALL DONE ==="
cat $STATUS
