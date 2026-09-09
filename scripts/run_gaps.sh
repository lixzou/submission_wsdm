#!/bin/bash
# 缺口实验批处理（后台运行，日志 gaps.log）
set -e
cd "$(dirname "$0")/.."
PY=venv/bin/python
DATA=data
BIG=$DATA/beir_big
SCIFACT=$DATA/beir/scifact
NQ=$BIG/nq
ARGU=$BIG/arguana
mkdir -p results/runs

log() { echo "[$(date +%H:%M:%S)] $*"; }

# 1. Thm2 谱分析（快）
log "thm2 spectra"
$PY gap_experiments.py thm2 --ds_dir $SCIFACT --out results/runs/thm2_spectra.json

# 2. 消融: 排序准则 + 块白化（scifact e5/bge）
for M in e5 bge; do
  log "ablate $M"
  $PY gap_experiments.py ablate --ds_dir $SCIFACT --model $M --out results/runs/ablate_scifact_$M.json
done

# 3. 校准集大小敏感性（scifact e5）
log "calib sizes"
$PY gap_experiments.py calib --ds_dir $SCIFACT --model e5 --out results/runs/calib_scifact_e5.json

# 4. OPQ 近似（Hadamard + PQ, scifact 3 模型）
for M in e5 bge mini; do
  log "opq_had $M"
  $PY gap_experiments.py opq_had --ds_dir $SCIFACT --model $M --out results/runs/opq_had_scifact_$M.json
done

# 5. ArguAna C1（3 模型）
for M in e5 bge mini; do
  log "arguana c1 $M"
  $PY gap_experiments.py c1 --ds_dir $ARGU --model $M --out results/runs/c1_arguana_$M.json
done

# 6. ArguAna C3 熵（3 模型, 1000 文档子集）
for M in e5 bge mini; do
  log "arguana c3 $M"
  $PY gap_experiments.py c3 --ds_dir $ARGU --model $M --out results/runs/c3_arguana_$M.json
done

# 7. 域漂移: scifact 校准序 → NQ 2M 评测（e5, 重）
log "drift scifact->nq e5"
$PY gap_experiments.py drift --calib_ds $SCIFACT --eval_ds $NQ --model e5 --out results/runs/drift_scifact_nq_e5.json

# 8. NQ 2M 均匀 SQ 对照（e5, 最重）
log "nq uniform e5"
$PY gap_experiments.py nq_uniform --ds_dir $NQ --model e5 --out results/runs/nq_uniform_e5.json

log "ALL GAPS DONE"
