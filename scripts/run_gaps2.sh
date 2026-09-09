#!/bin/bash
# 缺口实验续跑（opq_had bug 已修）
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

for M in e5 bge mini; do
  log "opq_had $M"
  $PY gap_experiments.py opq_had --ds_dir $SCIFACT --model $M --out results/runs/opq_had_scifact_$M.json
done

for M in e5 bge mini; do
  log "arguana c1 $M"
  $PY gap_experiments.py c1 --ds_dir $ARGU --model $M --out results/runs/c1_arguana_$M.json
done

for M in e5 bge mini; do
  log "arguana c3 $M"
  $PY gap_experiments.py c3 --ds_dir $ARGU --model $M --out results/runs/c3_arguana_$M.json
done

log "drift scifact->nq e5"
$PY gap_experiments.py drift --calib_ds $SCIFACT --eval_ds $NQ --model e5 --out results/runs/drift_scifact_nq_e5.json

log "nq uniform e5"
$PY gap_experiments.py nq_uniform --ds_dir $NQ --model e5 --out results/runs/nq_uniform_e5.json

log "ALL GAPS2 DONE"
