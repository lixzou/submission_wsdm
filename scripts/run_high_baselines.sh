#!/bin/bash
# run_high_baselines.sh — 高质量 baseline 批量跑（2026-08-17 用户指令）
cd /ssd1/zoulixin/tencent_previous_compression/experiments || exit 1
PY=./venv/bin/python
LOG=/tmp/hb_logs; mkdir -p $LOG
run() { local name="$1"; shift; "$@" > "$LOG/$name.log" 2>&1; echo "$name exit=$?"; }
for m in e5 bge mini; do
  run opq_scifact_$m $PY scripts/baseline_faiss.py --kind opq --ds_dir data/beir/scifact --model $m --out results/runs/hb_opq_scifact_$m.json &
  run aq_scifact_$m  $PY scripts/baseline_faiss.py --kind aq  --ds_dir data/beir/scifact --model $m --out results/runs/hb_aq_scifact_$m.json &
  run rq_scifact_$m  $PY scripts/baseline_faiss.py --kind rq  --ds_dir data/beir/scifact --model $m --out results/runs/hb_rq_scifact_$m.json &
  run tq_scifact_$m  $PY scripts/turboquant_py.py --ds_dir data/beir/scifact --model $m --out results/runs/hb_tq_scifact_$m.json &
done
wait
echo ALL-DONE
