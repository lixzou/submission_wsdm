#!/bin/bash
cd "$(dirname "$0")"
export PATH="/home/zoulixin/ratc_venv/bin:$PATH"
echo "=== 集群环境 ==="
python3 -c "import faiss,scipy,numpy; print('faiss',faiss.__version__,'ok')"

echo "=== 1. NQ 200k 主表 (e5/bge/mini) ==="
for m in e5 bge mini; do
  echo "--- nq_sub200k $m ---"
  python3 -u main_table.py --ds_dir nq_sub200k --model $m --methods pq opq itq scalar tq reco --out results_main_nq_${m}.json 2>&1 | grep -vE "WARNING"
done

echo "=== 2. RQ/LSQ 慢方法 (scifact 5 模型) ==="
for m in e5 bge mini bgem3 qwen3; do
  echo "--- scifact $m slow ---"
  python3 -u main_table.py --ds_dir scifact --model $m --methods rq lsq --out results_main_scifact_${m}_slow.json 2>&1 | grep -vE "WARNING"
done

echo "=== 3. RQ/LSQ (arguana) ==="
for m in e5 bge mini; do
  echo "--- arguana $m slow ---"
  python3 -u main_table.py --ds_dir arguana --model $m --methods rq lsq --out results_main_arguana_${m}_slow.json 2>&1 | grep -vE "WARNING"
done

echo "=== ALL DONE ==="
ls -la results_*.json 2>/dev/null | wc -l
