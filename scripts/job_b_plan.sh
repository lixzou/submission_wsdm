#!/bin/bash
cd "$(dirname "$0")"
export PATH="/home/zoulixin/ratc_venv/bin:$PATH"
export DS_ROOT="$HOME/ray_hdd/datasets/beir"
echo "=== 集群环境 ==="
python3 -c "import faiss,scipy,numpy; print('faiss',faiss.__version__,'ok')"

echo "=== 方案 B 主表: train 校准, keep=0.5, B=2/4/8 (32x/16x/8x) ==="
for ds in scifact fiqa nfcorpus; do
  for m in e5 bge mini bgem3 qwen3; do
    echo "--- $ds $m ---"
    python3 -u train_calib_main.py --model $m --ds_dir $DS_ROOT/$ds \
      --keeps 0.5 --bits 2,4,8 \
      --out results_b_${ds}_${m}.json 2>&1 | grep -vE "WARNING"
  done
done
echo "=== ALL DONE ==="
ls -la results_b_*.json 2>/dev/null | wc -l
