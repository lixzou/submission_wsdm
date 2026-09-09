#!/bin/bash
# Phase 3 Step 3.1/3.2: clone baseline 官方代码 + 下载对比数据集
set -e
cd /ssd1/zoulixin/tencent_previous_compression/experiments/third_party
mkdir -p data

echo "=== clone repos ==="
[ -d RaBitQ ] || git -c http.proxy= -c https.proxy= clone --depth 1 https://github.com/gaoj0017/RaBitQ.git
[ -d Extended-RaBitQ ] || git -c http.proxy= -c https.proxy= clone --depth 1 https://github.com/VectorDB-NTU/Extended-RaBitQ.git
[ -d SAQ ] || git -c http.proxy= -c https.proxy= clone --depth 1 https://github.com/howarlii/SAQ.git
[ -d faiss ] || git -c http.proxy= -c https.proxy= clone --depth 1 https://github.com/facebookresearch/faiss.git

echo "=== download SIFT1M / GIST1M (ann-benchmarks) ==="
cd data
[ -f sift_base.fbin ] || curl -sL -o sift_base.fbin https://dl.fbaipublicfiles.com/ann-benchmarks/sift-128-euclidean/base.fbin
[ -f sift_query.fbin ] || curl -sL -o sift_query.fbin https://dl.fbaipublicfiles.com/ann-benchmarks/sift-128-euclidean/query.fbin
[ -f sift_gt.bin ] || curl -sL -o sift_gt.bin https://dl.fbaipublicfiles.com/ann-benchmarks/sift-128-euclidean/gt.bin
[ -f gist_base.fbin ] || curl -sL -o gist_base.fbin https://dl.fbaipublicfiles.com/ann-benchmarks/gist-960-euclidean/base.fbin
[ -f gist_query.fbin ] || curl -sL -o gist_query.fbin https://dl.fbaipublicfiles.com/ann-benchmarks/gist-960-euclidean/query.fbin
[ -f gist_gt.bin ] || curl -sL -o gist_gt.bin https://dl.fbaipublicfiles.com/ann-benchmarks/gist-960-euclidean/gt.bin
ls -la

echo "=== build RaBitQ ==="
cd ../RaBitQ && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. > /dev/null 2>&1 && make -j8 > /dev/null 2>&1 && echo "RaBitQ built" || echo "RaBitQ build FAILED (see log)"

echo "=== build Extended-RaBitQ ==="
cd ../../Extended-RaBitQ && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. > /dev/null 2>&1 && make -j8 > /dev/null 2>&1 && echo "Extended-RaBitQ built" || echo "Extended-RaBitQ build FAILED (see log)"

echo "SETUP DONE"
