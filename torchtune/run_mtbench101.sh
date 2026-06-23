#!/usr/bin/env bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"

# Usage: ./run_mtbench101.sh <gpu_id> [beta] [decay]
# Example: ./run_mtbench101.sh 0 1300 0.97

GPU=${1:-0}
BETA=${2:-1300}
DECAY=${3:-0.97}

cd "$(dirname "$0")/recipes"

INPUT_FILE="../data/mt_bench_101.json"
OUTPUT_FILE="../output/mtbench101_ihdec/mt_bench_101_beta${BETA}_decay${DECAY}.json"
LOG="../logs/mtbench101.log"

mkdir -p "../output/mtbench101_ihdec" "../logs"

echo "GPU: $GPU | beta=$BETA decay=$DECAY"
echo "Output: $OUTPUT_FILE"
echo "Log: $LOG"

CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n IHDec nohup python -u ihdec.py \
    --input-file  "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --beta        $BETA \
    --beta-decay  $DECAY \
    --max-tokens  1024 \
    >> "$LOG" 2>&1 &
echo "PID: $!"
