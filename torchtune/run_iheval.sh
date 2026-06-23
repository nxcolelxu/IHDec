#!/usr/bin/env bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"

# Usage: ./run_iheval.sh <gpu_id> <dataset_name> [beta] [decay]
# dataset_name: filename under torchtune/data/iheval/, without .json
#
# Rule-following: both_conflict_default_request, first_conflict_default_request, single_conflict_request
# Safety:         hijack_strong, hijack_weak, extract_strong, extract_weak
#
# Example:
#   ./run_iheval.sh 0 both_conflict_default_request 1300 0.97
#   ./run_iheval.sh 1 hijack_weak 1300 0.97

GPU=${1:-0}
DATASET=${2:-"both_conflict_default_request"}
BETA=${3:-1300}
DECAY=${4:-0.97}

cd "$(dirname "$0")/recipes"

SLUG="$DATASET"
INPUT_FILE="../data/iheval/${DATASET}.json"
OUTPUT_FILE="../output/iheval_ihdec/${SLUG}_beta${BETA}_decay${DECAY}.json"
LOG="../logs/iheval_${SLUG}.log"

mkdir -p "../output/iheval_ihdec" "../logs"

echo "GPU: $GPU | dataset=$DATASET | beta=$BETA decay=$DECAY"
echo "Output: $OUTPUT_FILE"
echo "Log: $LOG"

CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n IHDec nohup python -u ihdec.py \
    --input-file  "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --beta        $BETA \
    --beta-decay  $DECAY \
    >> "$LOG" 2>&1 &
echo "PID: $!"
