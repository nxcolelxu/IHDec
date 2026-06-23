#!/usr/bin/env bash
set -e

# conda 환경 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"


cd recipes

config="configs/llama3_1_eval.yaml"
model=$1 # the model name
checkpoint=$2 # the checkpoint number (0,1,2)
dataset=$3 # the dataset name (share_gpt_attack_0) 

conda run --no-capture-output -n IHDec nohup python -u evaluate.py --config $config \
    checkpointer.checkpoint_dir=$model \
    checkpointer.checkpoint_files=["original/consolidated.00.pth"] \
    checkpointer.output_dir="../output/" \
    output_dir="../output/instrct_prompted2_eval/" \
    batch_size=1 \
    dataset.source="../data/iheval/$dataset.json" >> ../logs/instruct_prompted2_$dataset.log 2>&1 &
echo "PID: $!"

# CUDA_VISIBLE_DEVICES=1 ./base_respond.sh ../pretrained_models/llama3_1_8B_Instruct 2 first_conflict_default_request