#!/bin/bash
set -euo pipefail

DATASET_DIR="/home/agilex/data/sss"
TASK_NAME="123"
EPISODE_IDX=0
FRAME_RATE=30

source ~/miniconda3/bin/activate
conda activate aloha
cd collect_data

python3 replay_data.py \
    --dataset_dir "$DATASET_DIR" \
    --task_name "$TASK_NAME" \
    --episode_idx "$EPISODE_IDX" \
    --frame_rate "$FRAME_RATE"
