#!/bin/bash
source install/setup.bash
set -euo pipefail

DATASET_DIR="/home/agilex/data/sss"
TASK_NAME="123"
EPISODE_IDX=7
FRAME_RATE=30
POS_CMD_MODE1=0
POS_CMD_MODE2=0

# source ~/miniconda3/bin/activate
# conda activate aloha
cd collect_data

python3 replay_data.py \
    --dataset_dir "$DATASET_DIR" \
    --task_name "$TASK_NAME" \
    --episode_idx "$EPISODE_IDX" \
    --frame_rate "$FRAME_RATE" \
    --pos_cmd_mode1 "$POS_CMD_MODE1" \
    --pos_cmd_mode2 "$POS_CMD_MODE2"
