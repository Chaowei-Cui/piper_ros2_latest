#!/bin/bash
set -euo pipefail

DATASET_DIR="/home/agilex/data/sss"
TASK_NAME="123"
MAX_STEPS=4000
FRAME_RATE=30
LANGUAGE_RAW="Do folds on the shorts."

source ~/miniconda3/bin/activate
conda activate aloha
cd collect_data

mkdir -p "$DATASET_DIR/$TASK_NAME"

max_num=$(find "$DATASET_DIR/$TASK_NAME" -type f -name "episode_*.hdf5" |
    sed -E 's/.*episode_([0-9]+)\.hdf5/\1/' |
    sort -n |
    tail -1)

if [ -z "$max_num" ]; then
    max_num=-1
fi

next_num=$((max_num + 1))

python3 collect_data_four_camera.py \
    --dataset_dir "$DATASET_DIR" \
    --task_name "$TASK_NAME" \
    --max_timesteps "$MAX_STEPS" \
    --frame_rate "$FRAME_RATE" \
    --episode_idx "$next_num" \
    --language_raw "$LANGUAGE_RAW"
