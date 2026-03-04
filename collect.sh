#!/bin/bash
source install/setup.bash
set -euo pipefail

DATASET_DIR="/home/agilex/data/sss"
TASK_NAME="123"
MAX_STEPS=4000
FRAME_RATE=30
LANGUAGE_RAW="Do folds on the shorts."

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

python3 collect_data.py \
    --dataset_dir "$DATASET_DIR" \
    --task_name "$TASK_NAME" \
    --max_timesteps "$MAX_STEPS" \
    --frame_rate "$FRAME_RATE" \
    --episode_idx "$next_num" \
    --language_raw "$LANGUAGE_RAW" \
    --joint_states_left_topic /joint_states_left \
    --joint_states_right_topic /joint_states_right \
    --joint_left_topic /joint_left \
    --joint_right_topic /joint_right \
    --end_pose_left_topic /end_pose_left \
    --end_pose_right_topic /end_pose_right \
    --arm_status_left_topic /arm_status_left \
    --arm_status_right_topic /arm_status_right \
    --img_left_topic /camera/left/color/image_raw \
    --img_right_topic /camera/right/color/image_raw \
    --img_top_topic /camera/top/color/image_raw \
    --img_left_depth_topic /camera/left/depth/image_rect_raw \
    --img_right_depth_topic /camera/right/depth/image_rect_raw \
    --img_top_depth_topic /camera/top/depth/image_rect_raw 
    # --use_depth_image False \
