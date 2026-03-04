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
    --pos_cmd_mode2 "$POS_CMD_MODE2" \
    --joint_states_left_topic /joint_states_left \
    --joint_states_right_topic /joint_states_right \
    --joint_left_topic /joint_left \
    --joint_right_topic /joint_right \
    --joint_ctrl_cmd_left_topic /joint_ctrl_cmd_left \
    --joint_ctrl_cmd_right_topic /joint_ctrl_cmd_right \
    --end_pose_left_topic /end_pose_left \
    --end_pose_right_topic /end_pose_right \
    --arm_status_left_topic /arm_status_left \
    --arm_status_right_topic /arm_status_right \
    --pos_cmd_left_topic /pos_cmd_left \
    --pos_cmd_right_topic /pos_cmd_right \
    --img_left_topic /camera/left/color/image_raw \
    --img_right_topic /camera/right/color/image_raw \
    --img_top_topic /camera/top/color/image_raw \
    --img_left_depth_topic /camera/left/depth/image_rect_raw \
    --img_right_depth_topic /camera/right/depth/image_rect_raw \
    --img_top_depth_topic /camera/top/depth/image_rect_raw
