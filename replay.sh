#!/bin/bash
# bash replay.sh eef
# bash replay.sh joint
source install/setup.bash
set -euo pipefail

DATASET_DIR="/home/agilex/data/sss"
TASK_NAME="123"
EPISODE_IDX=24
FRAME_RATE=30
POS_CMD_MODE1=0
POS_CMD_MODE2=0
PRINT_DATA_INFO=1
PRINT_EVERY_N=1
REPLAY_MODE="${1:-eef}"  # eef: 末端数据回放, joint: 关节数据回放

POS_CMD_LEFT_TOPIC="/pos_cmd_left"
POS_CMD_RIGHT_TOPIC="/pos_cmd_right"
JOINT_CTRL_CMD_LEFT_TOPIC="/joint_ctrl_cmd_left"
JOINT_CTRL_CMD_RIGHT_TOPIC="/joint_ctrl_cmd_right"

case "$REPLAY_MODE" in
    eef|end|cartesian|末端)
        # 末端数据回放: 仅让控制节点接收 /pos_cmd_*，关节控制命令发布到无订阅话题
        JOINT_CTRL_CMD_LEFT_TOPIC="/unused_joint_ctrl_cmd_left"
        JOINT_CTRL_CMD_RIGHT_TOPIC="/unused_joint_ctrl_cmd_right"
        ;;
    joint|joints|关节)
        # 关节数据回放: 仅让控制节点接收 /joint_ctrl_cmd_*，末端控制命令发布到无订阅话题
        POS_CMD_LEFT_TOPIC="/unused_pos_cmd_left"
        POS_CMD_RIGHT_TOPIC="/unused_pos_cmd_right"
        ;;
    *)
        echo "用法: $0 [eef|joint]"
        echo "  eef   : 使用末端数据回放 (默认)"
        echo "  joint : 使用关节数据回放"
        exit 1
        ;;
esac

echo "Replay mode: $REPLAY_MODE"
echo "Dataset: $DATASET_DIR/$TASK_NAME/episode_$EPISODE_IDX.hdf5"

# source ~/miniconda3/bin/activate
# conda activate aloha
cd collect_data

PRINT_ARGS=()
if [ "$PRINT_DATA_INFO" = "1" ]; then
    PRINT_ARGS+=(--print_data_info)
fi

python3 replay_data.py \
    --dataset_dir "$DATASET_DIR" \
    --task_name "$TASK_NAME" \
    --episode_idx "$EPISODE_IDX" \
    --frame_rate "$FRAME_RATE" \
    --pos_cmd_mode1 "$POS_CMD_MODE1" \
    --pos_cmd_mode2 "$POS_CMD_MODE2" \
    --print_every_n "$PRINT_EVERY_N" \
    "${PRINT_ARGS[@]}" \
    --joint_states_left_topic /joint_states_left \
    --joint_states_right_topic /joint_states_right \
    --joint_left_topic /joint_left \
    --joint_right_topic /joint_right \
    --joint_ctrl_cmd_left_topic "$JOINT_CTRL_CMD_LEFT_TOPIC" \
    --joint_ctrl_cmd_right_topic "$JOINT_CTRL_CMD_RIGHT_TOPIC" \
    --end_pose_left_topic /end_pose_left \
    --end_pose_right_topic /end_pose_right \
    --pos_cmd_left_topic "$POS_CMD_LEFT_TOPIC" \
    --pos_cmd_right_topic "$POS_CMD_RIGHT_TOPIC" \
    --img_left_topic /camera/left/color/image_raw \
    --img_right_topic /camera/right/color/image_raw \
    --img_top_topic /camera/top/color/image_raw \
    --img_left_depth_topic /camera/left/depth/image_rect_raw \
    --img_right_depth_topic /camera/right/depth/image_rect_raw \
    --img_top_depth_topic /camera/top/depth/image_rect_raw
