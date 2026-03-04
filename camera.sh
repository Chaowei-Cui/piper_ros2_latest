#!/bin/bash
source /opt/ros/humble/share/realsense2_camera/install/setup.bash
# 切换到 launch 文件目录
cd /opt/ros/humble/share/realsense2_camera/launch

# 定义一个函数来启动相机
start_camera() {
    local cam_name=$1
    local serial=$2
    
    echo "正在启动 $cam_name (序列号: $serial)..."
    
    # 注意：这里使用 Python Launch 文件的思路最稳，
    # 但如果非要用 rs_launch.py，我们需要用一个临时 yaml 来传参
    
    # 1. 为每个相机创建一个临时的 yaml 参数文件（确保 serial_no 是字符串）
    cat > /tmp/realsense_${cam_name}_params.yaml << EOF
/**:
  ros__parameters:
    serial_no: "$serial"
EOF

    # 2. 启动相机并放入后台 (&)
    ros2 launch realsense2_camera rs_launch.py \
        camera_namespace:=camera \
        camera_name:=${cam_name} \
        serial_no:="'$serial'" &
}

# --- 主程序 ---

echo "开始启动所有 RealSense 相机..."

# 启动顶部相机
start_camera "top" '335222073051'
sleep 2 # 稍微间隔一下，避免 USB 带宽冲突

# 启动左侧相机
start_camera "left" '239122070290'
sleep 2

# 启动右侧相机
start_camera "right" '327122076728'

echo "所有相机启动命令已发送。"
echo "使用 'ros2 topic list' 查看话题。"
echo "使用 'ros2 node list' 查看节点。"

# 保持脚本运行，或者按 Ctrl+C 退出（相机进程仍在后台）
wait