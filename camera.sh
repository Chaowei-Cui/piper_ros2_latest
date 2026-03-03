#!/bin/bash
cd /home/agilex/piper_ros
source camera_ws/devel/setup.bash
roslaunch realsense2_camera  four_camera.launch
