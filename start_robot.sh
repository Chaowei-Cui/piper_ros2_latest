source install/setup.bash
ros2 launch piper start_two_piper.launch.py can_left_port:=can_arm1 can_right_port:=can_arm2 auto_enable:=false girpper_exist:=true
