# -- coding: UTF-8
import os
import time
import numpy as np
import h5py
import argparse
import dm_env
import threading
import collections
from collections import deque
from pynput import keyboard
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import sys
import cv2
import signal
from colorama import init,Fore,Back,Style # print have color
MAX_DEQUE = 10
global saveData
saveData = False

global SUBTASK_STEP
SUBTASK_STEP = 0

global DEQUE_SIZE
DEQUE_SIZE = 10

global CUR_STEP, PRE_STEP
CUR_STEP = 0
PRE_STEP = 0
global CUR_TIME_STAMP
CUR_TIME_STAMP = time.time()
global PRE_TIME_STAMP
PRE_TIME_STAMP = time.time()

SUBTASK_FLAG = np.zeros((10000, 1))


def onPress(key):
    # print(f"Pressing the {key}")

    global saveData
    if key == keyboard.Key.enter:
        saveData = True

    if key == keyboard.Key.esc:
        return False


def onRelease(key):
    global CUR_STEP, CUR_TIME_STAMP, PRE_TIME_STAMP, SUBTASK_FLAG, SUBTASK_STEP, PRE_STEP
    # print(f"Releasing the {key}")
    if key == keyboard.Key.scroll_lock:
        PRE_TIME_STAMP = CUR_TIME_STAMP
        CUR_TIME_STAMP = time.time()
        if CUR_TIME_STAMP - PRE_TIME_STAMP < 0.4:
            print(
                f">>>>>Double steps occur at {PRE_STEP} && {CUR_STEP} | Time are {CUR_TIME_STAMP} | {PRE_TIME_STAMP} <<<<<<")
            SUBTASK_FLAG[CUR_STEP][0] = 100
            SUBTASK_FLAG[PRE_STEP][0] = 100
        else:
            print(f'One step occurs at {CUR_STEP}')
            SUBTASK_FLAG[CUR_STEP][0] = 10
            SUBTASK_STEP += 1
            PRE_STEP = CUR_STEP


def listen_for_keyboard():
    with keyboard.Listener(on_release=onRelease, on_press=onPress) as listener:
        listener.join()


# 保存数据函数
def save_data(args, timesteps, actions, dataset_path):
    # 数据字典
    data_size = len(actions)
    global SUBTASK_FLAG

    # Determine number of joints from actual data
    num_joints = len(actions[0]) if len(actions) > 0 else args.num_joints

    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
        # '/base_action_t265': [],
        '/subtask': SUBTASK_FLAG[:data_size, :],
    }

    # 相机字典  观察的图像
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    # 动作长度 遍历动作
    while actions:
        # 循环弹出一个队列
        action = actions.pop(0)  # 动作  当前动作
        ts = timesteps.pop(0)  # 奖励  前一帧

        # 往字典里面添值
        # Timestep返回的qpos，qvel,effort
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])

        # 实际发的action
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        # 相机数据
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩
        #
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations')
        image = obs.create_group('images')
        if "language_raw" not in root.keys():
            print("languge_raw", args.language_raw)
            root.create_dataset("language_raw", data=[args.language_raw])
        for cam_name in args.camera_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                     chunks=(1, 480, 640, 3), )
        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint16',
                                               chunks=(1, 480, 640), )

        _ = obs.create_dataset('qpos', (data_size, num_joints))
        _ = obs.create_dataset('qvel', (data_size, num_joints))
        _ = obs.create_dataset('effort', (data_size, num_joints))
        _ = root.create_dataset('action', (data_size, num_joints))
        _ = root.create_dataset('base_action', (data_size, 2))
        _ = root.create_dataset('subtask', (data_size, 1), dtype='uint8')

        # data_dict write into h5py.File
        for name, array in data_dict.items():
            root[name][...] = array
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)


class RosOperator(Node):
    def __init__(self, args):
        super().__init__('record_episodes')
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.master_arm_right_deque = None
        self.master_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_top_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_top_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.args = args
        self.init_deques()
        self.init_ros()

    def init_deques(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque(maxlen=MAX_DEQUE)
        self.img_right_deque = deque(maxlen=MAX_DEQUE)
        self.img_front_deque = deque(maxlen=MAX_DEQUE)
        self.img_left_depth_deque = deque(maxlen=MAX_DEQUE)
        self.img_right_depth_deque = deque(maxlen=MAX_DEQUE)
        # self.img_front_depth_deque = deque(maxlen=MAX_DEQUE)
        self.img_top_deque = deque(maxlen=MAX_DEQUE)
        self.img_top_depth_deque = deque(maxlen=MAX_DEQUE)
        self.master_arm_left_deque = deque(maxlen=MAX_DEQUE)
        self.master_arm_right_deque = deque(maxlen=MAX_DEQUE)
        self.puppet_arm_left_deque = deque(maxlen=MAX_DEQUE)

        self.puppet_arm_right_deque = deque(maxlen=MAX_DEQUE)
        self.robot_base_deque = deque(maxlen=MAX_DEQUE)

    def get_frame(self):

        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_top_deque) == 0 or \
                (self.args.use_depth_image and (
                        len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(
                    self.img_top_depth_deque) == 0)):
            if len(self.img_left_deque) == 0:
                print("left image deque")
            if len(self.img_top_deque) == 0:
                print("top image fisrt")
            if len(self.img_right_deque) == 0:
                print("right image deque")

            return False
        if self.args.use_depth_image:
            frame_time = min(
                [self.img_left_deque[-1].header.stamp.sec + self.img_left_deque[-1].header.stamp.nanosec * 1e-9,
                 self.img_right_deque[-1].header.stamp.sec + self.img_right_deque[-1].header.stamp.nanosec * 1e-9,
                 self.img_front_deque[-1].header.stamp.sec + self.img_front_deque[-1].header.stamp.nanosec * 1e-9,
                 self.img_left_depth_deque[-1].header.stamp.sec + self.img_left_depth_deque[-1].header.stamp.nanosec * 1e-9,
                 self.img_right_depth_deque[-1].header.stamp.sec + self.img_right_depth_deque[-1].header.stamp.nanosec * 1e-9,
                 self.img_top_depth_deque[-1].header.stamp.sec + self.img_top_depth_deque[-1].header.stamp.nanosec * 1e-9])
        else:
            frame_time = min(
                [self.img_left_deque[-1].header.stamp.sec + self.img_left_deque[-1].header.stamp.nanosec * 1e-9,
                 self.img_right_deque[-1].header.stamp.sec + self.img_right_deque[-1].header.stamp.nanosec * 1e-9,
                 self.img_top_deque[-1].header.stamp.sec + self.img_top_deque[-1].header.stamp.nanosec * 1e-9])

        if len(self.img_left_deque) == 0 or (self.img_left_deque[-1].header.stamp.sec + self.img_left_deque[-1].header.stamp.nanosec * 1e-9) < frame_time:
            print("left image")
            return False
        if len(self.img_right_deque) == 0 or (self.img_right_deque[-1].header.stamp.sec + self.img_right_deque[-1].header.stamp.nanosec * 1e-9) < frame_time:
            print("right image")
            return False
        # if len(self.img_front_deque) == 0 or (self.img_front_deque[-1].header.stamp.sec + self.img_front_deque[-1].header.stamp.nanosec * 1e-9) < frame_time:
        #     print("front image")
        #     return False
        if len(self.master_arm_left_deque) == 0 or (self.master_arm_left_deque[-1].header.stamp.sec + self.master_arm_left_deque[-1].header.stamp.nanosec * 1e-9) < frame_time:
            print("master left")
            return False
        if len(self.master_arm_right_deque) == 0 or (self.master_arm_right_deque[-1].header.stamp.sec + self.master_arm_right_deque[-1].header.stamp.nanosec * 1e-9) < frame_time:
            print("master right")
            return False
        if len(self.puppet_arm_left_deque) == 0 or (self.puppet_arm_left_deque[-1].header.stamp.sec + self.puppet_arm_left_deque[-1].header.stamp.nanosec * 1e-9) < frame_time:
            print("pupet left")
            return False
        if len(self.puppet_arm_right_deque) == 0 or (self.puppet_arm_right_deque[-1].header.stamp.sec + self.puppet_arm_right_deque[-1].header.stamp.nanosec * 1e-9) < frame_time:
            print("pupet right")
            return False
        # add top image
        if len(self.img_top_deque) == 0 or (self.img_top_deque[-1].header.stamp.sec + self.img_top_deque[-1].header.stamp.nanosec * 1e-9) < frame_time:
            print("top image")
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or (self.img_left_depth_deque[
            -1].header.stamp.sec + self.img_left_depth_deque[-1].header.stamp.nanosec * 1e-9) < frame_time):
            print("left depth ")
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or (self.img_right_depth_deque[
            -1].header.stamp.sec + self.img_right_depth_deque[-1].header.stamp.nanosec * 1e-9) < frame_time):
            print("right depth")
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or (self.img_front_depth_deque[
            -1].header.stamp.sec + self.img_front_depth_deque[-1].header.stamp.nanosec * 1e-9) < frame_time):
            print("front depth")
            return False
        if self.args.use_robot_base and (
                len(self.robot_base_deque) == 0 or (self.robot_base_deque[-1].header.stamp.sec + self.robot_base_deque[-1].header.stamp.nanosec * 1e-9) < frame_time):
            print("base")
            return False
        if self.args.use_depth_image and (
                len(self.img_top_depth_deque) == 0 or (self.img_top_depth_deque[-1].header.stamp.sec + self.img_top_depth_deque[-1].header.stamp.nanosec * 1e-9) < frame_time):
            print("top depth")
            return False

        while (self.img_left_deque[0].header.stamp.sec + self.img_left_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')
        # print("img_left:", img_left.shape)

        while (self.img_right_deque[0].header.stamp.sec + self.img_right_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        # while (self.img_front_deque[0].header.stamp.sec + self.img_front_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
        #     self.img_front_deque.popleft()
        # img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        img_top = self.bridge.imgmsg_to_cv2(self.img_top_deque.popleft(), 'passthrough')

        while (self.master_arm_left_deque[0].header.stamp.sec + self.master_arm_left_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
            self.master_arm_left_deque.popleft()
        master_arm_left = self.master_arm_left_deque.popleft()

        while (self.master_arm_right_deque[0].header.stamp.sec + self.master_arm_right_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
            self.master_arm_right_deque.popleft()
        master_arm_right = self.master_arm_right_deque.popleft()

        while (self.puppet_arm_left_deque[0].header.stamp.sec + self.puppet_arm_left_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while (self.puppet_arm_right_deque[0].header.stamp.sec + self.puppet_arm_right_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while (self.img_left_depth_deque[0].header.stamp.sec + self.img_left_depth_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')
            top, bottom, left, right = 40, 40, 0, 0
            img_left_depth = cv2.copyMakeBorder(img_left_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_right_depth = None
        if self.args.use_depth_image:
            while (self.img_right_depth_deque[0].header.stamp.sec + self.img_right_depth_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')
        top, bottom, left, right = 40, 40, 0, 0
        img_right_depth = cv2.copyMakeBorder(img_right_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_top_depth = None
        if self.args.use_depth_image:
            while (self.img_top_depth_deque[0].header.stamp.sec + self.img_top_depth_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
                self.img_top_depth_deque.popleft()
            img_top_depth = self.bridge.imgmsg_to_cv2(self.img_top_depth_deque.popleft(), 'passthrough')
        top, bottom, left, right = 40, 40, 0, 0
        img_top_depth = cv2.copyMakeBorder(img_top_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        robot_base = None
        if self.args.use_robot_base:
            while (self.robot_base_deque[0].header.stamp.sec + self.robot_base_deque[0].header.stamp.nanosec * 1e-9) < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_left, img_right, img_left_depth, img_right_depth, img_top, img_top_depth,
                puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= MAX_DEQUE:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)
        # Debug: print first few messages
        if len(self.img_left_deque) <= 3:
            print(f"Left image received, deque size: {len(self.img_left_deque)}")

    def img_top_callback(self, msg):
        if len(self.img_top_deque) >= MAX_DEQUE:
            self.img_top_deque.popleft()
        self.img_top_deque.append(msg)
        # Debug: print first few messages
        if len(self.img_top_deque) <= 3:
            print(f"Top image received, deque size: {len(self.img_top_deque)}")

    def img_top_depth_callback(self, msg):
        if len(self.img_top_depth_deque) >= MAX_DEQUE:
            self.img_top_depth_deque.popleft()
        self.img_top_depth_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= MAX_DEQUE:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)
        # Debug: print first few messages
        if len(self.img_right_deque) <= 3:
            print(f"Right image received, deque size: {len(self.img_right_deque)}")

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= MAX_DEQUE:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= MAX_DEQUE:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= MAX_DEQUE:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= MAX_DEQUE:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= MAX_DEQUE:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)
        # Debug: print first few messages
        if len(self.master_arm_left_deque) <= 3:
            print(f"Master left arm received, deque size: {len(self.master_arm_left_deque)}")

    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= MAX_DEQUE:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)
        # Debug: print first few messages
        if len(self.master_arm_right_deque) <= 3:
            print(f"Master right arm received, deque size: {len(self.master_arm_right_deque)}")

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= MAX_DEQUE:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)
        # Debug: print first few messages
        if len(self.puppet_arm_left_deque) <= 3:
            print(f"Puppet left arm received, deque size: {len(self.puppet_arm_left_deque)}")

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= MAX_DEQUE:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)
        # Debug: print first few messages
        if len(self.puppet_arm_right_deque) <= 3:
            print(f"Puppet right arm received, deque size: {len(self.puppet_arm_right_deque)}")

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= MAX_DEQUE:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        # ROS2 subscriptions
        print(f"Subscribing to image topics:")
        print(f"  Left: {self.args.img_left_topic}")
        print(f"  Right: {self.args.img_right_topic}")
        print(f"  Top: {self.args.img_top_topic}")

        self.create_subscription(Image, self.args.img_left_topic, self.img_left_callback, MAX_DEQUE)
        self.create_subscription(Image, self.args.img_right_topic, self.img_right_callback, MAX_DEQUE)
        # self.create_subscription(Image, self.args.img_front_topic, self.img_front_callback, MAX_DEQUE)
        self.create_subscription(Image, self.args.img_top_topic, self.img_top_callback, MAX_DEQUE)

        if self.args.use_depth_image:
            print(f"Subscribing to depth topics:")
            print(f"  Left depth: {self.args.img_left_depth_topic}")
            print(f"  Right depth: {self.args.img_right_depth_topic}")
            print(f"  Top depth: {self.args.img_top_depth_topic}")
            self.create_subscription(Image, self.args.img_left_depth_topic, self.img_left_depth_callback, MAX_DEQUE)
            self.create_subscription(Image, self.args.img_right_depth_topic, self.img_right_depth_callback, MAX_DEQUE)
            # self.create_subscription(Image, self.args.img_front_depth_topic, self.img_front_depth_callback, MAX_DEQUE)
            self.create_subscription(Image, self.args.img_top_depth_topic, self.img_top_depth_callback, MAX_DEQUE)

        print(f"Subscribing to arm topics:")
        print(f"  Master left: {self.args.master_arm_left_topic}")
        print(f"  Master right: {self.args.master_arm_right_topic}")
        print(f"  Puppet left: {self.args.puppet_arm_left_topic}")
        print(f"  Puppet right: {self.args.puppet_arm_right_topic}")

        self.create_subscription(JointState, self.args.master_arm_left_topic, self.master_arm_left_callback, MAX_DEQUE)
        self.create_subscription(JointState, self.args.master_arm_right_topic, self.master_arm_right_callback, MAX_DEQUE)
        self.create_subscription(JointState, self.args.puppet_arm_left_topic, self.puppet_arm_left_callback, MAX_DEQUE)
        self.create_subscription(JointState, self.args.puppet_arm_right_topic, self.puppet_arm_right_callback, MAX_DEQUE)
        self.create_subscription(Odometry, self.args.robot_base_topic, self.robot_base_callback, MAX_DEQUE)

        print("All subscriptions created successfully!")

    def process(self):
        timesteps = []
        actions = []
        # 图像数据
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        image_dict = dict()
        for cam_name in self.args.camera_names:
            image_dict[cam_name] = image
        count = 0

        # input_key = input("please input s:")
        # while input_key != 's' and rclpy.ok():
        #     input_key = input("please input s:")

        rate = self.create_rate(self.args.frame_rate)
        print_flag = True
        global CUR_STEP

        while ((count < self.args.max_timesteps + 1) and not saveData) and rclpy.ok():
            # 2 收集数据
            # Spin once to process callbacks
            rclpy.spin_once(self, timeout_sec=0.001)

            # print(Back.BLUE) # Set background color red
            result = self.get_frame()
            CUR_STEP = count

            if not result:
                if print_flag:
                    print(Fore.RED,">>>>>>>>>>>>>>>>>>>>>>>>>>syn fail>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print_flag = False
                # rate.sleep()
                time.sleep(0.01)  # Small delay to allow callbacks to populate deques
                continue
            print_flag = True
            count += 1
            (img_left, img_right, img_left_depth, img_right_depth, img_top, img_top_depth,
             puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base) = result
            # 2.1 图像信息
            image_dict = dict()
            image_dict[self.args.camera_names[0]] = img_top
            image_dict[self.args.camera_names[1]] = img_left
            image_dict[self.args.camera_names[2]] = img_right
            # image_dict[self.args.camera_names[3]] = img_front

            # 2.2 从臂的信息从臂的状态 机械臂示教模式时 会自动订阅
            obs = collections.OrderedDict()  # 有序的字典
            obs['images'] = image_dict
            if self.args.use_depth_image:
                image_dict_depth = dict()
                image_dict_depth[self.args.camera_names[0]] = img_top_depth
                image_dict_depth[self.args.camera_names[1]] = img_left_depth
                image_dict_depth[self.args.camera_names[2]] = img_right_depth
                # image_dict_depth[self.args.camera_names[3]]=img_front_depth
                obs['images_depth'] = image_dict_depth
            print(Style.RESET_ALL)#  restart color format
            if np.array(puppet_arm_left.position).sum() < 0.001:
                print(Fore.GREEN,">>>>>>>>>>>>>>>>>>>>>>>>>>puppet arm left acion is zero>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(np.array(puppet_arm_left.position))
                # exit(0)
            if np.array(puppet_arm_right.position).sum() < 0.001:
                print(Fore.GREEN,">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>puppet right arm qpos is zero>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(np.array(puppet_arm_right.position))
                # exit(0)
            obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)),
                                         axis=0)
            obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)),
                                         axis=0)
            obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)),
                                           axis=0)
            if self.args.use_robot_base:
                obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
                print("base", obs['base_vel'])
            else:
                obs['base_vel'] = [0.0, 0.0]

            # 第一帧 只包含first， fisrt只保存StepType.FIRST
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                timesteps.append(ts)
                continue

            # 时间步
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            # 主臂保存状态
            action = np.concatenate((np.array(master_arm_left.position), np.array(master_arm_right.position)), axis=0)
            # print(Fore.YELLOW,"left action:\n", action[0:7])
            # print(Fore.GREEN, "right action:\n", action[7:])
            actions.append(action)
            if len(actions) >100:
                text_action=np.array(actions[-30:])
                print(Fore.RED,"arm state not change >>",np.all(text_action==text_action[0]),">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(Fore.GREEN,"mean",np.sum(text_action-np.mean(text_action,axis=0)))
                print(Fore.RED,"left var",np.var(text_action[:,0:7]))
                print(Fore.RED, "right var", np.var(text_action[:, 7:]))
            timesteps.append(ts)
            print(Fore.BLUE,f"Frame data: {count}, current subtask: {SUBTASK_STEP}")
            print(Style.RESET_ALL)
            if not rclpy.ok():
                exit(-1)
            rate.sleep()

        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        return timesteps, actions


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)

    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=500, required=False)

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)

    #  topic name of color image
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera/front/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera/left/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera/right/color/image_raw', required=False)
    parser.add_argument('--img_top_topic', action='store', type=str, help='img_top_topic',
                        default='/camera/top/color/image_raw', required=False)

    # topic name of depth image
    parser.add_argument('--img_top_depth_topic', action='store', type=str, help='img_top_depth_topic',
                        default='/camera/top/depth/image_rect_raw', required=False)
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera/front/depth/image_rect_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera/left/depth/image_rect_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera/right/depth/image_rect_raw', required=False)

    # topic name of arm
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/joint_states_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/joint_states_right', required=False)

    # topic name of robot_base
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom', required=False)

    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)

    # collect depth image
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    parser.add_argument('--language_raw', type=str, help="task instruction", default="None")
    parser.add_argument('--num_joints', action='store', type=int, help='Total number of joints (both arms)',
                        default=12, required=False)

    args = parser.parse_args()
    return args


def main():
    listener_thread = threading.Thread(target=listen_for_keyboard)
    listener_thread.start()

    rclpy.init()
    args = get_arguments()
    ros_operator = RosOperator(args)

    # Don't use a separate spin thread - spin_once is called in process()
    # Wait a bit for initial messages to arrive
    print("Waiting for initial messages...")
    for i in range(50):
        rclpy.spin_once(ros_operator, timeout_sec=0.1)
    print("Starting data collection...")

    timesteps, actions = ros_operator.process()
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)

    # if(len(actions) < args.max_timesteps):
    #     print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %args.max_timesteps)
    #     exit(-1)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, "episode_" + str(args.episode_idx))
    save_data(args, timesteps, actions, dataset_path)

    ros_operator.destroy_node()
    rclpy.shutdown()
    listener_thread.join()


if __name__ == '__main__':
    def signal_handler(sig, frame):
        print("Signal received, shutting down!")
        print("remember to delete the uncompleted data!")
        print("\033[31m\nenter ESC to exit")
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    main()

# python collect_data_ros2.py --dataset_dir ~/data --max_timesteps 500 --episode_idx 1
