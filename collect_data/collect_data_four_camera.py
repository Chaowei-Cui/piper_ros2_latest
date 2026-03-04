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
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from piper_msgs.msg import PiperStatusMsg
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


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {value}')


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
    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
        '/observations/end_pose': [],
        '/observations/arm_status': [],
        '/observations/frame_timestamp': [],
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
        data_dict['/observations/end_pose'].append(ts.observation['end_pose'])
        data_dict['/observations/arm_status'].append(ts.observation['arm_status'])
        data_dict['/observations/frame_timestamp'].append(ts.observation['frame_timestamp'])

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

        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = obs.create_dataset('end_pose', (data_size, 14))
        _ = obs.create_dataset('arm_status', (data_size, 38), dtype='int64')
        _ = obs.create_dataset('frame_timestamp', (data_size, 1))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))
        _ = root.create_dataset('subtask', (data_size, 1), dtype='uint8')

        # data_dict write into h5py.File
        for name, array in data_dict.items():
            root[name][...] = array
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)


class RosOperator(Node):
    def __init__(self, args):
        super().__init__('record_episodes')
        self.args = args
        self.bridge = CvBridge()
        self.qos_sensor = qos_profile_sensor_data
        self.qos_reliable = QoSProfile(depth=MAX_DEQUE)
        self.subscribers = []
        self.arm_status_fields = [
            'ctrl_mode', 'arm_status', 'mode_feedback', 'teach_status', 'motion_status',
            'trajectory_num', 'err_code', 'joint_1_angle_limit', 'joint_2_angle_limit',
            'joint_3_angle_limit', 'joint_4_angle_limit', 'joint_5_angle_limit',
            'joint_6_angle_limit', 'communication_status_joint_1',
            'communication_status_joint_2', 'communication_status_joint_3',
            'communication_status_joint_4', 'communication_status_joint_5',
            'communication_status_joint_6'
        ]

        self.img_left_deque = deque(maxlen=MAX_DEQUE)
        self.img_right_deque = deque(maxlen=MAX_DEQUE)
        self.img_top_deque = deque(maxlen=MAX_DEQUE)
        self.img_left_depth_deque = deque(maxlen=MAX_DEQUE)
        self.img_right_depth_deque = deque(maxlen=MAX_DEQUE)
        self.img_top_depth_deque = deque(maxlen=MAX_DEQUE)

        self.joint_states_left_deque = deque(maxlen=MAX_DEQUE)
        self.joint_states_right_deque = deque(maxlen=MAX_DEQUE)
        self.joint_left_deque = deque(maxlen=MAX_DEQUE)
        self.joint_right_deque = deque(maxlen=MAX_DEQUE)
        self.end_pose_left_deque = deque(maxlen=MAX_DEQUE)
        self.end_pose_right_deque = deque(maxlen=MAX_DEQUE)
        self.arm_status_left_deque = deque(maxlen=MAX_DEQUE)
        self.arm_status_right_deque = deque(maxlen=MAX_DEQUE)
        self.robot_base_deque = deque(maxlen=MAX_DEQUE)
        self.init_ros()

    def _now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _stamp_to_sec(stamp):
        if hasattr(stamp, 'to_sec'):
            return stamp.to_sec()
        if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return None

    def _append_with_stamp(self, topic_deque, msg, use_now=False):
        stamp = None
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = self._stamp_to_sec(msg.header.stamp)
        if use_now or stamp is None or stamp <= 0:
            stamp = self._now_sec()
        topic_deque.append((stamp, msg))

    def _latest_stamp(self, topic_deque):
        return topic_deque[-1][0]

    def _pop_aligned_msg(self, topic_deque, frame_time):
        while topic_deque and topic_deque[0][0] < frame_time:
            topic_deque.popleft()
        if not topic_deque:
            return None
        return topic_deque.popleft()[1]

    def _normalize_joint_array(self, arr_like, target_len=7):
        arr = np.asarray(arr_like, dtype=np.float64)
        if arr.size >= target_len:
            return arr[:target_len]
        out = np.zeros(target_len, dtype=np.float64)
        out[:arr.size] = arr
        return out

    def _pose_to_np(self, pose_msg):
        return np.array([
            pose_msg.position.x, pose_msg.position.y, pose_msg.position.z,
            pose_msg.orientation.x, pose_msg.orientation.y,
            pose_msg.orientation.z, pose_msg.orientation.w
        ], dtype=np.float64)

    def _arm_status_to_np(self, status_msg):
        values = [int(getattr(status_msg, field)) for field in self.arm_status_fields]
        return np.asarray(values, dtype=np.int64)

    def get_frame(self):
        required_deques = [
            self.img_left_deque, self.img_right_deque, self.img_top_deque,
            self.joint_states_left_deque, self.joint_states_right_deque,
            self.joint_left_deque, self.joint_right_deque,
            self.end_pose_left_deque, self.end_pose_right_deque,
            self.arm_status_left_deque, self.arm_status_right_deque
        ]
        if self.args.use_depth_image:
            required_deques.extend([
                self.img_left_depth_deque,
                self.img_right_depth_deque,
                self.img_top_depth_deque,
            ])
        if self.args.use_robot_base:
            required_deques.append(self.robot_base_deque)

        if any(len(topic_deque) == 0 for topic_deque in required_deques):
            return False

        frame_time = min(self._latest_stamp(topic_deque) for topic_deque in required_deques)

        img_left_msg = self._pop_aligned_msg(self.img_left_deque, frame_time)
        img_right_msg = self._pop_aligned_msg(self.img_right_deque, frame_time)
        img_top_msg = self._pop_aligned_msg(self.img_top_deque, frame_time)
        joint_states_left = self._pop_aligned_msg(self.joint_states_left_deque, frame_time)
        joint_states_right = self._pop_aligned_msg(self.joint_states_right_deque, frame_time)
        joint_left = self._pop_aligned_msg(self.joint_left_deque, frame_time)
        joint_right = self._pop_aligned_msg(self.joint_right_deque, frame_time)
        end_pose_left = self._pop_aligned_msg(self.end_pose_left_deque, frame_time)
        end_pose_right = self._pop_aligned_msg(self.end_pose_right_deque, frame_time)
        arm_status_left = self._pop_aligned_msg(self.arm_status_left_deque, frame_time)
        arm_status_right = self._pop_aligned_msg(self.arm_status_right_deque, frame_time)

        if any(item is None for item in [
            img_left_msg, img_right_msg, img_top_msg, joint_states_left,
            joint_states_right, joint_left, joint_right, end_pose_left,
            end_pose_right, arm_status_left, arm_status_right
        ]):
            return False

        img_left = self.bridge.imgmsg_to_cv2(img_left_msg, 'passthrough')
        img_right = self.bridge.imgmsg_to_cv2(img_right_msg, 'passthrough')
        img_top = self.bridge.imgmsg_to_cv2(img_top_msg, 'passthrough')

        img_left_depth, img_right_depth, img_top_depth = None, None, None
        if self.args.use_depth_image:
            img_left_depth_msg = self._pop_aligned_msg(self.img_left_depth_deque, frame_time)
            img_right_depth_msg = self._pop_aligned_msg(self.img_right_depth_deque, frame_time)
            img_top_depth_msg = self._pop_aligned_msg(self.img_top_depth_deque, frame_time)
            if img_left_depth_msg is None or img_right_depth_msg is None or img_top_depth_msg is None:
                return False

            img_left_depth = self.bridge.imgmsg_to_cv2(img_left_depth_msg, 'passthrough')
            img_right_depth = self.bridge.imgmsg_to_cv2(img_right_depth_msg, 'passthrough')
            img_top_depth = self.bridge.imgmsg_to_cv2(img_top_depth_msg, 'passthrough')
            top, bottom, left, right = 40, 40, 0, 0
            img_left_depth = cv2.copyMakeBorder(img_left_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            img_right_depth = cv2.copyMakeBorder(img_right_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            img_top_depth = cv2.copyMakeBorder(img_top_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        robot_base = None
        if self.args.use_robot_base:
            robot_base = self._pop_aligned_msg(self.robot_base_deque, frame_time)
            if robot_base is None:
                return False

        return (
            frame_time, img_left, img_right, img_left_depth, img_right_depth, img_top, img_top_depth,
            joint_states_left, joint_states_right, joint_left, joint_right,
            end_pose_left, end_pose_right, arm_status_left, arm_status_right, robot_base
        )

    def img_left_callback(self, msg):
        self._append_with_stamp(self.img_left_deque, msg)

    def img_top_callback(self, msg):
        self._append_with_stamp(self.img_top_deque, msg)

    def img_top_depth_callback(self, msg):
        self._append_with_stamp(self.img_top_depth_deque, msg)

    def img_right_callback(self, msg):
        self._append_with_stamp(self.img_right_deque, msg)

    def img_left_depth_callback(self, msg):
        self._append_with_stamp(self.img_left_depth_deque, msg)

    def img_right_depth_callback(self, msg):
        self._append_with_stamp(self.img_right_depth_deque, msg)

    def joint_states_left_callback(self, msg):
        self._append_with_stamp(self.joint_states_left_deque, msg)

    def joint_states_right_callback(self, msg):
        self._append_with_stamp(self.joint_states_right_deque, msg)

    def joint_left_callback(self, msg):
        self._append_with_stamp(self.joint_left_deque, msg)

    def joint_right_callback(self, msg):
        self._append_with_stamp(self.joint_right_deque, msg)

    def end_pose_left_callback(self, msg):
        self._append_with_stamp(self.end_pose_left_deque, msg, use_now=True)

    def end_pose_right_callback(self, msg):
        self._append_with_stamp(self.end_pose_right_deque, msg, use_now=True)

    def arm_status_left_callback(self, msg):
        self._append_with_stamp(self.arm_status_left_deque, msg, use_now=True)

    def arm_status_right_callback(self, msg):
        self._append_with_stamp(self.arm_status_right_deque, msg, use_now=True)

    def robot_base_callback(self, msg):
        self._append_with_stamp(self.robot_base_deque, msg)

    def init_ros(self):
        self.subscribers.append(self.create_subscription(Image, self.args.img_left_topic, self.img_left_callback, self.qos_sensor))
        self.subscribers.append(self.create_subscription(Image, self.args.img_right_topic, self.img_right_callback, self.qos_sensor))
        self.subscribers.append(self.create_subscription(Image, self.args.img_top_topic, self.img_top_callback, self.qos_sensor))

        self.get_logger().info(f'img_left_topic: {self.args.img_left_topic}')
        self.get_logger().info(f'img_right_topic: {self.args.img_right_topic}')
        self.get_logger().info(f'img_top_topic: {self.args.img_top_topic}')

        if self.args.use_depth_image:
            self.subscribers.append(self.create_subscription(Image, self.args.img_left_depth_topic, self.img_left_depth_callback, self.qos_sensor))
            self.subscribers.append(self.create_subscription(Image, self.args.img_right_depth_topic, self.img_right_depth_callback, self.qos_sensor))
            self.subscribers.append(self.create_subscription(Image, self.args.img_top_depth_topic, self.img_top_depth_callback, self.qos_sensor))
            self.get_logger().info(f'img_left_depth_topic: {self.args.img_left_depth_topic}')
            self.get_logger().info(f'img_right_depth_topic: {self.args.img_right_depth_topic}')
            self.get_logger().info(f'img_top_depth_topic: {self.args.img_top_depth_topic}')

        self.subscribers.append(self.create_subscription(JointState, self.args.joint_states_left_topic, self.joint_states_left_callback, self.qos_reliable))
        self.subscribers.append(self.create_subscription(JointState, self.args.joint_states_right_topic, self.joint_states_right_callback, self.qos_reliable))
        self.subscribers.append(self.create_subscription(JointState, self.args.joint_left_topic, self.joint_left_callback, self.qos_reliable))
        self.subscribers.append(self.create_subscription(JointState, self.args.joint_right_topic, self.joint_right_callback, self.qos_reliable))
        self.subscribers.append(self.create_subscription(Pose, self.args.end_pose_left_topic, self.end_pose_left_callback, self.qos_reliable))
        self.subscribers.append(self.create_subscription(Pose, self.args.end_pose_right_topic, self.end_pose_right_callback, self.qos_reliable))
        self.subscribers.append(self.create_subscription(PiperStatusMsg, self.args.arm_status_left_topic, self.arm_status_left_callback, self.qos_reliable))
        self.subscribers.append(self.create_subscription(PiperStatusMsg, self.args.arm_status_right_topic, self.arm_status_right_callback, self.qos_reliable))

        if self.args.use_robot_base:
            self.subscribers.append(self.create_subscription(Odometry, self.args.robot_base_topic, self.robot_base_callback, self.qos_reliable))
            self.get_logger().info(f'robot_base_topic: {self.args.robot_base_topic}')

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
        # while input_key != 's' and not rclpy.ok():
        #     input_key = input("please input s:")

        frame_sleep = 1.0 / max(1, self.args.frame_rate)
        print_flag = True
        global CUR_STEP

        while ((count < self.args.max_timesteps + 1) and not saveData) and rclpy.ok():
            # 2 收集数据
            # print(Back.BLUE) # Set background color red
            result = self.get_frame()
            CUR_STEP = count

            if not result:
                if print_flag:
                    print(Fore.RED,">>>>>>>>>>>>>>>>>>>>>>>>>>syn fail>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print_flag = False
                # rate.sleep()
                continue
            print_flag = True
            count += 1
            (frame_time, img_left, img_right, img_left_depth, img_right_depth, img_top, img_top_depth,
             joint_states_left, joint_states_right, joint_left, joint_right,
             end_pose_left, end_pose_right, arm_status_left, arm_status_right, robot_base) = result
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
            joint_left_pos = self._normalize_joint_array(joint_left.position)
            joint_right_pos = self._normalize_joint_array(joint_right.position)
            if np.array(joint_left_pos).sum() < 0.001:
                print(Fore.GREEN,">>>>>>>>>>>>>>>>>>>>>>>>>>joint_left action is zero>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(joint_left_pos)
                # exit(0)
            if np.array(joint_right_pos).sum() < 0.001:
                print(Fore.GREEN,">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>joint_right qpos is zero>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(joint_right_pos)
                # exit(0)
            obs['qpos'] = np.concatenate((
                self._normalize_joint_array(joint_states_left.position),
                self._normalize_joint_array(joint_states_right.position)
            ), axis=0)
            obs['qvel'] = np.concatenate((
                self._normalize_joint_array(joint_states_left.velocity),
                self._normalize_joint_array(joint_states_right.velocity)
            ), axis=0)
            obs['effort'] = np.concatenate((
                self._normalize_joint_array(joint_states_left.effort),
                self._normalize_joint_array(joint_states_right.effort)
            ), axis=0)
            obs['end_pose'] = np.concatenate((self._pose_to_np(end_pose_left), self._pose_to_np(end_pose_right)), axis=0)
            obs['arm_status'] = np.concatenate(
                (self._arm_status_to_np(arm_status_left), self._arm_status_to_np(arm_status_right)),
                axis=0
            )
            obs['frame_timestamp'] = np.array([frame_time], dtype=np.float64)
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

            # action 保存 /joint_left + /joint_right
            action = np.concatenate((joint_left_pos, joint_right_pos), axis=0)
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
            time.sleep(frame_sleep)

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

    parser.add_argument('--camera_names', nargs='+', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)

    #  topic name of color image
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic (unused in current script)',
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
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic (unused in current script)',
                        default='/camera/front/depth/image_rect_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera/left/depth/image_rect_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera/right/depth/image_rect_raw', required=False)

    # topic name of arm (left/right data)
    parser.add_argument('--joint_states_left_topic', action='store', type=str, help='joint_states_left_topic',
                        default='/joint_states_left', required=False)
    parser.add_argument('--joint_states_right_topic', action='store', type=str, help='joint_states_right_topic',
                        default='/joint_states_right', required=False)
    parser.add_argument('--joint_left_topic', action='store', type=str, help='joint_left_topic',
                        default='/joint_left', required=False)
    parser.add_argument('--joint_right_topic', action='store', type=str, help='joint_right_topic',
                        default='/joint_right', required=False)
    parser.add_argument('--end_pose_left_topic', action='store', type=str, help='end_pose_left_topic',
                        default='/end_pose_left', required=False)
    parser.add_argument('--end_pose_right_topic', action='store', type=str, help='end_pose_right_topic',
                        default='/end_pose_right', required=False)
    parser.add_argument('--arm_status_left_topic', action='store', type=str, help='arm_status_left_topic',
                        default='/arm_status_left', required=False)
    parser.add_argument('--arm_status_right_topic', action='store', type=str, help='arm_status_right_topic',
                        default='/arm_status_right', required=False)

    # topic name of robot_base
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom', required=False)

    parser.add_argument('--use_robot_base', action='store', type=str2bool, help='use_robot_base',
                        default=False, required=False)

    # collect depth image
    parser.add_argument('--use_depth_image', action='store', type=str2bool, help='use_depth_image',
                        default=False, required=False)

    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    parser.add_argument('--language_raw', type=str, help="task instruction", default="None")

    args = parser.parse_args()
    return args


def main():
    listener_thread = threading.Thread(target=listen_for_keyboard)
    listener_thread.start()

    args = get_arguments()
    rclpy.init(args=None)
    ros_operator = RosOperator(args)
    executor = SingleThreadedExecutor()
    executor.add_node(ros_operator)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    timesteps, actions = [], []
    try:
        timesteps, actions = ros_operator.process()
        dataset_dir = os.path.join(args.dataset_dir, args.task_name)

        # if(len(actions) < args.max_timesteps):
        #     print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %args.max_timesteps)
        #     exit(-1)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        dataset_path = os.path.join(dataset_dir, "episode_" + str(args.episode_idx))
        save_data(args, timesteps, actions, dataset_path)
    finally:
        executor.shutdown()
        ros_operator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        listener_thread.join()


if __name__ == '__main__':
    def signal_handler(sig, frame):
        global saveData
        print("Signal received, shutting down!")
        print("remember to delete the uncompleted data!")
        print("\033[31m\nenter ESC to exit")
        saveData = True
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    main()

# python collect_data.py --dataset_dir ~/data --max_timesteps 500 --episode_idx1
