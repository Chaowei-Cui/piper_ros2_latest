# -- coding: UTF-8
import argparse
import collections
import os
import signal
import sys
import threading
import time
from collections import deque

import cv2
import dm_env
import h5py
import numpy as np
import rclpy
from colorama import Back, Fore, Style, init
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from piper_msgs.msg import PiperStatusMsg
from pynput import keyboard
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

MAX_DEQUE = 10
saveData = False
SUBTASK_STEP = 0
DEQUE_SIZE = 10
CUR_STEP = 0
PRE_STEP = 0
CUR_TIME_STAMP = time.time()
PRE_TIME_STAMP = time.time()
SUBTASK_FLAG = np.zeros((10000, 1))


def onPress(key):
    global saveData
    if key == keyboard.Key.enter:
        saveData = True
    if key == keyboard.Key.esc:
        return False


def onRelease(key):
    global CUR_STEP, CUR_TIME_STAMP, PRE_TIME_STAMP, SUBTASK_FLAG, SUBTASK_STEP, PRE_STEP
    if key == keyboard.Key.scroll_lock:
        PRE_TIME_STAMP = CUR_TIME_STAMP
        CUR_TIME_STAMP = time.time()
        if CUR_TIME_STAMP - PRE_TIME_STAMP < 0.4:
            print(f">>>>>Double steps occur at {PRE_STEP} && {CUR_STEP} | Time are {CUR_TIME_STAMP} | {PRE_TIME_STAMP} <<<<<<")
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


def _status_to_vec(msg):
    return np.array([
        msg.ctrl_mode,
        msg.arm_status,
        msg.mode_feedback,
        msg.teach_status,
        msg.motion_status,
        msg.trajectory_num,
        msg.err_code,
        int(msg.joint_1_angle_limit),
        int(msg.joint_2_angle_limit),
        int(msg.joint_3_angle_limit),
        int(msg.joint_4_angle_limit),
        int(msg.joint_5_angle_limit),
        int(msg.joint_6_angle_limit),
        int(msg.communication_status_joint_1),
        int(msg.communication_status_joint_2),
        int(msg.communication_status_joint_3),
        int(msg.communication_status_joint_4),
        int(msg.communication_status_joint_5),
        int(msg.communication_status_joint_6),
    ], dtype=np.int64)


def _pose_to_vec(msg):
    return np.array([
        msg.position.x,
        msg.position.y,
        msg.position.z,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w,
    ], dtype=np.float32)


# 保存数据函数
def save_data(args, timesteps, actions, dataset_path):
    data_size = len(actions)
    global SUBTASK_FLAG

    if data_size == 0:
        raise RuntimeError('No action data collected; nothing to save.')

    first_obs = timesteps[0].observation
    qpos_dim = int(np.asarray(first_obs['qpos']).shape[0])
    action_dim = int(np.asarray(actions[0]).shape[0])
    js_left_dim = int(np.asarray(first_obs['joint_states_left_pos']).shape[0])
    js_right_dim = int(np.asarray(first_obs['joint_states_right_pos']).shape[0])
    j_left_dim = int(np.asarray(first_obs['joint_left_pos']).shape[0])
    j_right_dim = int(np.asarray(first_obs['joint_right_pos']).shape[0])

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
        '/subtask': SUBTASK_FLAG[:data_size, :],

        '/arm/joint_states_left/position': [],
        '/arm/joint_states_left/velocity': [],
        '/arm/joint_states_left/effort': [],
        '/arm/joint_states_right/position': [],
        '/arm/joint_states_right/velocity': [],
        '/arm/joint_states_right/effort': [],
        '/arm/joint_left/position': [],
        '/arm/joint_left/velocity': [],
        '/arm/joint_left/effort': [],
        '/arm/joint_right/position': [],
        '/arm/joint_right/velocity': [],
        '/arm/joint_right/effort': [],
        '/arm/end_pose_left': [],
        '/arm/end_pose_right': [],
        '/arm/arm_status_left': [],
        '/arm/arm_status_right': [],
    }

    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)

        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        data_dict['/arm/joint_states_left/position'].append(ts.observation['joint_states_left_pos'])
        data_dict['/arm/joint_states_left/velocity'].append(ts.observation['joint_states_left_vel'])
        data_dict['/arm/joint_states_left/effort'].append(ts.observation['joint_states_left_eff'])
        data_dict['/arm/joint_states_right/position'].append(ts.observation['joint_states_right_pos'])
        data_dict['/arm/joint_states_right/velocity'].append(ts.observation['joint_states_right_vel'])
        data_dict['/arm/joint_states_right/effort'].append(ts.observation['joint_states_right_eff'])

        data_dict['/arm/joint_left/position'].append(ts.observation['joint_left_pos'])
        data_dict['/arm/joint_left/velocity'].append(ts.observation['joint_left_vel'])
        data_dict['/arm/joint_left/effort'].append(ts.observation['joint_left_eff'])
        data_dict['/arm/joint_right/position'].append(ts.observation['joint_right_pos'])
        data_dict['/arm/joint_right/velocity'].append(ts.observation['joint_right_vel'])
        data_dict['/arm/joint_right/effort'].append(ts.observation['joint_right_eff'])

        data_dict['/arm/end_pose_left'].append(ts.observation['end_pose_left'])
        data_dict['/arm/end_pose_right'].append(ts.observation['end_pose_right'])
        data_dict['/arm/arm_status_left'].append(ts.observation['arm_status_left'])
        data_dict['/arm/arm_status_right'].append(ts.observation['arm_status_right'])

        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        obs = root.create_group('observations')
        image = obs.create_group('images')
        if 'language_raw' not in root.keys():
            root.create_dataset('language_raw', data=[args.language_raw])

        for cam_name in args.camera_names:
            image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))

        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint16', chunks=(1, 480, 640))

        obs.create_dataset('qpos', (data_size, qpos_dim))
        obs.create_dataset('qvel', (data_size, qpos_dim))
        obs.create_dataset('effort', (data_size, qpos_dim))
        root.create_dataset('action', (data_size, action_dim))
        root.create_dataset('base_action', (data_size, 2))
        root.create_dataset('subtask', (data_size, 1), dtype='uint8')

        arm = root.create_group('arm')
        js_left = arm.create_group('joint_states_left')
        js_left.create_dataset('position', (data_size, js_left_dim))
        js_left.create_dataset('velocity', (data_size, js_left_dim))
        js_left.create_dataset('effort', (data_size, js_left_dim))

        js_right = arm.create_group('joint_states_right')
        js_right.create_dataset('position', (data_size, js_right_dim))
        js_right.create_dataset('velocity', (data_size, js_right_dim))
        js_right.create_dataset('effort', (data_size, js_right_dim))

        j_left = arm.create_group('joint_left')
        j_left.create_dataset('position', (data_size, j_left_dim))
        j_left.create_dataset('velocity', (data_size, j_left_dim))
        j_left.create_dataset('effort', (data_size, j_left_dim))

        j_right = arm.create_group('joint_right')
        j_right.create_dataset('position', (data_size, j_right_dim))
        j_right.create_dataset('velocity', (data_size, j_right_dim))
        j_right.create_dataset('effort', (data_size, j_right_dim))

        arm.create_dataset('end_pose_left', (data_size, 7), dtype='float32')
        arm.create_dataset('end_pose_right', (data_size, 7), dtype='float32')
        arm.create_dataset('arm_status_left', (data_size, 19), dtype='int64')
        arm.create_dataset('arm_status_right', (data_size, 19), dtype='int64')

        for name, array in data_dict.items():
            root[name][...] = array

    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. {dataset_path} \033[0m\n')


class RosOperator(Node):
    def __init__(self, args):
        super().__init__('record_episodes')
        self.args = args
        self.bridge = CvBridge()

        self.img_left_deque = deque(maxlen=MAX_DEQUE)
        self.img_right_deque = deque(maxlen=MAX_DEQUE)
        self.img_top_deque = deque(maxlen=MAX_DEQUE)
        self.img_left_depth_deque = deque(maxlen=MAX_DEQUE)
        self.img_right_depth_deque = deque(maxlen=MAX_DEQUE)
        self.img_top_depth_deque = deque(maxlen=MAX_DEQUE)

        self.joint_left_deque = deque(maxlen=MAX_DEQUE)
        self.joint_right_deque = deque(maxlen=MAX_DEQUE)
        self.joint_states_left_deque = deque(maxlen=MAX_DEQUE)
        self.joint_states_right_deque = deque(maxlen=MAX_DEQUE)

        self.end_pose_left_deque = deque(maxlen=MAX_DEQUE)
        self.end_pose_right_deque = deque(maxlen=MAX_DEQUE)
        self.arm_status_left_deque = deque(maxlen=MAX_DEQUE)
        self.arm_status_right_deque = deque(maxlen=MAX_DEQUE)
        self.robot_base_deque = deque(maxlen=MAX_DEQUE)

        self.init_ros()

    @staticmethod
    def _stamp_to_sec(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    @staticmethod
    def _pop_until_time(dq, frame_time, time_fn):
        while len(dq) > 1 and time_fn(dq[0]) < frame_time:
            dq.popleft()
        return dq.popleft()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_top_deque) == 0:
            return False

        if len(self.joint_left_deque) == 0 or len(self.joint_right_deque) == 0:
            return False

        if len(self.joint_states_left_deque) == 0 or len(self.joint_states_right_deque) == 0:
            return False

        if len(self.end_pose_left_deque) == 0 or len(self.end_pose_right_deque) == 0:
            return False

        if len(self.arm_status_left_deque) == 0 or len(self.arm_status_right_deque) == 0:
            return False

        if self.args.use_depth_image and (
            len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_top_depth_deque) == 0
        ):
            return False

        frame_candidates = [
            self._stamp_to_sec(self.img_left_deque[-1]),
            self._stamp_to_sec(self.img_right_deque[-1]),
            self._stamp_to_sec(self.img_top_deque[-1]),
            self._stamp_to_sec(self.joint_left_deque[-1]),
            self._stamp_to_sec(self.joint_right_deque[-1]),
            self._stamp_to_sec(self.joint_states_left_deque[-1]),
            self._stamp_to_sec(self.joint_states_right_deque[-1]),
        ]

        if self.args.use_depth_image:
            frame_candidates.extend([
                self._stamp_to_sec(self.img_left_depth_deque[-1]),
                self._stamp_to_sec(self.img_right_depth_deque[-1]),
                self._stamp_to_sec(self.img_top_depth_deque[-1]),
            ])

        if self.args.use_robot_base and len(self.robot_base_deque) > 0:
            frame_candidates.append(self._stamp_to_sec(self.robot_base_deque[-1]))

        frame_time = min(frame_candidates)

        img_left_msg = self._pop_until_time(self.img_left_deque, frame_time, self._stamp_to_sec)
        img_right_msg = self._pop_until_time(self.img_right_deque, frame_time, self._stamp_to_sec)
        img_top_msg = self._pop_until_time(self.img_top_deque, frame_time, self._stamp_to_sec)

        joint_left_msg = self._pop_until_time(self.joint_left_deque, frame_time, self._stamp_to_sec)
        joint_right_msg = self._pop_until_time(self.joint_right_deque, frame_time, self._stamp_to_sec)
        joint_states_left_msg = self._pop_until_time(self.joint_states_left_deque, frame_time, self._stamp_to_sec)
        joint_states_right_msg = self._pop_until_time(self.joint_states_right_deque, frame_time, self._stamp_to_sec)

        img_left = self.bridge.imgmsg_to_cv2(img_left_msg, 'passthrough')
        img_right = self.bridge.imgmsg_to_cv2(img_right_msg, 'passthrough')
        img_top = self.bridge.imgmsg_to_cv2(img_top_msg, 'passthrough')

        img_left_depth = None
        img_right_depth = None
        img_top_depth = None
        if self.args.use_depth_image:
            img_left_depth_msg = self._pop_until_time(self.img_left_depth_deque, frame_time, self._stamp_to_sec)
            img_right_depth_msg = self._pop_until_time(self.img_right_depth_deque, frame_time, self._stamp_to_sec)
            img_top_depth_msg = self._pop_until_time(self.img_top_depth_deque, frame_time, self._stamp_to_sec)

            img_left_depth = self.bridge.imgmsg_to_cv2(img_left_depth_msg, 'passthrough')
            img_right_depth = self.bridge.imgmsg_to_cv2(img_right_depth_msg, 'passthrough')
            img_top_depth = self.bridge.imgmsg_to_cv2(img_top_depth_msg, 'passthrough')

            top, bottom, left, right = 40, 40, 0, 0
            img_left_depth = cv2.copyMakeBorder(img_left_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            img_right_depth = cv2.copyMakeBorder(img_right_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            img_top_depth = cv2.copyMakeBorder(img_top_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        robot_base = None
        if self.args.use_robot_base and len(self.robot_base_deque) > 0:
            robot_base = self._pop_until_time(self.robot_base_deque, frame_time, self._stamp_to_sec)

        end_pose_left = self.end_pose_left_deque[-1]
        end_pose_right = self.end_pose_right_deque[-1]
        arm_status_left = self.arm_status_left_deque[-1]
        arm_status_right = self.arm_status_right_deque[-1]

        return (
            img_left,
            img_right,
            img_left_depth,
            img_right_depth,
            img_top,
            img_top_depth,
            joint_states_left_msg,
            joint_states_right_msg,
            joint_left_msg,
            joint_right_msg,
            end_pose_left,
            end_pose_right,
            arm_status_left,
            arm_status_right,
            robot_base,
        )

    def img_left_callback(self, msg):
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        self.img_right_deque.append(msg)

    def img_top_callback(self, msg):
        self.img_top_deque.append(msg)

    def img_left_depth_callback(self, msg):
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        self.img_right_depth_deque.append(msg)

    def img_top_depth_callback(self, msg):
        self.img_top_depth_deque.append(msg)

    def joint_left_callback(self, msg):
        self.joint_left_deque.append(msg)

    def joint_right_callback(self, msg):
        self.joint_right_deque.append(msg)

    def joint_states_left_callback(self, msg):
        self.joint_states_left_deque.append(msg)

    def joint_states_right_callback(self, msg):
        self.joint_states_right_deque.append(msg)

    def end_pose_left_callback(self, msg):
        self.end_pose_left_deque.append(msg)

    def end_pose_right_callback(self, msg):
        self.end_pose_right_deque.append(msg)

    def arm_status_left_callback(self, msg):
        self.arm_status_left_deque.append(msg)

    def arm_status_right_callback(self, msg):
        self.arm_status_right_deque.append(msg)

    def robot_base_callback(self, msg):
        self.robot_base_deque.append(msg)

    def init_ros(self):
        self.create_subscription(Image, self.args.img_left_topic, self.img_left_callback, MAX_DEQUE)
        self.create_subscription(Image, self.args.img_right_topic, self.img_right_callback, MAX_DEQUE)
        self.create_subscription(Image, self.args.img_top_topic, self.img_top_callback, MAX_DEQUE)

        if self.args.use_depth_image:
            self.create_subscription(Image, self.args.img_left_depth_topic, self.img_left_depth_callback, MAX_DEQUE)
            self.create_subscription(Image, self.args.img_right_depth_topic, self.img_right_depth_callback, MAX_DEQUE)
            self.create_subscription(Image, self.args.img_top_depth_topic, self.img_top_depth_callback, MAX_DEQUE)

        self.create_subscription(JointState, self.args.joint_left_topic, self.joint_left_callback, MAX_DEQUE)
        self.create_subscription(JointState, self.args.joint_right_topic, self.joint_right_callback, MAX_DEQUE)
        self.create_subscription(JointState, self.args.joint_states_left_topic, self.joint_states_left_callback, MAX_DEQUE)
        self.create_subscription(JointState, self.args.joint_states_right_topic, self.joint_states_right_callback, MAX_DEQUE)

        self.create_subscription(Pose, self.args.end_pose_left_topic, self.end_pose_left_callback, MAX_DEQUE)
        self.create_subscription(Pose, self.args.end_pose_right_topic, self.end_pose_right_callback, MAX_DEQUE)
        self.create_subscription(PiperStatusMsg, self.args.arm_status_left_topic, self.arm_status_left_callback, MAX_DEQUE)
        self.create_subscription(PiperStatusMsg, self.args.arm_status_right_topic, self.arm_status_right_callback, MAX_DEQUE)

        if self.args.use_robot_base:
            self.create_subscription(Odometry, self.args.robot_base_topic, self.robot_base_callback, MAX_DEQUE)

    def process(self):
        timesteps = []
        actions = []

        rate = self.create_rate(self.args.frame_rate)
        print_flag = True
        global CUR_STEP

        count = 0
        while ((count < self.args.max_timesteps + 1) and not saveData) and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)
            result = self.get_frame()
            CUR_STEP = count

            if not result:
                if print_flag:
                    print(Fore.RED, ">>>>>>>>>>>>>>>>>>>>>>>>>>syn fail>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print_flag = False
                time.sleep(0.01)
                continue

            print_flag = True
            count += 1
            (
                img_left,
                img_right,
                img_left_depth,
                img_right_depth,
                img_top,
                img_top_depth,
                joint_states_left,
                joint_states_right,
                joint_left,
                joint_right,
                end_pose_left,
                end_pose_right,
                arm_status_left,
                arm_status_right,
                robot_base,
            ) = result

            image_dict = {
                self.args.camera_names[0]: img_top,
                self.args.camera_names[1]: img_left,
                self.args.camera_names[2]: img_right,
            }

            obs = collections.OrderedDict()
            obs['images'] = image_dict

            if self.args.use_depth_image:
                obs['images_depth'] = {
                    self.args.camera_names[0]: img_top_depth,
                    self.args.camera_names[1]: img_left_depth,
                    self.args.camera_names[2]: img_right_depth,
                }

            print(Style.RESET_ALL)

            obs['joint_states_left_pos'] = np.array(joint_states_left.position)
            obs['joint_states_left_vel'] = np.array(joint_states_left.velocity)
            obs['joint_states_left_eff'] = np.array(joint_states_left.effort)
            obs['joint_states_right_pos'] = np.array(joint_states_right.position)
            obs['joint_states_right_vel'] = np.array(joint_states_right.velocity)
            obs['joint_states_right_eff'] = np.array(joint_states_right.effort)

            obs['joint_left_pos'] = np.array(joint_left.position)
            obs['joint_left_vel'] = np.array(joint_left.velocity)
            obs['joint_left_eff'] = np.array(joint_left.effort)
            obs['joint_right_pos'] = np.array(joint_right.position)
            obs['joint_right_vel'] = np.array(joint_right.velocity)
            obs['joint_right_eff'] = np.array(joint_right.effort)

            obs['end_pose_left'] = _pose_to_vec(end_pose_left)
            obs['end_pose_right'] = _pose_to_vec(end_pose_right)
            obs['arm_status_left'] = _status_to_vec(arm_status_left)
            obs['arm_status_right'] = _status_to_vec(arm_status_right)

            obs['qpos'] = np.concatenate((obs['joint_states_left_pos'], obs['joint_states_right_pos']), axis=0)
            obs['qvel'] = np.concatenate((obs['joint_states_left_vel'], obs['joint_states_right_vel']), axis=0)
            obs['effort'] = np.concatenate((obs['joint_states_left_eff'], obs['joint_states_right_eff']), axis=0)

            if self.args.use_robot_base and robot_base is not None:
                obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            else:
                obs['base_vel'] = [0.0, 0.0]

            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs,
                )
                timesteps.append(ts)
                continue

            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs,
            )

            action = np.concatenate((obs['joint_left_pos'], obs['joint_right_pos']), axis=0)
            actions.append(action)
            timesteps.append(ts)

            print(Fore.BLUE, f"Frame data: {count}, current subtask: {SUBTASK_STEP}")
            print(Style.RESET_ALL)

            if not rclpy.ok():
                exit(-1)
            rate.sleep()

        print('len(timesteps): ', len(timesteps))
        print('len(actions)  : ', len(actions))
        return timesteps, actions


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.', default='./data', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=0, required=False)
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.', default=500, required=False)

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)

    parser.add_argument('--img_left_topic', action='store', type=str, default='/camera/left/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, default='/camera/right/color/image_raw', required=False)
    parser.add_argument('--img_top_topic', action='store', type=str, default='/camera/top/color/image_raw', required=False)

    parser.add_argument('--img_left_depth_topic', action='store', type=str, default='/camera/left/depth/image_rect_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, default='/camera/right/depth/image_rect_raw', required=False)
    parser.add_argument('--img_top_depth_topic', action='store', type=str, default='/camera/top/depth/image_rect_raw', required=False)

    parser.add_argument('--joint_states_left_topic', action='store', type=str, default='/joint_states_left', required=False)
    parser.add_argument('--joint_states_right_topic', action='store', type=str, default='/joint_states_right', required=False)
    parser.add_argument('--joint_left_topic', action='store', type=str, default='/joint_left', required=False)
    parser.add_argument('--joint_right_topic', action='store', type=str, default='/joint_right', required=False)
    parser.add_argument('--end_pose_left_topic', action='store', type=str, default='/end_pose_left', required=False)
    parser.add_argument('--end_pose_right_topic', action='store', type=str, default='/end_pose_right', required=False)
    parser.add_argument('--arm_status_left_topic', action='store', type=str, default='/arm_status_left', required=False)
    parser.add_argument('--arm_status_right_topic', action='store', type=str, default='/arm_status_right', required=False)

    parser.add_argument('--robot_base_topic', action='store', type=str, default='/odom', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, default=False, required=False)

    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate', default=30, required=False)
    parser.add_argument('--language_raw', type=str, help='task instruction', default='None')

    return parser.parse_args()


def main():
    listener_thread = threading.Thread(target=listen_for_keyboard)
    listener_thread.start()

    rclpy.init()
    args = get_arguments()
    ros_operator = RosOperator(args)

    print('Waiting for initial messages...')
    for _ in range(50):
        rclpy.spin_once(ros_operator, timeout_sec=0.1)

    timesteps, actions = ros_operator.process()
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    dataset_path = os.path.join(dataset_dir, 'episode_' + str(args.episode_idx))
    save_data(args, timesteps, actions, dataset_path)

    ros_operator.destroy_node()
    rclpy.shutdown()
    listener_thread.join()


if __name__ == '__main__':
    def signal_handler(sig, frame):
        print('Signal received, shutting down!')
        print('remember to delete the uncompleted data!')
        print('\033[31m\nenter ESC to exit')
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    main()
