# -- coding: UTF-8
import argparse
import os
import signal
import sys
import threading
import time
from collections import deque

import cv2
import h5py
import numpy as np
from cv_bridge import CvBridge
from pynput import keyboard

import rclpy
from geometry_msgs.msg import Pose
from piper_msgs.msg import PiperStatusMsg
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState

MAX_DEQUE = 30

saveData = False
SUBTASK_STEP = 0
CUR_STEP = 0
PRE_STEP = 0
CUR_TIME_STAMP = time.time()
PRE_TIME_STAMP = time.time()
SUBTASK_FLAG = np.zeros((100000, 1), dtype=np.uint8)


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
            print(f">>>>>Double steps occur at {PRE_STEP} && {CUR_STEP} <<<<<<")
            SUBTASK_FLAG[CUR_STEP][0] = 100
            SUBTASK_FLAG[PRE_STEP][0] = 100
        else:
            print(f"One step occurs at {CUR_STEP}")
            SUBTASK_FLAG[CUR_STEP][0] = 10
            SUBTASK_STEP += 1
            PRE_STEP = CUR_STEP


def listen_for_keyboard():
    with keyboard.Listener(on_release=onRelease, on_press=onPress) as listener:
        listener.join()


def ros_stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def msg_stamp_or_now(node: Node, msg):
    if hasattr(msg, 'header') and msg.header is not None:
        stamp = msg.header.stamp
        if stamp.sec != 0 or stamp.nanosec != 0:
            return ros_stamp_to_sec(stamp)
    return node.get_clock().now().nanoseconds * 1e-9


def joint_to_fixed_vec(values, dof=7):
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[0] >= dof:
        return arr[:dof]
    out = np.zeros((dof,), dtype=np.float32)
    out[:arr.shape[0]] = arr
    return out


def pose_to_vec(msg: Pose):
    return np.array([
        msg.position.x,
        msg.position.y,
        msg.position.z,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w,
    ], dtype=np.float32)


def status_to_vec(msg: PiperStatusMsg):
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


STATUS_FIELDS = [
    'ctrl_mode',
    'arm_status',
    'mode_feedback',
    'teach_status',
    'motion_status',
    'trajectory_num',
    'err_code',
    'joint_1_angle_limit',
    'joint_2_angle_limit',
    'joint_3_angle_limit',
    'joint_4_angle_limit',
    'joint_5_angle_limit',
    'joint_6_angle_limit',
    'communication_status_joint_1',
    'communication_status_joint_2',
    'communication_status_joint_3',
    'communication_status_joint_4',
    'communication_status_joint_5',
    'communication_status_joint_6',
]


class Ros2Collector(Node):
    def __init__(self, args):
        super().__init__('record_episodes_ros2')
        self.args = args
        self.bridge = CvBridge()

        self.buffers = {
            'rgb_top': deque(maxlen=MAX_DEQUE),
            'rgb_left': deque(maxlen=MAX_DEQUE),
            'rgb_right': deque(maxlen=MAX_DEQUE),
            'depth_top': deque(maxlen=MAX_DEQUE),
            'depth_left': deque(maxlen=MAX_DEQUE),
            'depth_right': deque(maxlen=MAX_DEQUE),
            'joint_states_left': deque(maxlen=MAX_DEQUE),
            'joint_states_right': deque(maxlen=MAX_DEQUE),
            'joint_left': deque(maxlen=MAX_DEQUE),
            'joint_right': deque(maxlen=MAX_DEQUE),
            'end_pose_left': deque(maxlen=MAX_DEQUE),
            'end_pose_right': deque(maxlen=MAX_DEQUE),
            'arm_status_left': deque(maxlen=MAX_DEQUE),
            'arm_status_right': deque(maxlen=MAX_DEQUE),
        }

        self.required_streams = [
            'rgb_top',
            'rgb_left',
            'rgb_right',
            'depth_top',
            'depth_left',
            'depth_right',
            'joint_states_left',
            'joint_states_right',
            'joint_left',
            'joint_right',
            'end_pose_left',
            'end_pose_right',
            'arm_status_left',
            'arm_status_right',
        ]

        self._init_subscribers()

    def _push_header_msg(self, key, msg):
        self.buffers[key].append((msg_stamp_or_now(self, msg), msg))

    def _push_no_header_msg(self, key, msg):
        recv_ts = self.get_clock().now().nanoseconds * 1e-9
        self.buffers[key].append((recv_ts, msg))

    def _init_subscribers(self):
        sensor_qos = qos_profile_sensor_data
        default_qos = QoSProfile(depth=MAX_DEQUE)

        self.create_subscription(Image, self.args.img_top_topic, lambda m: self._push_header_msg('rgb_top', m), sensor_qos)
        self.create_subscription(Image, self.args.img_left_topic, lambda m: self._push_header_msg('rgb_left', m), sensor_qos)
        self.create_subscription(Image, self.args.img_right_topic, lambda m: self._push_header_msg('rgb_right', m), sensor_qos)

        self.create_subscription(Image, self.args.img_top_depth_topic, lambda m: self._push_header_msg('depth_top', m), sensor_qos)
        self.create_subscription(Image, self.args.img_left_depth_topic, lambda m: self._push_header_msg('depth_left', m), sensor_qos)
        self.create_subscription(Image, self.args.img_right_depth_topic, lambda m: self._push_header_msg('depth_right', m), sensor_qos)

        self.create_subscription(JointState, self.args.joint_states_left_topic,
                                 lambda m: self._push_header_msg('joint_states_left', m), sensor_qos)
        self.create_subscription(JointState, self.args.joint_states_right_topic,
                                 lambda m: self._push_header_msg('joint_states_right', m), sensor_qos)
        self.create_subscription(JointState, self.args.joint_left_topic,
                                 lambda m: self._push_header_msg('joint_left', m), sensor_qos)
        self.create_subscription(JointState, self.args.joint_right_topic,
                                 lambda m: self._push_header_msg('joint_right', m), sensor_qos)

        self.create_subscription(Pose, self.args.end_pose_left_topic,
                                 lambda m: self._push_no_header_msg('end_pose_left', m), default_qos)
        self.create_subscription(Pose, self.args.end_pose_right_topic,
                                 lambda m: self._push_no_header_msg('end_pose_right', m), default_qos)

        self.create_subscription(PiperStatusMsg, self.args.arm_status_left_topic,
                                 lambda m: self._push_no_header_msg('arm_status_left', m), default_qos)
        self.create_subscription(PiperStatusMsg, self.args.arm_status_right_topic,
                                 lambda m: self._push_no_header_msg('arm_status_right', m), default_qos)

    def get_aligned_frame(self):
        for k in self.required_streams:
            if len(self.buffers[k]) == 0:
                return None

        frame_time = min(self.buffers[k][-1][0] for k in self.required_streams)

        out = {}
        for k in self.required_streams:
            q = self.buffers[k]
            if q[-1][0] < frame_time:
                return None
            while len(q) > 0 and q[0][0] < frame_time:
                q.popleft()
            if len(q) == 0:
                return None
            out[k] = q.popleft()[1]

        return frame_time, out

    def process(self):
        data = {
            'timestamps': [],
            'images': {self.args.camera_names[0]: [], self.args.camera_names[1]: [], self.args.camera_names[2]: []},
            'images_depth': {self.args.camera_names[0]: [], self.args.camera_names[1]: [], self.args.camera_names[2]: []},
            'qpos': [],
            'qvel': [],
            'effort': [],
            'action': [],
            'joint_states_left': {'position': [], 'velocity': [], 'effort': []},
            'joint_states_right': {'position': [], 'velocity': [], 'effort': []},
            'joint_left': {'position': [], 'velocity': [], 'effort': []},
            'joint_right': {'position': [], 'velocity': [], 'effort': []},
            'end_pose_left': [],
            'end_pose_right': [],
            'arm_status_left': [],
            'arm_status_right': [],
        }

        count = 0
        global CUR_STEP

        while rclpy.ok() and (count < self.args.max_timesteps) and not saveData:
            rclpy.spin_once(self, timeout_sec=0.01)
            result = self.get_aligned_frame()
            CUR_STEP = count

            if result is None:
                continue

            frame_time, frame = result
            count += 1

            img_top = self.bridge.imgmsg_to_cv2(frame['rgb_top'], desired_encoding='passthrough')
            img_left = self.bridge.imgmsg_to_cv2(frame['rgb_left'], desired_encoding='passthrough')
            img_right = self.bridge.imgmsg_to_cv2(frame['rgb_right'], desired_encoding='passthrough')

            depth_top = self.bridge.imgmsg_to_cv2(frame['depth_top'], desired_encoding='passthrough')
            depth_left = self.bridge.imgmsg_to_cv2(frame['depth_left'], desired_encoding='passthrough')
            depth_right = self.bridge.imgmsg_to_cv2(frame['depth_right'], desired_encoding='passthrough')

            if self.args.depth_border_pad > 0:
                pad = self.args.depth_border_pad
                depth_top = cv2.copyMakeBorder(depth_top, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
                depth_left = cv2.copyMakeBorder(depth_left, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
                depth_right = cv2.copyMakeBorder(depth_right, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

            js_l = frame['joint_states_left']
            js_r = frame['joint_states_right']
            jl = frame['joint_left']
            jr = frame['joint_right']

            js_l_pos = joint_to_fixed_vec(js_l.position)
            js_l_vel = joint_to_fixed_vec(js_l.velocity)
            js_l_eff = joint_to_fixed_vec(js_l.effort)
            js_r_pos = joint_to_fixed_vec(js_r.position)
            js_r_vel = joint_to_fixed_vec(js_r.velocity)
            js_r_eff = joint_to_fixed_vec(js_r.effort)

            jl_pos = joint_to_fixed_vec(jl.position)
            jl_vel = joint_to_fixed_vec(jl.velocity)
            jl_eff = joint_to_fixed_vec(jl.effort)
            jr_pos = joint_to_fixed_vec(jr.position)
            jr_vel = joint_to_fixed_vec(jr.velocity)
            jr_eff = joint_to_fixed_vec(jr.effort)

            data['timestamps'].append(frame_time)

            data['images'][self.args.camera_names[0]].append(img_top)
            data['images'][self.args.camera_names[1]].append(img_left)
            data['images'][self.args.camera_names[2]].append(img_right)

            data['images_depth'][self.args.camera_names[0]].append(depth_top)
            data['images_depth'][self.args.camera_names[1]].append(depth_left)
            data['images_depth'][self.args.camera_names[2]].append(depth_right)

            data['qpos'].append(np.concatenate([js_l_pos, js_r_pos], axis=0))
            data['qvel'].append(np.concatenate([js_l_vel, js_r_vel], axis=0))
            data['effort'].append(np.concatenate([js_l_eff, js_r_eff], axis=0))
            data['action'].append(np.concatenate([jl_pos, jr_pos], axis=0))

            data['joint_states_left']['position'].append(js_l_pos)
            data['joint_states_left']['velocity'].append(js_l_vel)
            data['joint_states_left']['effort'].append(js_l_eff)

            data['joint_states_right']['position'].append(js_r_pos)
            data['joint_states_right']['velocity'].append(js_r_vel)
            data['joint_states_right']['effort'].append(js_r_eff)

            data['joint_left']['position'].append(jl_pos)
            data['joint_left']['velocity'].append(jl_vel)
            data['joint_left']['effort'].append(jl_eff)

            data['joint_right']['position'].append(jr_pos)
            data['joint_right']['velocity'].append(jr_vel)
            data['joint_right']['effort'].append(jr_eff)

            data['end_pose_left'].append(pose_to_vec(frame['end_pose_left']))
            data['end_pose_right'].append(pose_to_vec(frame['end_pose_right']))

            data['arm_status_left'].append(status_to_vec(frame['arm_status_left']))
            data['arm_status_right'].append(status_to_vec(frame['arm_status_right']))

            print(f"Frame data: {count}/{self.args.max_timesteps}, current subtask: {SUBTASK_STEP}")
            if self.args.frame_rate > 0:
                time.sleep(1.0 / self.args.frame_rate)

        print('Collected frames:', len(data['action']))
        return data


def save_data(args, data, dataset_path):
    data_size = len(data['action'])
    subtask = SUBTASK_FLAG[:data_size, :]

    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 4) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        root.attrs['ros_version'] = 'ros2'

        root.create_dataset('language_raw', data=[args.language_raw])
        root.create_dataset('timestamps', data=np.asarray(data['timestamps'], dtype=np.float64))

        obs = root.create_group('observations')
        image = obs.create_group('images')
        image_depth = obs.create_group('images_depth')

        for cam_name in args.camera_names:
            rgb_np = np.asarray(data['images'][cam_name], dtype=np.uint8)
            depth_np = np.asarray(data['images_depth'][cam_name], dtype=np.uint16)
            image.create_dataset(cam_name, data=rgb_np, chunks=(1,) + rgb_np.shape[1:])
            image_depth.create_dataset(cam_name, data=depth_np, chunks=(1,) + depth_np.shape[1:])

        obs.create_dataset('qpos', data=np.asarray(data['qpos'], dtype=np.float32))
        obs.create_dataset('qvel', data=np.asarray(data['qvel'], dtype=np.float32))
        obs.create_dataset('effort', data=np.asarray(data['effort'], dtype=np.float32))

        root.create_dataset('action', data=np.asarray(data['action'], dtype=np.float32))
        root.create_dataset('base_action', data=np.zeros((data_size, 2), dtype=np.float32))
        root.create_dataset('subtask', data=subtask, dtype=np.uint8)

        arm_group = root.create_group('arm')

        for side_name in ['joint_states_left', 'joint_states_right', 'joint_left', 'joint_right']:
            g = arm_group.create_group(side_name)
            g.create_dataset('position', data=np.asarray(data[side_name]['position'], dtype=np.float32))
            g.create_dataset('velocity', data=np.asarray(data[side_name]['velocity'], dtype=np.float32))
            g.create_dataset('effort', data=np.asarray(data[side_name]['effort'], dtype=np.float32))

        arm_group.create_dataset('end_pose_left', data=np.asarray(data['end_pose_left'], dtype=np.float32))
        arm_group.create_dataset('end_pose_right', data=np.asarray(data['end_pose_right'], dtype=np.float32))
        arm_group.create_dataset('arm_status_left', data=np.asarray(data['arm_status_left'], dtype=np.int64))
        arm_group.create_dataset('arm_status_right', data=np.asarray(data['arm_status_right'], dtype=np.int64))
        arm_group.create_dataset('arm_status_fields', data=np.asarray(STATUS_FIELDS, dtype='S64'))

    print(f"Saving done: {dataset_path}.hdf5")


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--task_name', type=str, default='aloha_mobile_dummy')
    parser.add_argument('--episode_idx', type=int, default=0)
    parser.add_argument('--max_timesteps', type=int, default=500)
    parser.add_argument('--frame_rate', type=int, default=30)
    parser.add_argument('--language_raw', type=str, default='None')

    parser.add_argument('--camera_names', nargs=3, default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'])
    parser.add_argument('--depth_border_pad', type=int, default=0)

    parser.add_argument('--img_left_topic', type=str, default='/camera/left/color/image_raw')
    parser.add_argument('--img_right_topic', type=str, default='/camera/right/color/image_raw')
    parser.add_argument('--img_top_topic', type=str, default='/camera/top/color/image_raw')

    parser.add_argument('--img_left_depth_topic', type=str, default='/camera/left/depth/image_rect_raw')
    parser.add_argument('--img_right_depth_topic', type=str, default='/camera/right/depth/image_rect_raw')
    parser.add_argument('--img_top_depth_topic', type=str, default='/camera/top/depth/image_rect_raw')

    parser.add_argument('--joint_states_left_topic', type=str, default='/joint_states_left')
    parser.add_argument('--joint_states_right_topic', type=str, default='/joint_states_right')
    parser.add_argument('--joint_left_topic', type=str, default='/joint_left')
    parser.add_argument('--joint_right_topic', type=str, default='/joint_right')
    parser.add_argument('--end_pose_left_topic', type=str, default='/end_pose_left')
    parser.add_argument('--end_pose_right_topic', type=str, default='/end_pose_right')
    parser.add_argument('--arm_status_left_topic', type=str, default='/arm_status_left')
    parser.add_argument('--arm_status_right_topic', type=str, default='/arm_status_right')

    return parser.parse_args()


def main():
    listener_thread = threading.Thread(target=listen_for_keyboard, daemon=True)
    listener_thread.start()

    args = get_arguments()

    rclpy.init(args=None)
    collector = Ros2Collector(args)

    try:
        data = collector.process()
        dataset_dir = os.path.join(args.dataset_dir, args.task_name)
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_path = os.path.join(dataset_dir, f'episode_{args.episode_idx}')
        save_data(args, data, dataset_path)
    finally:
        collector.destroy_node()
        rclpy.shutdown()

    listener_thread.join(timeout=0.5)


if __name__ == '__main__':
    def signal_handler(sig, frame):
        print('Signal received, shutting down!')
        print('Remember to delete incomplete data if needed.')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    main()
