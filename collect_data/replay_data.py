# coding=utf-8
import argparse
import os
import time

import h5py
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from piper_msgs.msg import PiperStatusMsg, PosCmd
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState


def load_hdf5(dataset_dir, task_name, episode_idx):
    dataset_path = os.path.join(dataset_dir, task_name, f'episode_{episode_idx}.hdf5')
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f'Dataset does not exist: {dataset_path}')

    data = {}
    with h5py.File(dataset_path, 'r') as root:
        data['timestamps'] = root['/timestamps'][()] if '/timestamps' in root else None

        # RGB / Depth images
        data['images'] = {}
        data['images_depth'] = {}
        for cam_name in root['/observations/images'].keys():
            data['images'][cam_name] = root[f'/observations/images/{cam_name}'][()]
        if '/observations/images_depth' in root:
            for cam_name in root['/observations/images_depth'].keys():
                data['images_depth'][cam_name] = root[f'/observations/images_depth/{cam_name}'][()]

        # Arm datasets (new format)
        if '/arm/joint_states_left/position' in root:
            def _read_joint(prefix):
                return {
                    'position': root[f'/arm/{prefix}/position'][()],
                    'velocity': root[f'/arm/{prefix}/velocity'][()],
                    'effort': root[f'/arm/{prefix}/effort'][()],
                }

            data['joint_states_left'] = _read_joint('joint_states_left')
            data['joint_states_right'] = _read_joint('joint_states_right')
            data['joint_left'] = _read_joint('joint_left')
            data['joint_right'] = _read_joint('joint_right')

            data['end_pose_left'] = root['/arm/end_pose_left'][()]
            data['end_pose_right'] = root['/arm/end_pose_right'][()]
            data['arm_status_left'] = root['/arm/arm_status_left'][()]
            data['arm_status_right'] = root['/arm/arm_status_right'][()]
        else:
            # Fallback for older files
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()] if '/observations/qvel' in root else np.zeros_like(qpos)
            effort = root['/observations/effort'][()] if '/observations/effort' in root else np.zeros_like(qpos)
            action = root['/action'][()] if '/action' in root else qpos

            q_split = qpos.shape[1] // 2
            a_split = action.shape[1] // 2
            data['joint_states_left'] = {
                'position': qpos[:, :q_split], 'velocity': qvel[:, :q_split], 'effort': effort[:, :q_split]
            }
            data['joint_states_right'] = {
                'position': qpos[:, q_split:], 'velocity': qvel[:, q_split:], 'effort': effort[:, q_split:]
            }
            data['joint_left'] = {
                'position': action[:, :a_split], 'velocity': np.zeros_like(action[:, :a_split]), 'effort': np.zeros_like(action[:, :a_split])
            }
            data['joint_right'] = {
                'position': action[:, a_split:], 'velocity': np.zeros_like(action[:, a_split:]), 'effort': np.zeros_like(action[:, a_split:])
            }
            n = action.shape[0]
            data['end_pose_left'] = np.zeros((n, 7), dtype=np.float32)
            data['end_pose_right'] = np.zeros((n, 7), dtype=np.float32)
            data['arm_status_left'] = np.zeros((n, 19), dtype=np.int64)
            data['arm_status_right'] = np.zeros((n, 19), dtype=np.int64)

    return data, dataset_path


class Ros2Replayer(Node):
    def __init__(self, args):
        super().__init__('replay_node_ros2')
        self.args = args
        self.bridge = CvBridge()

        sensor_qos = qos_profile_sensor_data
        default_qos = QoSProfile(depth=10)

        # RGB
        self.pub_rgb_left = self.create_publisher(Image, args.img_left_topic, sensor_qos)
        self.pub_rgb_right = self.create_publisher(Image, args.img_right_topic, sensor_qos)
        self.pub_rgb_top = self.create_publisher(Image, args.img_top_topic, sensor_qos)

        # Depth
        self.pub_depth_left = self.create_publisher(Image, args.img_left_depth_topic, sensor_qos)
        self.pub_depth_right = self.create_publisher(Image, args.img_right_depth_topic, sensor_qos)
        self.pub_depth_top = self.create_publisher(Image, args.img_top_depth_topic, sensor_qos)

        # Arm topics
        self.pub_joint_states_left = self.create_publisher(JointState, args.joint_states_left_topic, default_qos)
        self.pub_joint_states_right = self.create_publisher(JointState, args.joint_states_right_topic, default_qos)
        self.pub_joint_left = self.create_publisher(JointState, args.joint_left_topic, default_qos)
        self.pub_joint_right = self.create_publisher(JointState, args.joint_right_topic, default_qos)
        self.pub_joint_ctrl_cmd_left = self.create_publisher(JointState, args.joint_ctrl_cmd_left_topic, default_qos)
        self.pub_joint_ctrl_cmd_right = self.create_publisher(JointState, args.joint_ctrl_cmd_right_topic, default_qos)

        self.pub_end_pose_left = self.create_publisher(Pose, args.end_pose_left_topic, default_qos)
        self.pub_end_pose_right = self.create_publisher(Pose, args.end_pose_right_topic, default_qos)
        self.pub_pos_cmd_left = self.create_publisher(PosCmd, args.pos_cmd_left_topic, default_qos)
        self.pub_pos_cmd_right = self.create_publisher(PosCmd, args.pos_cmd_right_topic, default_qos)

        self.pub_arm_status_left = self.create_publisher(PiperStatusMsg, args.arm_status_left_topic, default_qos)
        self.pub_arm_status_right = self.create_publisher(PiperStatusMsg, args.arm_status_right_topic, default_qos)

    @staticmethod
    def _align_vec_len(vec, target_len):
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape[0] == target_len:
            return arr
        if arr.shape[0] > target_len:
            return arr[:target_len]
        out = np.zeros((target_len,), dtype=np.float64)
        out[:arr.shape[0]] = arr
        return out

    def _build_joint_msg(self, position, velocity, effort):
        pos = np.asarray(position, dtype=np.float64)
        vel = self._align_vec_len(velocity, pos.shape[0])
        eff = self._align_vec_len(effort, pos.shape[0])
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'joint{i + 1}' for i in range(pos.shape[0])]
        msg.position = pos.tolist()
        msg.velocity = vel.tolist()
        msg.effort = eff.tolist()
        return msg

    @staticmethod
    def _build_pose_msg(vec7):
        msg = Pose()
        msg.position.x = float(vec7[0])
        msg.position.y = float(vec7[1])
        msg.position.z = float(vec7[2])
        msg.orientation.x = float(vec7[3])
        msg.orientation.y = float(vec7[4])
        msg.orientation.z = float(vec7[5])
        msg.orientation.w = float(vec7[6])
        return msg

    @staticmethod
    def _build_status_msg(vec19):
        msg = PiperStatusMsg()
        msg.ctrl_mode = int(vec19[0])
        msg.arm_status = int(vec19[1])
        msg.mode_feedback = int(vec19[2])
        msg.teach_status = int(vec19[3])
        msg.motion_status = int(vec19[4])
        msg.trajectory_num = int(vec19[5])
        msg.err_code = int(vec19[6])
        msg.joint_1_angle_limit = bool(vec19[7])
        msg.joint_2_angle_limit = bool(vec19[8])
        msg.joint_3_angle_limit = bool(vec19[9])
        msg.joint_4_angle_limit = bool(vec19[10])
        msg.joint_5_angle_limit = bool(vec19[11])
        msg.joint_6_angle_limit = bool(vec19[12])
        msg.communication_status_joint_1 = bool(vec19[13])
        msg.communication_status_joint_2 = bool(vec19[14])
        msg.communication_status_joint_3 = bool(vec19[15])
        msg.communication_status_joint_4 = bool(vec19[16])
        msg.communication_status_joint_5 = bool(vec19[17])
        msg.communication_status_joint_6 = bool(vec19[18])
        return msg

    @staticmethod
    def _quat_to_rpy(qx, qy, qz, qw):
        # Quaternion (x, y, z, w) -> Euler (roll, pitch, yaw), radians.
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = np.pi / 2.0 * np.sign(sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return float(roll), float(pitch), float(yaw)

    def _build_pos_cmd(self, pose_vec7, gripper, mode1, mode2):
        msg = PosCmd()
        msg.x = float(pose_vec7[0])
        msg.y = float(pose_vec7[1])
        msg.z = float(pose_vec7[2])
        roll, pitch, yaw = self._quat_to_rpy(
            float(pose_vec7[3]), float(pose_vec7[4]), float(pose_vec7[5]), float(pose_vec7[6])
        )
        msg.roll = roll
        msg.pitch = pitch
        msg.yaw = yaw
        msg.gripper = float(gripper)
        msg.mode1 = int(mode1)
        msg.mode2 = int(mode2)
        return msg

    def replay(self, data):
        cam_top, cam_left, cam_right = self.args.camera_names
        for cam_name in [cam_top, cam_left, cam_right]:
            if cam_name not in data['images']:
                raise RuntimeError(f'Missing RGB camera stream in hdf5: {cam_name}')

        total = data['joint_left']['position'].shape[0]
        print(f'Replay frames: {total}')

        prev_ts = None
        for i in range(total):
            if not rclpy.ok():
                break

            # Publish RGB
            self.pub_rgb_top.publish(self.bridge.cv2_to_imgmsg(data['images'][cam_top][i], encoding='passthrough'))
            self.pub_rgb_left.publish(self.bridge.cv2_to_imgmsg(data['images'][cam_left][i], encoding='passthrough'))
            self.pub_rgb_right.publish(self.bridge.cv2_to_imgmsg(data['images'][cam_right][i], encoding='passthrough'))

            # Publish Depth (if present)
            if all(cam in data['images_depth'] for cam in [cam_top, cam_left, cam_right]):
                self.pub_depth_top.publish(self.bridge.cv2_to_imgmsg(data['images_depth'][cam_top][i], encoding='passthrough'))
                self.pub_depth_left.publish(self.bridge.cv2_to_imgmsg(data['images_depth'][cam_left][i], encoding='passthrough'))
                self.pub_depth_right.publish(self.bridge.cv2_to_imgmsg(data['images_depth'][cam_right][i], encoding='passthrough'))

            # Publish joints
            self.pub_joint_states_left.publish(
                self._build_joint_msg(
                    data['joint_states_left']['position'][i],
                    data['joint_states_left']['velocity'][i],
                    data['joint_states_left']['effort'][i],
                )
            )
            self.pub_joint_states_right.publish(
                self._build_joint_msg(
                    data['joint_states_right']['position'][i],
                    data['joint_states_right']['velocity'][i],
                    data['joint_states_right']['effort'][i],
                )
            )
            self.pub_joint_left.publish(
                self._build_joint_msg(
                    data['joint_left']['position'][i],
                    data['joint_left']['velocity'][i],
                    data['joint_left']['effort'][i],
                )
            )
            self.pub_joint_right.publish(
                self._build_joint_msg(
                    data['joint_right']['position'][i],
                    data['joint_right']['velocity'][i],
                    data['joint_right']['effort'][i],
                )
            )
            # Replay control inputs for controller subscribers.
            self.pub_joint_ctrl_cmd_left.publish(
                self._build_joint_msg(
                    data['joint_left']['position'][i],
                    data['joint_left']['velocity'][i],
                    data['joint_left']['effort'][i],
                )
            )
            self.pub_joint_ctrl_cmd_right.publish(
                self._build_joint_msg(
                    data['joint_right']['position'][i],
                    data['joint_right']['velocity'][i],
                    data['joint_right']['effort'][i],
                )
            )

            # Publish end poses and arm status
            self.pub_end_pose_left.publish(self._build_pose_msg(data['end_pose_left'][i]))
            self.pub_end_pose_right.publish(self._build_pose_msg(data['end_pose_right'][i]))
            self.pub_arm_status_left.publish(self._build_status_msg(data['arm_status_left'][i]))
            self.pub_arm_status_right.publish(self._build_status_msg(data['arm_status_right'][i]))
            left_pos = data['joint_left']['position'][i]
            right_pos = data['joint_right']['position'][i]
            left_gripper = float(left_pos[6]) if len(left_pos) > 6 else (float(left_pos[-1]) if len(left_pos) > 0 else 0.0)
            right_gripper = float(right_pos[6]) if len(right_pos) > 6 else (float(right_pos[-1]) if len(right_pos) > 0 else 0.0)
            self.pub_pos_cmd_left.publish(
                self._build_pos_cmd(
                    data['end_pose_left'][i],
                    left_gripper,
                    self.args.pos_cmd_mode1,
                    self.args.pos_cmd_mode2,
                )
            )
            self.pub_pos_cmd_right.publish(
                self._build_pos_cmd(
                    data['end_pose_right'][i],
                    right_gripper,
                    self.args.pos_cmd_mode1,
                    self.args.pos_cmd_mode2,
                )
            )

            # Replay timing
            if self.args.use_saved_timestamps and data['timestamps'] is not None:
                cur_ts = float(data['timestamps'][i])
                if prev_ts is not None:
                    dt = max(0.0, min(cur_ts - prev_ts, self.args.max_sleep_s))
                    time.sleep(dt)
                prev_ts = cur_ts
            else:
                if self.args.frame_rate > 0:
                    time.sleep(1.0 / self.args.frame_rate)

            if i % 50 == 0:
                print(f'Replay {i}/{total}')


def main(args):
    data, path = load_hdf5(args.dataset_dir, args.task_name, args.episode_idx)
    print(f'Loaded: {path}')

    rclpy.init(args=None)
    node = Ros2Replayer(args)
    try:
        node.replay(data)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset root dir')
    parser.add_argument('--task_name', type=str, default='aloha_mobile_dummy')
    parser.add_argument('--episode_idx', type=int, default=0)

    parser.add_argument('--frame_rate', type=int, default=30)
    parser.add_argument('--use_saved_timestamps', action='store_true')
    parser.add_argument('--max_sleep_s', type=float, default=0.2)
    parser.add_argument('--camera_names', nargs=3, default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'])

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
    parser.add_argument('--joint_ctrl_cmd_left_topic', type=str, default='/joint_ctrl_cmd_left')
    parser.add_argument('--joint_ctrl_cmd_right_topic', type=str, default='/joint_ctrl_cmd_right')
    parser.add_argument('--end_pose_left_topic', type=str, default='/end_pose_left')
    parser.add_argument('--end_pose_right_topic', type=str, default='/end_pose_right')
    parser.add_argument('--pos_cmd_left_topic', type=str, default='/pos_cmd_left')
    parser.add_argument('--pos_cmd_right_topic', type=str, default='/pos_cmd_right')
    parser.add_argument('--pos_cmd_mode1', type=int, default=0)
    parser.add_argument('--pos_cmd_mode2', type=int, default=0)
    parser.add_argument('--arm_status_left_topic', type=str, default='/arm_status_left')
    parser.add_argument('--arm_status_right_topic', type=str, default='/arm_status_right')

    main(parser.parse_args())
