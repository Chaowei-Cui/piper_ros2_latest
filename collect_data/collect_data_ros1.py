#!/usr/bin/env python3
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
from colorama import Fore, Style
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from pynput import keyboard
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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
            print(
                f">>>>>Double steps occur at {PRE_STEP} && {CUR_STEP} | Time are {CUR_TIME_STAMP} | {PRE_TIME_STAMP} <<<<<<"
            )
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


def _pose_to_vec(msg):
    return np.array(
        [
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ],
        dtype=np.float32,
    )


def _fmt_vec(vec, precision=4):
    arr = np.asarray(vec, dtype=np.float64)
    return np.array2string(arr, precision=precision, separator=", ", suppress_small=False)


def _align_joint_vec(vec, target_dim):
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape[0] == target_dim:
        return arr
    if arr.shape[0] == 0:
        return np.zeros((target_dim,), dtype=np.float32)
    if arr.shape[0] > target_dim:
        return arr[:target_dim]
    out = np.zeros((target_dim,), dtype=np.float32)
    out[:arr.shape[0]] = arr
    return out


# 保存数据函数
def save_data(args, timesteps, actions, dataset_path):
    data_size = len(actions)
    global SUBTASK_FLAG

    if data_size == 0:
        raise RuntimeError("No action data collected; nothing to save.")

    first_obs = timesteps[0].observation
    js_left_pos_dim = int(np.asarray(first_obs["joint_states_left_pos"]).shape[0])
    js_left_vel_dim = int(np.asarray(first_obs["joint_states_left_vel"]).shape[0])
    js_left_eff_dim = int(np.asarray(first_obs["joint_states_left_eff"]).shape[0])
    js_right_pos_dim = int(np.asarray(first_obs["joint_states_right_pos"]).shape[0])
    js_right_vel_dim = int(np.asarray(first_obs["joint_states_right_vel"]).shape[0])
    js_right_eff_dim = int(np.asarray(first_obs["joint_states_right_eff"]).shape[0])

    data_dict = {
        "/base_action": [],
        "/subtask": SUBTASK_FLAG[:data_size, :],
        "/arm/joint_states_left/position": [],
        "/arm/joint_states_left/velocity": [],
        "/arm/joint_states_left/effort": [],
        "/arm/joint_states_right/position": [],
        "/arm/joint_states_right/velocity": [],
        "/arm/joint_states_right/effort": [],
        "/arm/end_pose_left": [],
        "/arm/end_pose_right": [],
    }

    for cam_name in args.camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []
        if args.use_depth_image:
            data_dict[f"/observations/images_depth/{cam_name}"] = []

    while actions:
        actions.pop(0)
        ts = timesteps.pop(0)

        data_dict["/base_action"].append(ts.observation["base_vel"])
        data_dict["/arm/joint_states_left/position"].append(ts.observation["joint_states_left_pos"])
        data_dict["/arm/joint_states_left/velocity"].append(ts.observation["joint_states_left_vel"])
        data_dict["/arm/joint_states_left/effort"].append(ts.observation["joint_states_left_eff"])
        data_dict["/arm/joint_states_right/position"].append(ts.observation["joint_states_right_pos"])
        data_dict["/arm/joint_states_right/velocity"].append(ts.observation["joint_states_right_vel"])
        data_dict["/arm/joint_states_right/effort"].append(ts.observation["joint_states_right_eff"])
        data_dict["/arm/end_pose_left"].append(ts.observation["end_pose_left"])
        data_dict["/arm/end_pose_right"].append(ts.observation["end_pose_right"])

        for cam_name in args.camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(ts.observation["images"][cam_name])
            if args.use_depth_image:
                data_dict[f"/observations/images_depth/{cam_name}"].append(ts.observation["images_depth"][cam_name])

    t0 = time.time()
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = False

        obs = root.create_group("observations")
        image = obs.create_group("images")
        if "language_raw" not in root.keys():
            print("languge_raw", args.language_raw)
            root.create_dataset("language_raw", data=[args.language_raw])

        for cam_name in args.camera_names:
            sample = np.asarray(data_dict[f"/observations/images/{cam_name}"][0])
            sample_shape = tuple(sample.shape)
            image.create_dataset(
                cam_name,
                (data_size, *sample_shape),
                dtype=sample.dtype,
                chunks=(1, *sample_shape),
            )

        if args.use_depth_image:
            image_depth = obs.create_group("images_depth")
            for cam_name in args.camera_names:
                sample = np.asarray(data_dict[f"/observations/images_depth/{cam_name}"][0])
                sample_shape = tuple(sample.shape)
                image_depth.create_dataset(
                    cam_name,
                    (data_size, *sample_shape),
                    dtype=sample.dtype,
                    chunks=(1, *sample_shape),
                )

        root.create_dataset("base_action", (data_size, 2))
        root.create_dataset("subtask", (data_size, 1), dtype="uint8")
        arm = root.create_group("arm")
        js_left = arm.create_group("joint_states_left")
        js_left.create_dataset("position", (data_size, js_left_pos_dim))
        js_left.create_dataset("velocity", (data_size, js_left_vel_dim))
        js_left.create_dataset("effort", (data_size, js_left_eff_dim))

        js_right = arm.create_group("joint_states_right")
        js_right.create_dataset("position", (data_size, js_right_pos_dim))
        js_right.create_dataset("velocity", (data_size, js_right_vel_dim))
        js_right.create_dataset("effort", (data_size, js_right_eff_dim))

        arm.create_dataset("end_pose_left", (data_size, 7), dtype="float32")
        arm.create_dataset("end_pose_right", (data_size, 7), dtype="float32")

        for name, array in data_dict.items():
            root[name][...] = np.asarray(array)

    print(f"\033[32m\nSaving: {time.time() - t0:.1f} secs. {dataset_path} \033[0m\n")


class RosOperator(Node):
    def __init__(self, args):
        super().__init__("record_episodes")
        self.args = args
        self.bridge = CvBridge()
        self._last_sync_warn_ts = 0.0

        self.img_left_deque = deque(maxlen=MAX_DEQUE)
        self.img_right_deque = deque(maxlen=MAX_DEQUE)
        self.img_top_deque = deque(maxlen=MAX_DEQUE)
        self.img_left_depth_deque = deque(maxlen=MAX_DEQUE)
        self.img_right_depth_deque = deque(maxlen=MAX_DEQUE)
        self.img_top_depth_deque = deque(maxlen=MAX_DEQUE)

        self.joint_states_left_deque = deque(maxlen=MAX_DEQUE)
        self.joint_states_right_deque = deque(maxlen=MAX_DEQUE)
        self.end_pose_left_deque = deque(maxlen=MAX_DEQUE)
        self.end_pose_right_deque = deque(maxlen=MAX_DEQUE)
        self.robot_base_deque = deque(maxlen=MAX_DEQUE)

        self.init_ros()

    def spin_input_callbacks(self):
        spin_calls = max(1, int(self.args.spin_once_per_loop))
        for idx in range(spin_calls):
            timeout = 0.001 if idx == 0 else 0.0
            rclpy.spin_once(self, timeout_sec=timeout)

    @staticmethod
    def _stamp_to_sec(msg):
        stamp = msg.header.stamp
        if hasattr(stamp, "to_sec"):
            return stamp.to_sec()
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _find_index_at_or_after_time(self, dq, frame_time):
        for idx, msg in enumerate(dq):
            if self._stamp_to_sec(msg) >= frame_time:
                return idx
        return None

    @staticmethod
    def _consume_index(dq, idx):
        for _ in range(idx):
            dq.popleft()
        # Keep the selected sample in deque so slow topics can be reused
        # on the next cycle instead of forcing a hard sync miss.
        return dq[0]

    def get_frame(self):
        missing = []
        if len(self.img_left_deque) == 0:
            missing.append(f"img_left({self.args.img_left_topic})")
        if len(self.img_right_deque) == 0:
            missing.append(f"img_right({self.args.img_right_topic})")
        if len(self.img_top_deque) == 0:
            missing.append(f"img_top({self.args.img_top_topic})")

        if len(self.joint_states_left_deque) == 0:
            missing.append(f"joint_states_left({self.args.joint_states_left_topic})")
        if len(self.joint_states_right_deque) == 0:
            missing.append(f"joint_states_right({self.args.joint_states_right_topic})")
        if len(self.end_pose_left_deque) == 0:
            missing.append(f"end_pose_left({self.args.end_pose_left_topic})")
        if len(self.end_pose_right_deque) == 0:
            missing.append(f"end_pose_right({self.args.end_pose_right_topic})")

        if self.args.use_depth_image:
            if len(self.img_left_depth_deque) == 0:
                missing.append(f"depth_left({self.args.img_left_depth_topic})")
            if len(self.img_right_depth_deque) == 0:
                missing.append(f"depth_right({self.args.img_right_depth_topic})")
            if len(self.img_top_depth_deque) == 0:
                missing.append(f"depth_top({self.args.img_top_depth_topic})")

        if missing:
            now = time.time()
            if now - self._last_sync_warn_ts > 1.0:
                self._last_sync_warn_ts = now
                print(Fore.YELLOW, "waiting topics:", ", ".join(missing))
            return False

        frame_candidates = [
            self._stamp_to_sec(self.img_left_deque[-1]),
            self._stamp_to_sec(self.img_right_deque[-1]),
            self._stamp_to_sec(self.img_top_deque[-1]),
            self._stamp_to_sec(self.joint_states_left_deque[-1]),
            self._stamp_to_sec(self.joint_states_right_deque[-1]),
        ]

        if self.args.use_depth_image:
            frame_candidates.extend(
                [
                    self._stamp_to_sec(self.img_left_depth_deque[-1]),
                    self._stamp_to_sec(self.img_right_depth_deque[-1]),
                    self._stamp_to_sec(self.img_top_depth_deque[-1]),
                ]
            )

        if self.args.use_robot_base and len(self.robot_base_deque) > 0:
            frame_candidates.append(self._stamp_to_sec(self.robot_base_deque[-1]))

        frame_time = min(frame_candidates)

        sync_streams = {
            "img_left": self.img_left_deque,
            "img_right": self.img_right_deque,
            "img_top": self.img_top_deque,
            "joint_states_left": self.joint_states_left_deque,
            "joint_states_right": self.joint_states_right_deque,
        }

        if self.args.use_depth_image:
            sync_streams.update(
                {
                    "depth_left": self.img_left_depth_deque,
                    "depth_right": self.img_right_depth_deque,
                    "depth_top": self.img_top_depth_deque,
                }
            )

        stream_indices = {}
        for name, dq in sync_streams.items():
            idx = self._find_index_at_or_after_time(dq, frame_time)
            if idx is None:
                return False
            stream_indices[name] = idx

        robot_base_idx = None
        if self.args.use_robot_base and len(self.robot_base_deque) > 0:
            robot_base_idx = self._find_index_at_or_after_time(self.robot_base_deque, frame_time)

        img_left_msg = self._consume_index(self.img_left_deque, stream_indices["img_left"])
        img_right_msg = self._consume_index(self.img_right_deque, stream_indices["img_right"])
        img_top_msg = self._consume_index(self.img_top_deque, stream_indices["img_top"])
        joint_states_left = self._consume_index(self.joint_states_left_deque, stream_indices["joint_states_left"])
        joint_states_right = self._consume_index(self.joint_states_right_deque, stream_indices["joint_states_right"])

        img_left = self.bridge.imgmsg_to_cv2(img_left_msg, "passthrough")
        img_right = self.bridge.imgmsg_to_cv2(img_right_msg, "passthrough")
        img_top = self.bridge.imgmsg_to_cv2(img_top_msg, "passthrough")

        img_left_depth = None
        img_right_depth = None
        img_top_depth = None
        if self.args.use_depth_image:
            img_left_depth_msg = self._consume_index(self.img_left_depth_deque, stream_indices["depth_left"])
            img_right_depth_msg = self._consume_index(self.img_right_depth_deque, stream_indices["depth_right"])
            img_top_depth_msg = self._consume_index(self.img_top_depth_deque, stream_indices["depth_top"])

            img_left_depth = self.bridge.imgmsg_to_cv2(img_left_depth_msg, "passthrough")
            img_right_depth = self.bridge.imgmsg_to_cv2(img_right_depth_msg, "passthrough")
            img_top_depth = self.bridge.imgmsg_to_cv2(img_top_depth_msg, "passthrough")

            top, bottom, left, right = 40, 40, 0, 0
            img_left_depth = cv2.copyMakeBorder(img_left_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            img_right_depth = cv2.copyMakeBorder(
                img_right_depth,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            img_top_depth = cv2.copyMakeBorder(img_top_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        robot_base = None
        if robot_base_idx is not None:
            robot_base = self._consume_index(self.robot_base_deque, robot_base_idx)
        end_pose_left = self.end_pose_left_deque[-1]
        end_pose_right = self.end_pose_right_deque[-1]

        return (
            img_left,
            img_right,
            img_left_depth,
            img_right_depth,
            img_top,
            img_top_depth,
            joint_states_left,
            joint_states_right,
            end_pose_left,
            end_pose_right,
            robot_base,
        )

    def img_left_callback(self, msg):
        self.img_left_deque.append(msg)

    def img_top_callback(self, msg):
        self.img_top_deque.append(msg)

    def img_top_depth_callback(self, msg):
        self.img_top_depth_deque.append(msg)

    def img_right_callback(self, msg):
        self.img_right_deque.append(msg)

    def img_left_depth_callback(self, msg):
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        self.img_right_depth_deque.append(msg)

    def joint_states_left_callback(self, msg):
        self.joint_states_left_deque.append(msg)

    def joint_states_right_callback(self, msg):
        self.joint_states_right_deque.append(msg)

    def end_pose_left_callback(self, msg):
        self.end_pose_left_deque.append(msg)

    def end_pose_right_callback(self, msg):
        self.end_pose_right_deque.append(msg)

    def robot_base_callback(self, msg):
        self.robot_base_deque.append(msg)

    def init_ros(self):
        qos = qos_profile_sensor_data

        self.create_subscription(Image, self.args.img_left_topic, self.img_left_callback, qos)
        self.create_subscription(Image, self.args.img_right_topic, self.img_right_callback, qos)
        self.create_subscription(Image, self.args.img_top_topic, self.img_top_callback, qos)

        if self.args.use_depth_image:
            self.create_subscription(Image, self.args.img_left_depth_topic, self.img_left_depth_callback, qos)
            self.create_subscription(Image, self.args.img_right_depth_topic, self.img_right_depth_callback, qos)
            self.create_subscription(Image, self.args.img_top_depth_topic, self.img_top_depth_callback, qos)

        self.create_subscription(JointState, self.args.joint_states_left_topic, self.joint_states_left_callback, qos)
        self.create_subscription(JointState, self.args.joint_states_right_topic, self.joint_states_right_callback, qos)
        self.create_subscription(Pose, self.args.end_pose_left_topic, self.end_pose_left_callback, qos)
        self.create_subscription(Pose, self.args.end_pose_right_topic, self.end_pose_right_callback, qos)

        if self.args.use_robot_base:
            self.create_subscription(Odometry, self.args.robot_base_topic, self.robot_base_callback, qos)

    def process(self):
        timesteps = []
        actions = []
        count = 0
        print_flag = True
        global CUR_STEP

        loop_dt = 1.0 / float(max(1, self.args.frame_rate))

        while ((count < self.args.max_timesteps + 1) and not saveData) and rclpy.ok():
            self.spin_input_callbacks()
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
                end_pose_left,
                end_pose_right,
                robot_base,
            ) = result

            image_dict = {
                self.args.camera_names[0]: img_top,
                self.args.camera_names[1]: img_left,
                self.args.camera_names[2]: img_right,
            }

            obs = collections.OrderedDict()
            obs["images"] = image_dict

            if self.args.use_depth_image:
                image_dict_depth = {
                    self.args.camera_names[0]: img_top_depth,
                    self.args.camera_names[1]: img_left_depth,
                    self.args.camera_names[2]: img_right_depth,
                }
                obs["images_depth"] = image_dict_depth

            print(Style.RESET_ALL)
            if np.array(joint_states_left.position).sum() < 0.001:
                print(
                    Fore.GREEN,
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>joint_states left action is zero>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
                )
                print(np.array(joint_states_left.position))

            if np.array(joint_states_right.position).sum() < 0.001:
                print(
                    Fore.GREEN,
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>joint_states right qpos is zero>>>>>>>>>>>>>>>>>>>>>>>>>>",
                )
                print(np.array(joint_states_right.position))

            obs["qpos"] = np.concatenate(
                (np.asarray(joint_states_left.position), np.asarray(joint_states_right.position)),
                axis=0,
            )
            obs["qvel"] = np.concatenate(
                (np.asarray(joint_states_left.velocity), np.asarray(joint_states_right.velocity)),
                axis=0,
            )
            obs["effort"] = np.concatenate(
                (np.asarray(joint_states_left.effort), np.asarray(joint_states_right.effort)),
                axis=0,
            )
            obs["joint_states_left_pos"] = np.asarray(joint_states_left.position, dtype=np.float32)
            obs["joint_states_left_vel"] = _align_joint_vec(joint_states_left.velocity, obs["joint_states_left_pos"].shape[0])
            obs["joint_states_left_eff"] = _align_joint_vec(joint_states_left.effort, obs["joint_states_left_pos"].shape[0])
            obs["joint_states_right_pos"] = np.asarray(joint_states_right.position, dtype=np.float32)
            obs["joint_states_right_vel"] = _align_joint_vec(joint_states_right.velocity, obs["joint_states_right_pos"].shape[0])
            obs["joint_states_right_eff"] = _align_joint_vec(joint_states_right.effort, obs["joint_states_right_pos"].shape[0])
            obs["end_pose_left"] = _pose_to_vec(end_pose_left)
            obs["end_pose_right"] = _pose_to_vec(end_pose_right)
            action = np.concatenate((obs["joint_states_left_pos"], obs["joint_states_right_pos"]), axis=0)

            if self.args.print_data_info and (count % self.args.print_every_n == 0):
                print(
                    f"[collect frame {count}] "
                    f"qpos={_fmt_vec(obs['qpos'])} "
                    f"action={_fmt_vec(action)} "
                    f"end_pose_left={_fmt_vec(obs['end_pose_left'])} "
                    f"end_pose_right={_fmt_vec(obs['end_pose_right'])}"
                )

            if self.args.use_robot_base and robot_base is not None:
                obs["base_vel"] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
                print("base", obs["base_vel"])
            else:
                obs["base_vel"] = [0.0, 0.0]

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

            actions.append(action)

            if len(actions) > 100:
                text_action = np.array(actions[-30:])
                print(
                    Fore.RED,
                    "arm state not change >>",
                    np.all(text_action == text_action[0]),
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
                )
                print(Fore.GREEN, "mean", np.sum(text_action - np.mean(text_action, axis=0)))
                print(Fore.RED, "left var", np.var(text_action[:, 0:7]))
                print(Fore.RED, "right var", np.var(text_action[:, 7:]))

            timesteps.append(ts)
            print(Fore.BLUE, f"Frame data: {count}, current subtask: {SUBTASK_STEP}")
            print(Style.RESET_ALL)

            if not rclpy.ok():
                exit(-1)

            time.sleep(loop_dt)

        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        return timesteps, actions


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", action="store", type=str, help="Dataset_dir.", default="./data", required=False)
    parser.add_argument("--task_name", action="store", type=str, help="Task name.", default="aloha_mobile_dummy", required=False)
    parser.add_argument("--episode_idx", action="store", type=int, help="Episode index.", default=0, required=False)

    parser.add_argument("--max_timesteps", action="store", type=int, help="Max_timesteps.", default=500, required=False)

    parser.add_argument(
        "--camera_names",
        nargs="+",
        help="camera_names",
        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        required=False,
    )

    parser.add_argument("--img_front_topic", action="store", type=str, help="img_front_topic", default="/camera/front/color/image_raw", required=False)
    parser.add_argument("--img_left_topic", action="store", type=str, help="img_left_topic", default="/camera/left/color/image_raw", required=False)
    parser.add_argument("--img_right_topic", action="store", type=str, help="img_right_topic", default="/camera/right/color/image_raw", required=False)
    parser.add_argument("--img_top_topic", action="store", type=str, help="img_top_topic", default="/camera/top/color/image_raw", required=False)

    parser.add_argument("--img_top_depth_topic", action="store", type=str, help="img_top_depth_topic", default="/camera/top/depth/image_rect_raw", required=False)
    parser.add_argument("--img_front_depth_topic", action="store", type=str, help="img_front_depth_topic", default="/camera/front/depth/image_rect_raw", required=False)
    parser.add_argument("--img_left_depth_topic", action="store", type=str, help="img_left_depth_topic", default="/camera/left/depth/image_rect_raw", required=False)
    parser.add_argument("--img_right_depth_topic", action="store", type=str, help="img_right_depth_topic", default="/camera/right/depth/image_rect_raw", required=False)

    parser.add_argument("--joint_states_left_topic", action="store", type=str, help="joint_states_left_topic", default="/joint_states_left", required=False)
    parser.add_argument("--joint_states_right_topic", action="store", type=str, help="joint_states_right_topic", default="/joint_states_right", required=False)
    parser.add_argument("--end_pose_left_topic", action="store", type=str, help="end_pose_left_topic", default="/end_pose_left", required=False)
    parser.add_argument("--end_pose_right_topic", action="store", type=str, help="end_pose_right_topic", default="/end_pose_right", required=False)

    parser.add_argument("--robot_base_topic", action="store", type=str, help="robot_base_topic", default="/odom", required=False)

    parser.add_argument("--use_robot_base", type=str2bool, help="use_robot_base", default=False, required=False)
    parser.add_argument("--use_depth_image", type=str2bool, help="use_depth_image", default=False, required=False)

    parser.add_argument("--frame_rate", action="store", type=int, help="frame_rate", default=30, required=False)
    parser.add_argument(
        "--spin_once_per_loop",
        action="store",
        type=int,
        help="How many spin_once calls to run per control loop to drain incoming topics.",
        default=20,
        required=False,
    )
    parser.add_argument("--language_raw", type=str, help="task instruction", default="None")
    parser.add_argument("--print_data_info", action="store_true", help="Print collected joint/end-pose data.")
    parser.add_argument("--print_every_n", action="store", type=int, help="Print data info every N frames.", default=1, required=False)

    return parser.parse_args(argv)


def main():
    listener_thread = threading.Thread(target=listen_for_keyboard)
    listener_thread.start()

    rclpy.init(args=None)
    args = get_arguments(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    args.print_every_n = max(1, int(args.print_every_n))
    ros_operator = RosOperator(args)
    print("topic config:")
    print("  img_left_topic:", args.img_left_topic)
    print("  img_right_topic:", args.img_right_topic)
    print("  img_top_topic:", args.img_top_topic)
    print("  joint_states_left_topic:", args.joint_states_left_topic)
    print("  joint_states_right_topic:", args.joint_states_right_topic)
    print("  end_pose_left_topic:", args.end_pose_left_topic)
    print("  end_pose_right_topic:", args.end_pose_right_topic)
    if args.use_depth_image:
        print("  img_left_depth_topic:", args.img_left_depth_topic)
        print("  img_right_depth_topic:", args.img_right_depth_topic)
        print("  img_top_depth_topic:", args.img_top_depth_topic)

    print("Waiting for initial messages...")
    for _ in range(50):
        ros_operator.spin_input_callbacks()

    timesteps, actions = ros_operator.process()
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    dataset_path = os.path.join(dataset_dir, "episode_" + str(args.episode_idx))
    save_data(args, timesteps, actions, dataset_path)

    ros_operator.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    listener_thread.join()


if __name__ == "__main__":

    def signal_handler(sig, frame):
        print("Signal received, shutting down!")
        print("remember to delete the uncompleted data!")
        print("\033[31m\nenter ESC to exit")
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    main()

# python collect_data_ros1.py --dataset_dir ~/data --max_timesteps 500 --episode_idx 1
