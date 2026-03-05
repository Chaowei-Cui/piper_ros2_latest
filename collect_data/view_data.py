#!/usr/bin/env python3
# coding=utf-8
"""Interactive HDF5 episode viewer for camera images and arm joint data."""
'''
  python3 /home/agilex/piper_ros/collect_data/view_data.py \
    --dataset_dir /home/agilex/data \
    --task_name bin_packing_ljm_all_up_random_screwdriver_pan_hand_cream_lipstick \
    --episode_idx 0 --show_depth --show_velocity --show_effort

'''
import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np


class EpisodeViewerData:
    def __init__(self, hdf5_path: str, show_depth: bool = False):
        self.hdf5_path = hdf5_path
        self.show_depth = show_depth
        self.root = h5py.File(hdf5_path, "r")

        self.compress_len = self.root["/compress_len"][()] if "/compress_len" in self.root else None

        self.rgb_cams = self._list_group_keys("/observations/images")
        self.depth_cams = self._list_group_keys("/observations/images_depth") if show_depth else []

        self.image_streams: List[Tuple[str, h5py.Dataset, bool, Optional[int]]] = []
        cam_idx = 0
        for cam_name in self.rgb_cams:
            ds = self.root[f"/observations/images/{cam_name}"]
            self.image_streams.append((cam_name, ds, self._is_compressed_dataset(ds), cam_idx))
            cam_idx += 1

        for cam_name in self.depth_cams:
            ds = self.root[f"/observations/images_depth/{cam_name}"]
            self.image_streams.append((f"depth/{cam_name}", ds, self._is_compressed_dataset(ds), cam_idx))
            cam_idx += 1

        self.joints = self._load_joint_data(self.root)
        self.num_frames = self._infer_num_frames()

    def _list_group_keys(self, path: str) -> List[str]:
        if path not in self.root:
            return []
        return list(self.root[path].keys())

    @staticmethod
    def _is_compressed_dataset(ds: h5py.Dataset) -> bool:
        return ds.ndim == 2 and ds.dtype == np.uint8

    def _infer_num_frames(self) -> int:
        candidates = []

        for _, ds, _, _ in self.image_streams:
            candidates.append(int(ds.shape[0]))

        for key in [
            "qpos_left",
            "qpos_right",
            "cmd_left",
            "cmd_right",
            "qvel_left",
            "qvel_right",
            "effort_left",
            "effort_right",
        ]:
            arr = self.joints.get(key)
            if arr is not None:
                candidates.append(int(arr.shape[0]))

        if not candidates:
            raise RuntimeError("No image or arm data found in this hdf5 file.")
        return min(candidates)

    @staticmethod
    def _safe_get(root: h5py.File, path: str) -> Optional[np.ndarray]:
        return root[path][()] if path in root else None

    @classmethod
    def _load_joint_data(cls, root: h5py.File) -> Dict[str, Optional[np.ndarray]]:
        required = [
            "/arm/joint_states_left/position",
            "/arm/joint_states_right/position",
        ]
        missing = [p for p in required if p not in root]
        if missing:
            raise RuntimeError(
                "Only new-format arm data is supported. Missing required datasets: "
                + ", ".join(missing)
            )

        out: Dict[str, Optional[np.ndarray]] = {
            "qpos_left": None,
            "qpos_right": None,
            "qvel_left": None,
            "qvel_right": None,
            "effort_left": None,
            "effort_right": None,
            "cmd_left": None,
            "cmd_right": None,
        }

        out["qpos_left"] = root["/arm/joint_states_left/position"][()]
        out["qpos_right"] = root["/arm/joint_states_right/position"][()]
        out["qvel_left"] = cls._safe_get(root, "/arm/joint_states_left/velocity")
        out["qvel_right"] = cls._safe_get(root, "/arm/joint_states_right/velocity")
        out["effort_left"] = cls._safe_get(root, "/arm/joint_states_left/effort")
        out["effort_right"] = cls._safe_get(root, "/arm/joint_states_right/effort")
        # In this dataset format, joint command equals joint state.
        out["cmd_left"] = out["qpos_left"]
        out["cmd_right"] = out["qpos_right"]

        return out

    @staticmethod
    def _ensure_bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            if img.dtype != np.uint8:
                img = EpisodeViewerData._normalize_to_u8(img)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if img.dtype != np.uint8:
            img = EpisodeViewerData._normalize_to_u8(img)
        return img

    @staticmethod
    def _normalize_to_u8(img: np.ndarray) -> np.ndarray:
        arr = img.astype(np.float32)
        if arr.size == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        vmin = float(np.percentile(arr, 1.0))
        vmax = float(np.percentile(arr, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1.0
        arr = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)

    def _decode_compressed(self, row: np.ndarray, cam_idx: Optional[int], frame_idx: int) -> np.ndarray:
        if self.compress_len is not None and cam_idx is not None:
            if cam_idx < self.compress_len.shape[0] and frame_idx < self.compress_len.shape[1]:
                n = int(self.compress_len[cam_idx, frame_idx])
                n = max(0, min(n, row.shape[0]))
                payload = row[:n]
            else:
                payload = row
        else:
            # Fallback: trim trailing zeros used as padding in compressed buffers.
            end = row.shape[0]
            while end > 0 and row[end - 1] == 0:
                end -= 1
            payload = row[:end] if end > 0 else row

        decoded = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise RuntimeError("Failed to decode compressed image frame.")
        return decoded

    def get_frame_images(self, frame_idx: int) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for stream_name, ds, is_compressed, cam_idx in self.image_streams:
            frame = ds[frame_idx]
            if is_compressed:
                frame = self._decode_compressed(frame, cam_idx, frame_idx)
            out[stream_name] = self._ensure_bgr(frame)
        return out

    def close(self):
        self.root.close()


def format_vec(vec: Optional[np.ndarray], decimals: int = 3) -> str:
    if vec is None:
        return "N/A"
    return np.array2string(np.asarray(vec), precision=decimals, separator=", ", suppress_small=False)


def resize_keep_ratio(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    target_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def make_mosaic(images: Dict[str, np.ndarray], target_h: int = 360) -> np.ndarray:
    if not images:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8)

    tiles = []
    for name, img in images.items():
        vis = resize_keep_ratio(img, target_h)
        cv2.putText(vis, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(vis)

    return cv2.hconcat(tiles)


def draw_joint_panel(
    h: int,
    frame_idx: int,
    total: int,
    joints: Dict[str, Optional[np.ndarray]],
    show_velocity: bool,
    show_effort: bool,
) -> np.ndarray:
    panel_w = 640
    panel = np.full((h, panel_w, 3), 20, dtype=np.uint8)
    fg = (230, 230, 230)
    em = (80, 220, 255)

    y = 36
    cv2.putText(panel, f"frame: {frame_idx}/{total - 1}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, em, 2, cv2.LINE_AA)
    y += 34

    ql = joints.get("qpos_left")
    qr = joints.get("qpos_right")
    cl = joints.get("cmd_left")
    cr = joints.get("cmd_right")
    vl = joints.get("qvel_left")
    vr = joints.get("qvel_right")
    el = joints.get("effort_left")
    er = joints.get("effort_right")

    left_q = ql[frame_idx] if ql is not None else None
    right_q = qr[frame_idx] if qr is not None else None
    left_c = cl[frame_idx] if cl is not None else None
    right_c = cr[frame_idx] if cr is not None else None
    left_v = vl[frame_idx] if vl is not None and frame_idx < vl.shape[0] else None
    right_v = vr[frame_idx] if vr is not None and frame_idx < vr.shape[0] else None
    left_e = el[frame_idx] if el is not None and frame_idx < el.shape[0] else None
    right_e = er[frame_idx] if er is not None and frame_idx < er.shape[0] else None

    lines = [
        "Left Arm",
        f"qpos: {format_vec(left_q)}",
        f"cmd : {format_vec(left_c)}",
    ]
    if show_velocity:
        lines.append(f"qvel: {format_vec(left_v)}")
    if show_effort:
        lines.append(f"eff : {format_vec(left_e)}")

    lines += [
        "",
        "Right Arm",
        f"qpos: {format_vec(right_q)}",
        f"cmd : {format_vec(right_c)}",
    ]
    if show_velocity:
        lines.append(f"qvel: {format_vec(right_v)}")
    if show_effort:
        lines.append(f"eff : {format_vec(right_e)}")

    lines += [
        "",
        "Controls:",
        "space: play/pause",
        "a/d: prev/next frame",
        "q or Esc: quit",
    ]

    for line in lines:
        if y > h - 16:
            break
        color = em if line in ["Left Arm", "Right Arm", "Controls:"] else fg
        cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 24

    return panel


def resolve_hdf5_path(args) -> str:
    if args.file:
        return os.path.abspath(os.path.expanduser(args.file))

    if not args.dataset_dir or not args.task_name:
        raise ValueError("Need --file, or --dataset_dir + --task_name + --episode_idx")

    path = os.path.join(
        os.path.abspath(os.path.expanduser(args.dataset_dir)),
        args.task_name,
        f"episode_{args.episode_idx}.hdf5",
    )
    return path


def main():
    parser = argparse.ArgumentParser(description="View episode images + arm joint data from hdf5.")
    parser.add_argument("--file", type=str, default=None, help="Direct path to *.hdf5")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset root, e.g. ~/data")
    parser.add_argument("--task_name", type=str, default=None, help="Task folder under dataset_dir")
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode index")
    parser.add_argument("--show_depth", action="store_true", help="Also show /observations/images_depth")
    parser.add_argument("--fps", type=float, default=20.0, help="Playback fps")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--show_velocity", action="store_true", help="Show qvel text")
    parser.add_argument("--show_effort", action="store_true", help="Show effort text")
    args = parser.parse_args()

    hdf5_path = resolve_hdf5_path(args)
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"File not found: {hdf5_path}")

    data = EpisodeViewerData(hdf5_path, show_depth=args.show_depth)
    print(f"Opened: {hdf5_path}")
    print(f"Frames: {data.num_frames}")
    print(f"RGB cameras: {data.rgb_cams}")
    if args.show_depth:
        print(f"Depth cameras: {data.depth_cams}")

    frame_idx = max(0, min(args.start, data.num_frames - 1))
    playing = True

    wait_ms = max(1, int(1000.0 / max(1e-3, args.fps)))

    win_name = "HDF5 Data Viewer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        images = data.get_frame_images(frame_idx)
        mosaic = make_mosaic(images, target_h=380)
        panel = draw_joint_panel(
            h=mosaic.shape[0],
            frame_idx=frame_idx,
            total=data.num_frames,
            joints=data.joints,
            show_velocity=args.show_velocity,
            show_effort=args.show_effort,
        )

        view = cv2.hconcat([mosaic, panel])
        cv2.imshow(win_name, view)

        key = cv2.waitKey(wait_ms if playing else 0) & 0xFF
        if key in [ord("q"), 27]:
            break
        if key == ord(" "):
            playing = not playing
        elif key in [ord("a"), 81]:
            frame_idx = max(0, frame_idx - 1)
            playing = False
        elif key in [ord("d"), 83]:
            frame_idx = min(data.num_frames - 1, frame_idx + 1)
            playing = False

        if playing:
            frame_idx += 1
            if frame_idx >= data.num_frames:
                frame_idx = data.num_frames - 1
                playing = False

    data.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
