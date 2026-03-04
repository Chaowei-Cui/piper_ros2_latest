# `collect_data_four_camera.py` 数据采集脚本逻辑详解

本文档用于帮助快速理解 [`collect_data/collect_data_four_camera.py`](/Users/zzz/Documents/piper_ros2_latest/collect_data/collect_data_four_camera.py) 的完整采集流程、时间同步机制、数据结构与常见注意事项。

## 1. 脚本目标与输出

该脚本在 ROS2（`rclpy`）环境下同步采集多路传感器与机械臂状态，最终写入单个 `HDF5` 数据文件：

- 输出路径：`{dataset_dir}/{task_name}/episode_{episode_idx}.hdf5`
- 典型数据内容：
1. 彩色图像（默认 3 路：`top/left/right`）
2. 可选深度图
3. 双臂关节状态（`qpos/qvel/effort`）
4. 末端位姿 `end_pose`
5. 双臂状态字 `arm_status`
6. 可选底盘速度 `base_action`
7. 动作 `action`（来自 `/joint_left` + `/joint_right`）
8. 子任务标注 `subtask`（键盘触发）

## 2. 全局变量与交互控制

脚本有几个关键全局变量：

- `saveData`：按下 `Enter` 后置 `True`，触发提前结束采集。
- `CUR_STEP / PRE_STEP`：当前帧索引与上一个打点帧索引。
- `CUR_TIME_STAMP / PRE_TIME_STAMP`：用于检测是否“快速双击”。
- `SUBTASK_FLAG`：形状 `(10000, 1)` 的数组，保存每帧的子任务标签。
- `SUBTASK_STEP`：记录子任务计数（用于打印）。

键盘监听线程：

- `Enter`：结束采集并进入保存。
- `Esc`：停止键盘监听（并不直接结束 ROS 主循环）。
- `Scroll Lock`：打子任务标记。
1. 单次触发：当前 `CUR_STEP` 置为 `10`
2. 两次触发间隔 < `0.4s`：当前步和上一步都置为 `100`（双击标志）

## 3. ROS 订阅与缓存结构

`RosOperator` 在初始化时创建多个 `deque(maxlen=10)` 缓存（`MAX_DEQUE=10`），每条消息按 `(timestamp, msg)` 形式入队。

### 3.1 订阅的话题（默认）

- 彩色图：
1. `--img_top_topic` = `/camera/top/color/image_raw`
2. `--img_left_topic` = `/camera/left/color/image_raw`
3. `--img_right_topic` = `/camera/right/color/image_raw`
- 深度图（可选）：
1. `--img_top_depth_topic` = `/camera/top/depth/image_rect_raw`
2. `--img_left_depth_topic` = `/camera/left/depth/image_rect_raw`
3. `--img_right_depth_topic` = `/camera/right/depth/image_rect_raw`
- 机械臂状态：
1. `--joint_states_left_topic`, `--joint_states_right_topic`
2. `--joint_left_topic`, `--joint_right_topic`
3. `--end_pose_left_topic`, `--end_pose_right_topic`
4. `--arm_status_left_topic`, `--arm_status_right_topic`
- 底盘（可选）：
1. `--robot_base_topic` = `/odom`

### 3.2 时间戳来源

- 大部分消息使用 `msg.header.stamp`（`sec + nanosec`）转换为秒。
- `Pose` 和 `PiperStatusMsg` 这两类在回调中用了 `use_now=True`，即使用 `node.get_clock().now()` 作为时间戳。

这意味着：末端位姿和臂状态并非“消息原始头时间”，而是“回调到达时间”。

### 3.3 QoS 策略

- 彩色图与深度图：`SensorDataQoS`（best effort，适配 ROS2 相机流）。
- 机械臂状态与底盘：`QoSProfile(depth=10)`（reliable）。

## 4. 时间对齐核心：`get_frame()`

`get_frame()` 是该脚本最核心的同步逻辑。

### 4.1 先判断“数据是否齐全”

- 所有必需队列都非空才继续。
- 若启用深度图/底盘，对应队列也必须非空。

### 4.2 选取对齐时刻

- 计算 `frame_time = min(各必需队列的最新时间戳)`。

这个策略的含义是：
- 以“最慢（最旧）的最新帧”为对齐目标，避免取到未来数据。

### 4.3 弹出对齐数据

每个队列都执行：

- 丢弃 `< frame_time` 的旧消息
- 弹出第一条 `>= frame_time` 的消息作为本帧数据

任一路失败（弹空）就返回 `False`，主循环会继续等待下一轮同步。

### 4.4 图像解码与深度补边

- 图像通过 `CvBridge.imgmsg_to_cv2(..., 'passthrough')` 转换。
- 深度图会执行 `cv2.copyMakeBorder(top=40, bottom=40, left=0, right=0, value=0)`。

## 5. 主采集循环：`process()`

循环条件：

- `count < max_timesteps + 1`
- 且未按 `Enter`（`saveData == False`）
- 且 `rclpy` 未关闭

注意这里是 `max_timesteps + 1`，因为它把第一帧作为 `FIRST`，后续帧才形成动作配对。

### 5.1 每轮步骤

1. 调 `get_frame()` 拿同步帧；失败则打印 `syn fail` 并继续。
2. `CUR_STEP = count`，用于键盘打点关联到当前帧。
3. 组织 `obs`（`OrderedDict`）：
   - `images`
   - 可选 `images_depth`
   - `qpos/qvel/effort`：左右臂拼接成 14 维
   - `end_pose`：左右末端 pose 拼接成 14 维
   - `arm_status`：左右各 19 个字段，拼成 38 维
   - `frame_timestamp`：`[frame_time]`
   - `base_vel`：启用底盘则写 `[linear.x, angular.z]`，否则 `[0.0, 0.0]`
4. 若是第一帧（`count == 1`）：
   - 仅创建 `dm_env.TimeStep(step_type=FIRST)` 加入 `timesteps`
   - 不生成 `action`
5. 非第一帧：
   - 创建 `TimeStep(step_type=MID)`
   - `action = concat(joint_left.position[:7], joint_right.position[:7])`
   - `actions.append(action)`，`timesteps.append(ts)`
6. `time.sleep(1.0/frame_rate)` 节流，频率由 `--frame_rate` 控制（默认 30Hz）。

### 5.2 关键对齐关系

结束后总是满足：

- `len(timesteps) = len(actions) + 1`

保存时用 `while actions:` 每次同时 `pop` 一个 `action` 与一个 `timestep`，因此第一帧 `FIRST` 会被自然对齐消费，不会越界。

## 6. 数据保存：`save_data()`

### 6.1 保存流程

1. 以 `len(actions)` 作为 `data_size`
2. 按步展开到 `data_dict`
3. 创建 HDF5 结构并一次性写入

### 6.2 HDF5 顶层属性与字段

顶层属性：

- `sim = False`
- `compress = False`

数据集结构（主要）：

- `/observations/images/{cam_name}`: `(T,480,640,3) uint8`
- `/observations/images_depth/{cam_name}`: `(T,480,640) uint16`（可选）
- `/observations/qpos`: `(T,14)`
- `/observations/qvel`: `(T,14)`
- `/observations/effort`: `(T,14)`
- `/observations/end_pose`: `(T,14)`
- `/observations/arm_status`: `(T,38) int64`
- `/observations/frame_timestamp`: `(T,1)`
- `/action`: `(T,14)`
- `/base_action`: `(T,2)`
- `/subtask`: `(T,1) uint8`
- `/language_raw`: 文本任务描述（由参数 `--language_raw` 提供）

其中 `T = len(actions)`。

## 7. 参数说明（高频）

- `--dataset_dir`：数据根目录，默认 `./data`
- `--task_name`：任务名子目录，默认 `aloha_mobile_dummy`
- `--episode_idx`：样本编号
- `--max_timesteps`：最大步数，默认 `500`
- `--frame_rate`：采集频率，默认 `30`
- `--use_depth_image`：是否采深度图
- `--use_robot_base`：是否采底盘速度
- `--language_raw`：语言指令文本

## 8. 运行时序（简版）

1. 注册 `SIGINT` 处理函数。
2. 启动键盘监听线程。
3. 解析参数。
4. 初始化 ROS2 节点并订阅话题。
5. 进入主循环同步采样，累计 `timesteps/actions`。
6. 结束采样后创建目录并写 `episode_xxx.hdf5`。
7. 关闭 executor、销毁节点、`rclpy.shutdown()`，最后等待线程退出。

## 9. 代码中的重要注意点

1. 脚本名是“四相机”，但当前 `process()` 实际写入的是 3 路彩色图（`top/left/right`），`front` 相关字段在此版本未启用（代码中有注释痕迹）。
2. `camera_names` 已改为 `nargs='+'`，命令行传多相机名时会正确解析为列表。
3. `Esc` 仅结束键盘监听，不等价于立刻终止采集主循环；主循环结束条件主要是 `Enter`、`max_timesteps`、或 `rclpy shutdown`。
4. 深度图固定做上下各 40 像素补零边，后处理训练时需要确认是否保留该预处理。
5. `SUBTASK_FLAG` 固定长度 10000，如果未来单段采集长度超过该值会越界，需要提前扩容或改成动态结构。

## 10. 最小使用示例

```bash
python collect_data/collect_data_four_camera.py \
  --dataset_dir ./data \
  --task_name pick_place \
  --episode_idx 0 \
  --max_timesteps 500 \
  --frame_rate 30 \
  --use_depth_image true \
  --use_robot_base false \
  --language_raw "pick up object and place it in box"
```

采集中：

- 按 `Scroll Lock` 可打子任务点。
- 按 `Enter` 提前结束并保存当前片段。
