# Tracking Tools

用于相机 ArUco、Livox 点云人体聚类跟踪，以及两者融合跟踪的工具集。

## 项目结构

- `main.py`: 统一 CLI 入口
- `core/fusion_tracking.py`: ArUco 先验与 lidar 候选融合
- `core/detect_aruco.py`: ArUco 检测与位姿估计
- `core/tracking.py`: 单目标 ArUco 跟踪
- `core/lidar_tracking.py`: 点云聚类、候选筛选、简化跟踪与调试输出
- `core/segmentation.py`: 前视图分割加载、推理与可视化
- `core/geometry.py`: 坐标和投影工具
- `tools/`: 兼容旧用法的脚本入口
- `configs/aruco_config.yaml`: 相机与 ArUco 配置
- `configs/lidar_config.yaml`: lidar、分割、调试与融合相关配置
- `docker/`: Docker 启停脚本与镜像定义

## 环境

### 本地 Python

```bash
pip install -r requirements.txt
```

说明：

- 读取 ROS1 `.bag` 依赖 `rosbag`
- `rosbag` 不能通过 `pip install -r requirements.txt` 直接获得
- 涉及 `.bag` 的功能建议在 Noetic 环境或容器中运行

### Docker

项目默认运行环境在 [`docker/start.sh`](/mnt/ning_602/work/tracking/docker/start.sh)。

```bash
bash docker/start.sh
```

脚本行为：

- 若 `rospytorch` 容器已存在，则直接启动并进入容器
- 若容器不存在，则以 `ebhrz/ros-pytorch:noetic_pt110_cu113` 创建新容器
- 将上一级工作区挂载到 `/root/HDMap`
- 仓库路径在容器内对应为 `/root/HDMap/tracking`

可选环境变量：

- `CONTAINER_NAME`: 覆盖默认容器名 `rospytorch`
- `IMAGE_NAME`: 覆盖默认镜像
- `MOUNT_TARGET`: 覆盖容器内挂载根目录，默认 `/root/HDMap`

## 运行方式

推荐使用统一入口：

```bash
python3 main.py detect_aruco --config configs/aruco_config.yaml
python3 main.py tracking --config configs/aruco_config.yaml
python3 main.py lidar_tracking --config configs/lidar_config.yaml
python3 main.py segment_image --config configs/lidar_config.yaml --image path/to/front.png --output outputs/segmentation_preview.png
python3 main.py fusion_tracking --aruco-config configs/aruco_config.yaml --lidar-config configs/lidar_config.yaml
```

兼容旧入口：

```bash
python3 tools/detect_aruco.py --config configs/aruco_config.yaml
python3 tools/tracking.py --config configs/aruco_config.yaml
python3 tools/lidar_tracking.py --config configs/lidar_config.yaml
python3 tools/segment_image.py --config configs/lidar_config.yaml --image path/to/front.png
```

## 功能说明

### `detect_aruco`

- 从相机或 rosbag 读取图像
- 检测 ArUco
- 估计 marker 在相机坐标系和目标坐标系下的位置

### `tracking`

- 针对指定 `target_id` 做连续跟踪
- 支持短时丢失后的常速度预测
- 可输出 JSON 和视频

### `lidar_tracking`

当前处理链路：

- ROI 裁剪
- 车体自遮挡剔除
- 基于高度阈值的简单地面去除
- 欧式聚类
- 人体候选尺寸和点数筛选
- 简化常速度跟踪
- 可选前视图分割引导

默认输出内容：

- JSON 日志，含原始点数、过滤后点数、候选簇和跟踪状态
- 调试图目录 `outputs/lidar_debug/`

常见调参项位于 `configs/lidar_config.yaml`：

- `roi`
- `ground_removal`
- `person_cluster`
- `tracker.gating_distance_m`
- `segmentation.*`
- `debug.*`

### `fusion_tracking`

融合逻辑：

- 相机侧先生成 ArUco 目标先验
- 将先验投影到 lidar 跟踪流程中
- 优先在目标分割区域和 ArUco 近邻候选中选取目标

输出默认写到：

- `outputs/fusion_tracking_log.json`
- `outputs/fusion_debug/`

## 输出目录

常见输出包括：

- `outputs/segmentation_preview.png`
- `outputs/lidar_tracking_log.json`
- `outputs/fusion_tracking_log.json`
- `outputs/lidar_debug/`
- `outputs/fusion_debug/`

## 备注

- 分割模型加载失败时，`lidar_tracking` 会回退到纯几何流程
- 如果 `fusion_tracking` 使用了相机同步和分割配置，会从 `lidar_config.yaml` 读取 `segmentation` 和 `sync`
- 运行前请确认配置里的 bag 路径、topic、相机内参和目标坐标变换与实际数据一致
