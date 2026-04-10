# Tracking Tools

## 目录

- `core/detect_aruco.py`：ArUco 检测核心逻辑
- `core/tracking.py`：ArUco 跟踪核心逻辑
- `core/lidar_tracking.py`：Livox 点云人体聚类与跟踪核心逻辑
- `core/segmentation.py`：前视图语义/实例/全景分割统一入口
- `core/geometry.py`：几何与坐标变换
- `tools/detect_aruco.py`：ArUco 检测入口
- `tools/tracking.py`：ArUco 跟踪入口
- `tools/lidar_tracking.py`：雷达人体聚类入口
- `tools/segment_image.py`：单张图片分割预览入口
- `configs/aruco_config.yaml`：相机 ArUco 配置
- `configs/lidar_config.yaml`：雷达聚类与跟踪配置
- `outputs/`：默认日志和视频输出目录

## 安装

```bash
pip install -r requirements.txt
```

说明：

- ROS1 `.bag` 读取依赖 `rosbag`
- `rosbag` 不通过 `pip` 安装
- 读取 `data/geo_scan2/*.bag` 时，需要 ROS1 Python 环境或 Noetic 容器

## 运行

推荐从仓库根目录运行：

```bash
python3 tools/detect_aruco.py
python3 tools/tracking.py
python3 tools/lidar_tracking.py
python3 tools/segment_image.py --image path/to/front.png
```

或者直接用仓库根目录统一入口：

```bash
python3 main.py detect_aruco
python3 main.py tracking
python3 main.py lidar_tracking
python3 main.py segment_image --image path/to/front.png
python3 main.py fusion_tracking
```

融合入口会读取相机 ArUco 与 Livox 点云，两者按时间戳顺序推进，用 ArUco 在 lidar 坐标系下的目标位置给激光聚类提供先验：

```bash
python3 main.py fusion_tracking --aruco-config configs/aruco_config.yaml --lidar-config configs/lidar_config.yaml
```

显式指定配置：

```bash
python3 tools/detect_aruco.py --config configs/aruco_config.yaml
python3 tools/tracking.py --config configs/aruco_config.yaml
python3 tools/lidar_tracking.py --config configs/lidar_config.yaml
python3 tools/segment_image.py --config configs/lidar_config.yaml --image path/to/front.png --output outputs/segmentation_preview.png
```

## Docker

启动 Noetic 容器：

```bash
bash docker/start.sh
```

脚本会进入 `rospytorch` 容器，并把仓库挂载到：

```bash
/root/HDMap/tracking
```

我已经在这个容器里验证过：

- `rosbag` 可导入
- `tools/lidar_tracking.py` 可以读取 `data/geo_scan2/data_0.bag`

## 当前雷达脚本

`core/lidar_tracking.py` 当前流程：

- ROI 裁剪
- 车体自遮挡剔除
- 简单地面去除
- 欧式聚类
- 基于尺寸和点数的人体候选筛选
- 简化常速度跟踪

输出 JSON 会包含：

- 原始点数和过滤后点数
- 候选簇列表
- `centroid_lidar_m`
- 更稳的 `footpoint_lidar_m`
- `track.position_lidar_m`

另外默认会输出雷达 debug 图到 `outputs/lidar_debug/`：

- `raw_bev/`：原始点云俯视图
- `filtered_bev/`：过滤后点云俯视图，带候选簇和跟踪点
- `side_view/`：`x-z` 侧视图，方便看高度和地面去除
- `target_crop/`：围绕当前跟踪目标的局部俯视图
- `segmentation/`：前视分割预览
- `overview/`：front 图、分割图和 lidar 视图拼在一起的总览图

图例：

- 灰色点：点云
- 橙色框：候选簇包围盒
- 蓝点：簇中心
- 红点：脚点估计
- 绿色或黄色圆点：最终跟踪位置

如果图片太多，可以调 `configs/lidar_config.yaml` 里的 `debug.every_n_frames`。

当前结果已经能在部分时间段稳定抓到人形簇，但仍有误检。下一步主要靠调 `configs/lidar_config.yaml` 的：

- `roi`
- `ground_removal`
- `person_cluster`
- `tracker.gating_distance_m`
