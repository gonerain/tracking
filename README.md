# ArUco Tracking

基于 OpenCV ArUco 的检测与跟踪工具，支持相机输入和 ROS1 rosbag 输入。

## 功能

- 检测 ArUco marker
- 输出 marker 在相机坐标系下的位置
- 将点转换到目标坐标系
- 指定单个 marker id 持续跟踪
- 目标短暂消失时做短时匀速预测
- 将时间戳、位置和框信息保存为 JSON
- 保存带检测框和预测框的输出视频

## 文件

- `aruco_config.yaml`：配置文件
- `geometry.py`：坐标变换和投影函数
- `detect_aruco.py`：检测脚本
- `tracking.py`：跟踪脚本
- `requirements.txt`：Python 依赖

## 安装

```bash
pip install -r requirements.txt
```

`requirements.txt` 只包含当前工程的 Python 依赖。`rosbag` 不通过 pip 安装，使用 rosbag 输入时需要你本地已有 ROS1 Python 环境。

## 配置

### 输入源

```yaml
input:
  type: camera
  camera:
    device_id: 0
    width: 1280
    height: 720
    fps: 30
  rosbag:
    bag_path: demo.bag
    topic: /camera/image_raw
    start_time_s: 0.0
    end_time_s: null
    loop: false
```

说明：

- `input.type`：`camera` 或 `rosbag`
- `input.rosbag.topic`：图像话题名
- `input.rosbag.start_time_s` / `end_time_s`：按 bag 内时间截取
- `input.rosbag.loop`：读完后是否循环

当前 rosbag 支持：

- `sensor_msgs/Image`：`bgr8`、`rgb8`、`mono8`
- `sensor_msgs/CompressedImage`

### 相机内参

```yaml
camera:
  intrinsics:
    camera_matrix:
      - [800.0, 0.0, 640.0]
      - [0.0, 800.0, 360.0]
      - [0.0, 0.0, 1.0]
    dist_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]
```

### 目标坐标系

```yaml
target_frame:
  name: robot_base
  rotation_matrix:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  translation_m: [0.0, 0.0, 0.0]
```

### 跟踪

```yaml
tracking:
  target_id: 0
  history_size: 10
  max_prediction_duration_s: 1.0
```

### 视频保存

```yaml
video:
  save_enabled: false
  output_path: tracking_output.mp4
  fps: 30
  fourcc: mp4v
```

### JSON 日志

```yaml
logging:
  output_json: tracking_log.json
  flush_every_frame: true
```

## 运行

检测：

```bash
python3 detect_aruco.py --config aruco_config.yaml
```

跟踪：

```bash
python3 tracking.py --config aruco_config.yaml
```

如果要从 rosbag 读取，把配置改成：

```yaml
input:
  type: rosbag
```

## 跟踪状态

`tracking.py` 只跟踪 `tracking.target_id`。

- `observed`：当前帧检测到了目标
- `predicted`：当前帧没检测到，但还在允许预测时长内
- `lost`：超过最大预测时长，停止预测

显示或视频保存打开时：

- 真实检测框按检测结果绘制
- 预测状态会绘制预测框和预测中心点

## 输出 JSON

每条记录包含：

- `timestamp`
- `datetime`
- `target_id`
- `visible`
- `tracking_state`
- `position_camera_m`
- `position_target_m`
- `corners_px`

## 备注

- 运行前需要把相机内参改成你的真实标定值
- rosbag 模式下时间戳优先使用 bag 消息时间
- 如果 `cv2.imshow()` 的 Qt 后端有问题，可以把 `display.enabled` 设为 `false`
- 当前预测模型是简单匀速外推，只适合短时间遮挡
