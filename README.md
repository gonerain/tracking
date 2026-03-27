# ArUco Tracking

## 功能

该目录包含一个基于 OpenCV ArUco 的小型跟踪工具，支持：

- 从相机读取画面
- 检测 ArUco marker
- 计算 marker 在相机坐标系下的位置
- 将点从相机坐标系转换到目标坐标系
- 指定某个 marker id 进行持续跟踪
- 在目标短暂消失时利用历史位置做短时预测
- 保存时间戳、位置、框信息到 JSON 文件

## 文件说明

- `aruco_config.yaml`：相机、内参、显示、跟踪、日志配置
- `geometry.py`：坐标变换和投影函数
- `detect_aruco.py`：实时 ArUco 检测脚本
- `tracking.py`：指定 id 的实时跟踪与预测脚本
- `requirements.txt`：Python 依赖列表

## 安装依赖

```bash
pip install -r requirements.txt
```

当前依赖为：

- `numpy`
- `PyYAML`
- `opencv-contrib-python`

如果不需要 `cv2.imshow()` 显示窗口，也可以自行替换为 `opencv-contrib-python-headless`。

## 配置

主要配置文件为 `aruco_config.yaml`。

### 相机配置

```yaml
camera:
  source:
    device_id: 0
    width: 1280
    height: 720
    fps: 30
  intrinsics:
    camera_matrix:
      - [800.0, 0.0, 640.0]
      - [0.0, 800.0, 360.0]
      - [0.0, 0.0, 1.0]
    dist_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]
```

### 跟踪配置

```yaml
tracking:
  target_id: 0
  history_size: 10
  max_prediction_duration_s: 1.0
```

说明：

- `target_id`：要跟踪的 marker id
- `history_size`：保留多少个历史状态用于预测
- `max_prediction_duration_s`：目标消失后允许继续预测的最长时间

### 日志配置

```yaml
logging:
  output_json: tracking_log.json
  flush_every_frame: true
```

## 运行

检测脚本：

```bash
python3 detect_aruco.py --config aruco_config.yaml
```

跟踪脚本：

```bash
python3 tracking.py --config aruco_config.yaml
```

## 跟踪行为

`tracking.py` 只关注配置中的 `target_id`。

- 如果当前帧检测到该 id，则输出真实观测位置，状态为 `observed`
- 如果当前帧没检测到，但未超过 `max_prediction_duration_s`，则根据最近历史做匀速预测，状态为 `predicted`
- 如果超过最大预测时长，状态为 `lost`

当开启显示窗口时：

- 真实检测框使用检测结果绘制
- 预测状态会额外绘制预测框和预测中心点

## JSON 输出

输出文件默认是 `tracking_log.json`，每条记录包含：

- `timestamp`
- `datetime`
- `target_id`
- `visible`
- `tracking_state`
- `position_camera_m`
- `position_target_m`
- `corners_px`

## 注意事项

- 使用前需要根据实际相机修改 `camera_matrix` 和 `dist_coeffs`
- 如果 `display.enabled: true` 时遇到 OpenCV Qt 字体问题，可以先设为 `false`，只做跟踪和日志输出
- 当前预测模型是简单匀速外推，适合目标短时间遮挡，不适合长时间丢失
