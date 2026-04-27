# `target_world_positions.jsonl` 字段说明

本文档对应 `core/fusion_tracking.py` 生成的逐帧 JSONL 输出（`--output-target-world-json`）。

## 1. 文件格式

- 文件是 JSON Lines：每行一个 JSON 对象（1 帧）。
- 常见路径：
  - 单包：`outputs/fusion_batch/bags/<bag_stem>/target_world_positions.jsonl`
  - 合并：`outputs/fusion_batch/merged/target_world_positions.jsonl`

## 2. 顶层字段

每行记录包含以下顶层字段：

- `timestamp` (`float`)  
  当前帧时间戳（Unix 秒）。

- `datetime` (`string`)  
  `timestamp` 对应的 ISO8601 时间字符串（含时区）。

- `target_name` (`list`)  
  目标定义。通常是 ArUco 目标组 ID 列表，例如 `[[3,17,42,88],[2,19,55,81]]`。

- `target_lidar_m` (`[x, y, z]` 或 `null`)  
  单目标兼容字段（单位米，LiDAR 坐标系）。

- `target_world_lla` (`[lat, lon, alt]` 或 `null`)  
  单目标兼容字段（WGS84：纬度/经度/高程）。

- `target_world_enu_m` (`[e, n, u]` 或 `null`)  
  单目标兼容字段（相对配置的 ENU 原点，单位米）。

- `multi_targets` (`list`)  
  多目标跟踪结果列表（来自 lidar 多目标 tracker）。

- `aruco_group_targets` (`list`)  
  ArUco 目标组解算结果列表（每个目标组一项）。

- `ie_pose` (`object` 或 `null`)  
  当前帧插值后的 IE/GNSS/姿态解，用于把 LiDAR 坐标转换到世界坐标。

- `source_bag` (`string`, 可选)  
  仅在批处理 merge 后自动补充，表示该行来自哪个 bag 文件。

## 3. `multi_targets` 子字段

`multi_targets` 每个元素包含：

- `track_id` (`int`)：轨迹 ID  
- `state` (`string`)：当前观测状态（如 `observed` 等）  
- `lifecycle` (`string`)：轨迹生命周期（如 `confirmed`）  
- `quality` (`float`)：轨迹质量分数  
- `hits` (`int`)：命中次数  
- `misses` (`int`)：丢失次数  
- `target_lidar_m` (`[x, y, z]`)：LiDAR 坐标（m）  
- `target_world_lla` (`[lat, lon, alt]`)：世界坐标 LLA  
- `target_world_enu_m` (`[e, n, u]`)：世界坐标 ENU（m）

## 4. `aruco_group_targets` 子字段

`aruco_group_targets` 每个元素包含：

- `target_group_index` (`int`)：目标组索引  
- `target_ids` (`list[int]`)：该组定义的 marker IDs  
- `visible_ids` (`list[int]`)：该帧实际可见 marker IDs  
- `inferred_ids` (`list[int]`)：该帧通过偏移推断补全的 IDs  
- `offsets_applied` (`bool`)：是否应用了组内 offset 推断  
- `target_lidar_m` (`[x, y, z]`)：组中心 LiDAR 坐标（m）  
- `target_world_lla` (`[lat, lon, alt]`)：组中心世界坐标 LLA  
- `target_world_enu_m` (`[e, n, u]`)：组中心世界坐标 ENU（m）

## 5. 单目标兼容字段的取值优先级

`target_lidar_m` / `target_world_lla` / `target_world_enu_m` 的来源：

1. 若 `multi_targets` 非空：取 `multi_targets[0]`。  
2. 否则若 `aruco_group_targets` 非空：取 `aruco_group_targets[0]`。  
3. 两者都空则为 `null`。

因此这 3 个字段是“向后兼容镜像字段”，新逻辑建议优先读取 `multi_targets` 和 `aruco_group_targets`。

## 6. 字段为空的常见原因

- `ie_pose == null`：该帧没有可用的 IE 插值姿态。  
- `target_world_* == null`：没有可用目标，或缺少 IE/world 变换条件。  
- `multi_targets == []`：该帧 tracker 没有有效跟踪目标。  
- `aruco_group_targets == []`：该帧没有可用 ArUco 组解算结果。

