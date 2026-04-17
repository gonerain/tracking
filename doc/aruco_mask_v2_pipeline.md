# aruco_mask_v2 模式 Pipeline（当前实现）

本文档基于当前代码实现梳理 `detector.source: aruco_mask_v2` 的完整执行链路，覆盖输入、预处理、候选生成、跟踪、世界坐标输出与调参点。

- 主入口：`core/fusion_tracking.py:1554`
- 模式开关：`configs/lidar_config.yaml:87`（`source: aruco_mask_v2`）

---

## 1. 运行入口与总体流程

每帧主循环位于 `core/fusion_tracking.py:1631` 附近，核心顺序如下：

1. 读取当前帧（或时序融合后的帧）点云
2. 点云预处理：ROI -> 去自车 -> 去地面 -> ArUco 局部 ROI
3. 生成几何簇（全局）
4. 生成 `target mask` 投影点簇（分割引导）
5. 当模式是 `aruco_mask_v2` 时，仅使用 mask 投影簇生成 target 候选
6. 结合 ArUco prior 做候选选择与运动一致性约束
7. 更新跟踪器（必要时回退到 `aruco_fallback`）
8. 输出融合日志和世界轨迹 JSONL

---

## 2. 数据输入层

### 2.1 点云输入

- `RosbagLidarSource` 读取点云帧
- 可选时序融合（`temporal_fusion`）在主循环前完成缓存，逐帧调用：
  - `temporal_fuse_points(...)`（`core/fusion_tracking.py:406`）

### 2.2 ArUco + 分割输入

- `ArucoPriorProvider` 提供：
  - `aruco_prior`（目标在 LiDAR 坐标下的位置先验）
  - `aruco_debug.target_person_mask`（目标关联掩码）
- 在每帧通过：
  - `aruco_provider.get_prior(timestamp)`
  - `aruco_provider.get_debug_state()`

---

## 3. 点云预处理层

### 3.1 ROI 裁剪

- `crop_points(...)`
- 位置：`core/lidar_tracking.py`（在 fusion 中调用）

### 3.2 去自车

- `remove_ego_vehicle_points(...)`

### 3.3 去地面

- `remove_ground(...)`：`core/lidar_tracking.py:540`
- 支持 `z_threshold / adaptive_grid / adaptive_plane`
- 支持保护区 `protect_regions`（例如 tracker/aruco 附近不删）
- 当前 fusion 中会动态构造保护区：
  - `build_ground_protect_regions(...)`（`core/fusion_tracking.py`）

### 3.4 ArUco 局部 ROI（可恢复逻辑）

- `apply_aruco_local_roi(...)`：`core/fusion_tracking.py:1427`
- 目的：在 ArUco 已知附近裁剪点云，减少干扰
- 有点数下限保护，防止过裁剪后点太少直接失效

---

## 4. 候选生成层（v2核心）

### 4.1 全局几何簇（仍会计算）

- `euclidean_clusters(...)`：`core/lidar_tracking.py:672`
- `merge_vertical_person_clusters(...)`：`core/lidar_tracking.py:704`
- `filter_candidates(...)`：`core/lidar_tracking.py:802`
- 得到 `geometric_candidates`

说明：这些结果在 `aruco_mask_v2` 模式下不作为主候选来源，但仍用于日志/诊断与兼容逻辑。

### 4.2 目标掩码投影点（v2核心输入）

- `select_points_in_target_mask(...)`：`core/fusion_tracking.py:1390`
- 流程：
  1. 将每个 LiDAR 点投到相机像素平面
  2. 判断像素是否落在 `aruco_debug.target_person_mask`
  3. 保留命中点作为 `segmented_points`

### 4.3 对 mask 点做聚类

- `euclidean_clusters(segmented_points, ...)`
- `merge_vertical_person_clusters(...)`
- `filter_candidates(...)`

### 4.4 aruco_mask_v2 的候选构建

- `build_target_mask_v2_candidates(...)`：`core/fusion_tracking.py:1230`
- 逻辑：
  1. 对 mask 簇使用更宽松的人体约束（优先 `_with_aruco` 参数）
  2. 若筛选结果为空，则取最大簇兜底（点数达标）

### 4.5 模式切换分支

- 位置：`core/fusion_tracking.py:1708`
- 当 `detector.source == "aruco_mask_v2"`：
  - `candidates = build_target_mask_v2_candidates(segmented_clusters, person_cfg)`
  - `candidate_source = "aruco_mask_v2"`
  - `prioritized_candidates = candidates`

即：**不再走 cluster/centerpoint/hybrid 融合作为主路径**。

---

## 5. 候选选择与跟踪层

### 5.1 ArUco 先验选择

- `select_candidate_with_aruco_prior(...)`：`core/fusion_tracking.py:1114`
- 依据：
  - 与 ArUco prior 的平面距离（高权重）
  - 与 tracker 预测位置一致性（中权重）
  - 候选本身几何分数（低权重）

### 5.2 运动一致性过滤

- `candidate_motion_reject_reason(...)`
- 约束：
  - `footpoint_z`
  - 平面跳变、竖直跳变
  - 平面/竖直速度上限
- 可在 ArUco 可见时放宽（`aruco_observed_relax_motion`）

### 5.3 Tracker 更新

- `tracker.update(...)`：`core/fusion_tracking.py:1795`
- 优先输入：
  - 若有 `aruco_selected_candidate`，只送该一个候选
  - 否则送通过 motion 过滤的候选列表

### 5.4 失配回退

- 若 tracker 失配且有 prior，可回退：
  - `source = "aruco_fallback"`
- 对应字段：`used_aruco_fallback`

---

## 6. 世界坐标输出层

### 6.1 输出文件

- `outputs/fusion_tracking_log.json`
- `outputs/target_world_positions.jsonl`

### 6.2 world 结果生成

- 在每帧末尾构造 `world_record`：`core/fusion_tracking.py:1899`
- 关键字段：
  - `target_lidar_m`
  - `ie_pose`（SPAN 轨迹姿态）
  - `target_world_lla`
  - `target_world_enu_m`

---

## 7. 与旧模式（hybrid/cluster）的核心区别

`aruco_mask_v2` 的本质差异：

1. 目标候选来源改为“投到目标掩码的点云簇”
2. 候选优先由 ArUco+mask 决定，不依赖全局 cluster/centerpoint 主导
3. 在 mask 候选不足时有“最大簇兜底”，提高连续性

适合场景：

- 已有比较稳定的 ArUco 目标身份与相机视角
- 希望显著降低旁人/背景被误跟踪

---

## 8. 关键配置（建议重点看）

文件：`configs/lidar_config.yaml`

1. 模式开关
- `detector.source: aruco_mask_v2`

2. 地面滤波
- `ground_removal.*`（影响 target 点保留）
- `protect_*`（保护目标附近点不被删）

3. 聚类
- `clustering.tolerance_m`
- `clustering.min_cluster_points`
- `clustering.merge_xy_distance_m`
- `clustering.merge_z_gap_m`

4. 人体候选阈值
- `person_cluster.height_m/width_m/depth_m`
- `*_with_aruco`（v2优先使用）

5. 跟踪约束
- `tracker.max_footpoint_z_m`
- `tracker.max_planar_jump_m`
- `tracker.max_planar_speed_mps`
- `tracker.aruco_observed_relax_motion`

---

## 9. 常见问题定位（v2）

1. `candidate_source` 不是 `aruco_mask_v2`
- 检查 `configs/lidar_config.yaml` 的 `detector.source`
- 检查实际运行命令是否加载了这份配置

2. `point_count_in_target_mask` 很低
- 相机-雷达外参或投影模型偏差
- `target_person_mask` 本身过小/错位
- ROI/ground removal 过强导致有效点被提前删掉

3. 经常 `aruco_fallback=true`
- mask 候选点不足或被 motion 约束拒绝
- 可先放宽 `_with_aruco` 尺寸阈值、`max_footpoint_z_with_aruco_m`、速度门限

4. target 与 span 相对关系异常
- 首查外参方向（`ie_lidar_extrinsics`）
- 次查 heading/yaw 约定
- 再查时间同步

---

## 10. 最小验收清单

跑完后建议检查：

1. `fusion_tracking_log.json` 中 `candidate_source` 大部分帧为 `aruco_mask_v2`
2. `point_count_in_target_mask` 非零帧占比足够高
3. `used_aruco_fallback` 比例较旧版下降或至少不升高
4. 导出的 target 轨迹抖动和跳变减少

---

## 11. 坐标转换总表（公式 + 矩阵 + 代码实现）

本节把当前 `fusion_tracking` 中所有关键坐标转换集中说明。

### 11.1 使用的坐标系

1. `L`：LiDAR 车体系（按文档与 ROS 约定，FLU）
2. `B`：`base_link`（FLU）
3. `S`：SPAN/IE body（FRD）
4. `ENU`：局部东-北-天
5. `ECEF`：地心地固
6. `LLA`：经纬高（WGS84）

### 11.2 外参配置来源

配置位置：`configs/lidar_config.yaml` 的 `ie_lidar_extrinsics`。

当前配置矩阵：

- `T_{B<-L}` (`base_from_lidar`)
```text
R_B<-L =
[[ 0.9063, 0.0, -0.4226],
 [ 0.0,    1.0,  0.0   ],
 [ 0.4226, 0.0,  0.9063]]

t_B<-L = [-0.0315, 0.0, -0.1314]^T  (m)
```

- `T_{B<-S}` (`base_to_span`)
```text
R_B<-S =
[[ 1.0,  0.0,  0.0],
 [ 0.0, -1.0,  0.0],
 [ 0.0,  0.0, -1.0]]

t_B<-S = [-0.2684, -0.0820, -0.1527]^T  (m)
```

### 11.3 外参连乘与求逆

代码：`configure_span_lidar_extrinsics(...)`。

```text
R_S<-B = (R_B<-S)^T
t_S<-B = -R_S<-B * t_B<-S
R_S<-L = R_S<-B * R_B<-L
t_S<-L = R_S<-B * t_B<-L + t_S<-B
R_L<-S = (R_S<-L)^T
t_L<-S = -R_L<-S * t_S<-L
```

这组结果在运行时存到：
- `R_SPAN_FROM_LIDAR_ACTIVE`, `T_SPAN_FROM_LIDAR_M_ACTIVE`
- `R_LIDAR_FROM_SPAN_ACTIVE`, `T_LIDAR_FROM_SPAN_M_ACTIVE`

### 11.4 LiDAR <-> SPAN 点变换

代码函数：
- `lidar_to_ie_body(point_lidar)`
- `ie_body_to_lidar(point_body)`

公式：

```text
p_S = R_S<-L * p_L + t_S<-L
p_L = R_L<-S * p_S + t_L<-S
```

说明：外参链在 FLU 计算，进入 IE/SPAN body 前执行 FLU->FRD 轴变换。

### 11.5 IE 姿态到 ENU 旋转

代码函数：`rotation_matrix_from_ie(roll_deg, pitch_deg, heading_deg)`。

当前假设：
- `heading` 是“从北顺时针”
- 转 ENU yaw 用：`yaw_enu_deg = 90 - heading_deg`
- 欧拉顺序：`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`
```text
R_ENU<-S = Rz(yaw) * Ry(pitch) * Rx(roll)
```

该假设若不成立，会直接导致“SPAN 与 target 相对方位系统偏移”。

### 11.6 LLA <-> ECEF <-> ENU

代码函数：
- `_geodetic_to_ecef`
- `_ecef_to_geodetic`
- `_ecef_to_enu_matrix`
- `_enu_to_ecef_matrix`

核心关系：

```text
p_ENU  = R_ENU<-ECEF * (p_ECEF - p_ECEF_origin)
p_ECEF = p_ECEF_origin + R_ECEF<-ENU * p_ENU
```

其中 `origin` 取第一帧 IE pose。

### 11.7 时序融合中的跨帧点云变换

代码函数：`transform_points_lidar_between_ie_poses(...)`。

给定源帧 `k` 和目标帧 `j`：

1. `p_L^k -> p_S^k`（外参）
2. `p_S^k -> p_ENU`（用 `R_{ENU<-S}^k` 与位姿平移）
3. `p_ENU -> p_S^j`（减目标位姿，再乘 `R_{ENU<-S}^j` 的逆）
4. `p_S^j -> p_L^j`（外参逆）

这保证时序融合与 world 输出使用一致外参。

### 11.8 target 世界坐标输出变换

代码位置：`core/fusion_tracking.py` `world_record` 构造段。

1. tracker 给出 `target_lidar`（在 `L`）
2. 先变换到 `S`：`target_body = lidar_to_ie_body(target_lidar)`
3. 用当前 IE pose 姿态平移到 ENU/ECEF，再转 LLA

对应：

```text
p_ENU_target  = R_ENU<-S * p_S + p_ENU_span
p_ECEF_target = p_ECEF_span + R_ECEF<-ENU * p_ENU_target_offset
```

最终写入：
- `target_world_lla`
- `target_world_enu_m`

### 11.9 LiDAR 点投影到相机掩码（aruco_mask_v2核心）

代码函数：
- `lidar_point_to_camera(...)`
- `select_points_in_target_mask(...)`

当前公式：

```text
p_C = (R_target<-camera)^T * (p_L - t_target<-camera)
```

然后通过相机模型 `project_camera_point_to_image(...)` 投到像素，再与 `target_person_mask` 做命中判断。

### 11.10 当前转换链路的关键风险点

1. `heading` 方向假设（`90-heading`）可能与设备定义不一致
2. `base_to_span` 旋转是否确实是 `diag(1,-1,-1)`（即 `[0,0,π]` 对应轴顺序需确认）
3. 相机-雷达外参来源与 `aruco_context` 定义方向是否完全一致
4. 文档中的 FLU 与设备实际输出 frame 约定可能不一致（最常见导致平行轨迹错位）
