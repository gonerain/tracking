# Coordinate Transform Reference

本文档独立汇总当前工程中与 `fusion_tracking` 相关的坐标系转换与应用位置。

## 1. 坐标系定义

- `L` (LiDAR frame): 车体点云坐标系（当前实现按 FLU 处理）
- `B` (base_link): 机器人基坐标系（FLU）
- `S` (SPAN body): SPAN/IE 本体坐标系（FRD）
- `NED`: North-East-Down 本地导航坐标系
- `ENU`: East-North-Up 本地导航坐标系
- `ECEF`: Earth-Centered, Earth-Fixed
- `LLA`: Latitude, Longitude, Altitude
- `C` (camera): 相机坐标系（用于点云投影到像素）

## 2. 外参链（LiDAR / base / SPAN）

配置来源：`configs/lidar_config.yaml -> ie_lidar_extrinsics`

- `T_B<-L`: `base_from_lidar`
- `T_B<-S`: `base_to_span`

代码：`core/fusion_tracking.py -> configure_span_lidar_extrinsics()`

内部计算：

```text
R_S<-B = (R_B<-S)^T
t_S<-B = -R_S<-B * t_B<-S

R_S<-L = R_S<-B * R_B<-L
t_S<-L = R_S<-B * t_B<-L + t_S<-B

R_L<-S = (R_S<-L)^T
t_L<-S = -R_L<-S * t_S<-L
```

运行时生效变量：

- `R_SPAN_FROM_LIDAR_ACTIVE`, `T_SPAN_FROM_LIDAR_M_ACTIVE`
- `R_LIDAR_FROM_SPAN_ACTIVE`, `T_LIDAR_FROM_SPAN_M_ACTIVE`

## 3. 点级变换函数

代码：`core/fusion_tracking.py`

### 3.1 LiDAR -> SPAN

函数：`lidar_to_ie_body(point_lidar)`

```text
p_S = R_S<-L * p_L + t_S<-L
```

### 3.2 SPAN -> LiDAR

函数：`ie_body_to_lidar(point_body)`

```text
p_L = R_L<-S * p_S + t_L<-S
```

## 4. 姿态旋转（SPAN CPT7）

函数：`rotation_matrix_from_ie(roll_deg, pitch_deg, heading_deg)`

CPT7 的 RPY 输出定义了从 ENU 到 **CPT7 内部 RFU (Right/Forward/Up)** 的基变换：

```text
C_{ENU -> RFU} = R_y(R) * R_x(P) * R_z(-A)
```

其中 `R_*(θ)` 均为 passive 旋转（基变换）。

本项目车体（base_link）为 FLU，下游 `lidar_to_ie_body` 输出的是 SPAN 原点处的
车体 FLU 坐标，因此先取 `R_{ENU <- RFU} = C_{ENU->RFU}^T`，再乘上 CPT7 RFU 到
车体 FLU 的静态 90° 旋转：

```text
R_{RFU <- FLU} =
[[ 0, -1, 0],
 [ 1,  0, 0],
 [ 0,  0, 1]]

R_{ENU <- FLU} = R_{ENU <- RFU} * R_{RFU <- FLU}
```

注：`R_{RFU <- FLU}` 为绕 z 的 90° 旋转（det = +1），而不是 FRD↔RFU 的轴交换。

## 5. 地理坐标转换

代码：`core/fusion_tracking.py`

- `_geodetic_to_ecef(lat, lon, h)`
- `_ecef_to_geodetic(ecef)`
- `_ecef_to_enu_matrix(lat, lon)`
- `_enu_to_ecef_matrix(lat, lon)`

核心关系：

```text
p_ENU = R_ENU<-ECEF * (p_ECEF - p_ECEF_origin)
p_ECEF = p_ECEF_origin + R_ECEF<-ENU * p_ENU
```

说明：`p_ECEF_origin` 取 IE 轨迹第一帧。

## 6. 时序点云补偿中的变换应用

函数：`transform_points_lidar_between_ie_poses(...)`

每个点流程：

```text
p_L(src)
 -> p_S(src)              (LiDAR->SPAN外参)
 -> p_ENU(world)          (src姿态 + src平移)
 -> p_S(dst)              (去dst平移 + dst姿态逆)
 -> p_L(dst)              (SPAN->LiDAR外参)
```

用途：`temporal_fusion` 把邻近帧点云对齐到中心帧。

## 7. 世界轨迹输出中的变换应用

代码段：`core/fusion_tracking.py` 构造 `world_record`

输入：`target_lidar`（tracker 输出，LiDAR 坐标）

流程：

```text
target_lidar (L)
 -> target_body (S)                     via lidar_to_ie_body
 -> target_offset_enu                   via R_ENU<-body
 -> target_ecef                         via ENU->ECEF + span_ecef
 -> target_world_lla                    via ecef_to_geodetic
 -> target_world_enu_m                  via origin ECEF to ENU
```

输出字段：

- `target_lidar_m`
- `target_world_lla`
- `target_world_enu_m`
- `ie_pose`

## 8. 点云投影到 ArUco mask 的变换应用（aruco_mask_v2）

函数：

- `lidar_point_to_camera(point_lidar, aruco_context)`
- `select_points_in_target_mask(...)`

变换：

```text
p_C = (R_target<-camera)^T * (p_L - t_target<-camera)
```

然后执行：

1. 相机投影到像素 (`project_camera_point_to_image`)
2. 像素与 `target_person_mask` 命中判断
3. 命中点作为 `aruco_mask_v2` 候选点来源

## 9. 关键应用点索引

- 外参加载：`configure_span_lidar_extrinsics()`
- 姿态矩阵：`rotation_matrix_from_ie()`
- 时序补偿：`transform_points_lidar_between_ie_poses()`
- 轨迹输出：`world_record` 构造段
- mask 投影：`select_points_in_target_mask()`
- v2 候选：`build_target_mask_v2_candidates()`

## 10. 当前最易出错的转换环节

1. `base_to_span` 方向语义（`T_B<-S`，span_link 定义为 FRD，而 base_link 为 FLU）
2. CPT7 输出的 RPY 是 `C_{ENU->RFU}`（CPT7 内部 RFU），不是 `C_{ENU->FLU}` 也不是 `C_{NED<-FRD}`
3. CPT7 RFU 与车体 FLU 之间差一个 90° 绕 z 的旋转（不是 FRD↔RFU 轴交换）
4. `rotation_matrix_from_ie` 必须返回 `R_{ENU<-FLU}`，与 `lidar_to_ie_body` 的 FLU 输出匹配
5. camera-lidar 外参与 `aruco_context` 旋转方向

