# Coordinate Matrices Sequence

本文档按当前实现与当前配置，给出从 LiDAR 点到世界坐标计算中涉及的变换矩阵顺序与数值。

## 0. 约定

- 除 SPAN body 外，其它车体系按 FLU。
- SPAN body 为 FRD。
- 配置中给的是：
  - `T_B<-L` (`base_from_lidar`)
  - `T_B<-S` (`base_to_span`)

记号：

- `H_A<-B`：4x4 齐次变换，表示把 B 系点转换到 A 系。
- `R_A<-B`：3x3 旋转，`t_A<-B`：3x1 平移。

---

## 1) LiDAR -> Base

`H_B<-L`

```text
[[ 0.9063,  0.    ,  0.4226,  0.0315],
 [ 0.    ,  1.    ,  0.    ,  0.    ],
 [-0.4226,  0.    ,  0.9063,  0.1314],
 [ 0.    ,  0.    ,  0.    ,  1.    ]]
```

---

## 2) Span(FRD) -> Base(FLU) [配置给定]

`H_B<-S`

```text
[[ 1.    ,  0.    ,  0.    , -0.2684],
 [ 0.    , -1.    ,  0.    , -0.082 ],
 [ 0.    ,  0.    , -1.    , -0.1527],
 [ 0.    ,  0.    ,  0.    ,  1.    ]]
```

---

## 3) Base -> Span(FLU) [由上一步求逆]

`H_S<-B`

```text
[[ 1.    ,  0.    ,  0.    ,  0.2684],
 [ 0.    , -1.    ,  0.    , -0.082 ],
 [ 0.    ,  0.    , -1.    , -0.1527],
 [ 0.    ,  0.    ,  0.    ,  1.    ]]
```

---

## 4) LiDAR -> Span(FLU)

`H_S(flu)<-L = H_S<-B * H_B<-L`

```text
[[ 0.9063,  0.    ,  0.4226,  0.2999],
 [ 0.    , -1.    ,  0.    , -0.082 ],
 [ 0.4226,  0.    , -0.9063, -0.2841],
 [ 0.    ,  0.    ,  0.    ,  1.    ]]
```

---

## 5) 轴系变换：FLU -> FRD

`R_FRD<-FLU`

```text
[[ 1.,  0.,  0.],
 [ 0., -1.,  0.],
 [ 0.,  0., -1.]]
```

齐次形式 `H_FRD<-FLU`（平移为 0）：

```text
[[ 1.,  0.,  0.,  0.],
 [ 0., -1.,  0.,  0.],
 [ 0.,  0., -1.,  0.],
 [ 0.,  0.,  0.,  1.]]
```

---

## 6) LiDAR -> Span(FRD)

`H_S(frd)<-L = H_FRD<-FLU * H_S(flu)<-L`

```text
[[ 0.9063,  0.    ,  0.4226,  0.2999],
 [ 0.    ,  1.    ,  0.    ,  0.082 ],
 [-0.4226,  0.    ,  0.9063,  0.2841],
 [ 0.    ,  0.    ,  0.    ,  1.    ]]
```

---

## 7) Span(FRD) -> LiDAR [求逆]

`H_L<-S(frd)`

```text
[[ 0.9063  ,  0.      , -0.4226  , -0.151739],
 [ 0.      ,  1.      ,  0.      , -0.082   ],
 [ 0.4226  ,  0.      ,  0.9063  , -0.384218],
 [ 0.      ,  0.      ,  0.      ,  1.      ]]
```

---

## 8) 姿态：Body(FRD) -> NED（每帧动态）

由 IE/SPAN 的 `roll/pitch/heading` 构建：

```text
R_NED<-body = Rz(heading) * Rx(pitch) * Ry(roll)
```

其中：

- `heading`: body x 轴与正北夹角，顺时针为正。
- `pitch`: 绕 x 轴右手旋转。
- `roll`: 绕 y 轴右手旋转。
- 以上与 NovAtel SPAN 的姿态字段定义一致，采用 Z-X-Y 序列。

---

## 9) NED -> ENU（固定）

`R_ENU<-NED`

```text
[[ 0.,  1.,  0.],
 [ 1.,  0.,  0.],
 [ 0.,  0., -1.]]
```

---

## 10) Body(FRD) -> ENU（每帧动态）

```text
R_ENU<-body = R_ENU<-NED * R_NED<-body
```

这个矩阵用于：

- 时序点云补偿（跨帧点从 src body 转到 world，再到 dst body）
- target 世界坐标输出（`target_lidar -> target_body -> target_world`）

### 10.1 ENU -> Span/Body(FRD) 的矩阵计算过程

目标是把世界系 ENU 点转换到 SPAN body（FRD）：

```text
p_body = R_body<-ENU * (p_ENU - p_span_ENU)
```

其中 `p_span_ENU` 是该帧 SPAN 原点在 ENU 中的位置。

先由 IE/SPAN 姿态构建 `R_ENU<-body`，再转置得到 `R_body<-ENU`：

```text
R_body<-ENU = (R_ENU<-body)^T
```

#### Step 1: Body(FRD) -> NED

按当前实现：

```text
R_NED<-body = Rz(heading) * Rx(pitch) * Ry(roll)
```

```text
Rx(pitch) =
[[1, 0, 0],
 [0, cos(pitch), -sin(pitch)],
 [0, sin(pitch),  cos(pitch)]]
```

```text
Ry(roll) =
[[ cos(roll), 0, sin(roll)],
 [ 0,         1, 0        ],
 [-sin(roll), 0, cos(roll)]]
```

```text
Rz(heading) =
[[cos(heading), -sin(heading), 0],
 [sin(heading),  cos(heading), 0],
 [0,             0,            1]]
```

注：`heading` 为 NED 中相对正北顺时针角（与代码一致）。

#### Step 2: NED -> ENU

```text
R_ENU<-NED =
[[0, 1,  0],
 [1, 0,  0],
 [0, 0, -1]]
```

#### Step 3: Body(FRD) -> ENU

```text
R_ENU<-body = R_ENU<-NED * R_NED<-body
```

#### Step 4: ENU -> Body(FRD)

```text
R_body<-ENU = (R_ENU<-body)^T
```

代入点变换即：

```text
p_body = (R_ENU<-body)^T * (p_ENU - p_span_ENU)
```

与当前代码的行向量写法等价：

```text
pts_body = (pts_world - span_enu) @ R_ENU<-body
```

---

## 11) 地理坐标相关（每帧动态）

- `LLA -> ECEF`
- `ECEF -> ENU`

核心关系：

```text
p_ENU  = R_ENU<-ECEF * (p_ECEF - p_ECEF_origin)
p_ECEF = p_ECEF_origin + R_ECEF<-ENU * p_ENU
```

`p_ECEF_origin` 使用第一帧 IE pose。

### 11.1 ECEF -> ENU 具体计算

设 ENU 原点地理坐标为 `(lat0, lon0, h0)`，先转成：

```text
p0_ECEF = LLA2ECEF(lat0, lon0, h0)
```

任一点 `p_ECEF = [x, y, z]^T` 到 ENU：

```text
dx = p_ECEF - p0_ECEF
p_ENU = R_ENU<-ECEF(lat0, lon0) * dx
```

其中（`lat0/lon0` 用弧度）：

```text
R_ENU<-ECEF =
[[-sin(lon0),              cos(lon0),             0],
 [-sin(lat0)*cos(lon0), -sin(lat0)*sin(lon0),  cos(lat0)],
 [ cos(lat0)*cos(lon0),  cos(lat0)*sin(lon0),  sin(lat0)]]
```

对应分量可写为：

```text
e = -sin(lon0)*dx + cos(lon0)*dy
n = -sin(lat0)*cos(lon0)*dx - sin(lat0)*sin(lon0)*dy + cos(lat0)*dz
u =  cos(lat0)*cos(lon0)*dx + cos(lat0)*sin(lon0)*dy + sin(lat0)*dz
```

反变换：

```text
R_ECEF<-ENU = (R_ENU<-ECEF)^T
p_ECEF = p0_ECEF + R_ECEF<-ENU * p_ENU
```

注：

- 这里的 ENU 是局部切平面坐标，原点固定在 `p0_ECEF`（本项目使用第一帧 IE pose）。
- 若 `lat/lon` 误用角度制（degree）直接代入三角函数，会导致明显方向和尺度错误。

---

## 12) 在代码中的应用顺序（点级）

以 LiDAR 点 `p_L` 为例：

```text
p_L
 -> p_S(frd)                [H_S(frd)<-L]
 -> p_ENU/world             [R_ENU<-body + span position]
 -> p_S(frd)_dst            [目标帧姿态/位姿逆]
 -> p_L_dst                 [H_L<-S(frd)]
```

以上就是当前实现中涉及到的完整变换链。
