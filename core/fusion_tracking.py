from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.detect_aruco import FrameSource, build_detector_context, detect_markers, draw_results, load_config as load_aruco_config
from core.geometry import project_camera_point_to_image
from core.lidar_tracking import (
    ClusterCandidate,
    JsonRecordWriter,
    RosbagLidarSource,
    TrackState,
    compute_candidate,
    crop_points,
    euclidean_clusters,
    filter_candidates,
    format_candidate,
    merge_vertical_person_clusters,
    remove_ego_vehicle_points,
    remove_ground,
    render_debug_image,
    render_segmentation_preview,
    render_target_crop_image,
    stack_debug_images,
)
from core.segmentation import SegmentationPredictor, extract_person_masks, select_relevant_person_masks
from core.tracking import (
    draw_tracking_overlay,
    get_target_id_groups,
    get_target_ids,
    get_target_offsets,
    select_targets,
    target_ids_label,
)


@dataclass
class ArucoPrior:
    timestamp: float
    visible: bool
    source: str
    position_lidar: np.ndarray


@dataclass
class ArucoDebugState:
    frame: np.ndarray | None
    frame_timestamp: float | None
    sync_dt_ms: float | None
    results: list[dict[str, Any]]
    tracking_state: Any
    target_result: dict[str, Any] | None
    target_results: list[dict[str, Any]]
    segmentation: Any
    target_person_mask: np.ndarray | None
    selected_person_masks: list[np.ndarray]
    raw_person_mask_count: int


@dataclass
class MultiTrack:
    track_id: int
    state: TrackState
    hits: int
    misses: int


class MultiObjectTracker:
    def __init__(
        self,
        gating_distance_m: float,
        max_prediction_duration_s: float,
        process_gain: float,
        allow_prediction: bool,
        confirm_hits: int,
        forget_misses: int,
    ) -> None:
        self.gating_distance_m = float(gating_distance_m)
        self.max_prediction_duration_s = float(max_prediction_duration_s)
        self.process_gain = float(process_gain)
        self.allow_prediction = bool(allow_prediction)
        self.confirm_hits = max(int(confirm_hits), 1)
        self.forget_misses = max(int(forget_misses), 1)
        self._next_track_id = 1
        self._tracks: dict[int, MultiTrack] = {}

    def _predict_state(self, track: MultiTrack, timestamp: float) -> TrackState | None:
        dt = float(timestamp - track.state.timestamp)
        if dt < 0.0:
            return None
        if dt == 0.0:
            return track.state
        if dt > self.max_prediction_duration_s:
            return None
        predicted_position = track.state.position + track.state.velocity * dt
        return TrackState(
            timestamp=timestamp,
            position=predicted_position,
            velocity=track.state.velocity.copy(),
            source="predicted",
            candidate=None,
            track_quality=track.state.track_quality,
            track_lifecycle=track.state.track_lifecycle,
        )

    def _new_track(self, timestamp: float, candidate: ClusterCandidate) -> None:
        track_id = self._next_track_id
        self._next_track_id += 1
        lifecycle = "confirmed" if self.confirm_hits <= 1 else "tentative"
        self._tracks[track_id] = MultiTrack(
            track_id=track_id,
            state=TrackState(
                timestamp=timestamp,
                position=candidate.footpoint.copy(),
                velocity=np.zeros(3, dtype=np.float64),
                source="observed",
                candidate=candidate,
                track_quality=1.0,
                track_lifecycle=lifecycle,
            ),
            hits=1,
            misses=0,
        )

    def update(self, timestamp: float, candidates: list[ClusterCandidate]) -> list[MultiTrack]:
        timestamp = float(timestamp)
        if not self._tracks:
            for candidate in candidates:
                self._new_track(timestamp, candidate)
            return self._reported_tracks()

        track_ids = sorted(self._tracks.keys())
        predicted_states = {track_id: self._predict_state(self._tracks[track_id], timestamp) for track_id in track_ids}

        pair_costs: list[tuple[float, int, int]] = []
        for candidate_index, candidate in enumerate(candidates):
            for track_id in track_ids:
                predicted = predicted_states.get(track_id)
                if predicted is None:
                    continue
                distance = float(np.linalg.norm(candidate.footpoint[:2] - predicted.position[:2]))
                if distance <= self.gating_distance_m:
                    pair_costs.append((distance, track_id, candidate_index))
        pair_costs.sort(key=lambda item: item[0])

        assigned_tracks: set[int] = set()
        assigned_candidates: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _, track_id, candidate_index in pair_costs:
            if track_id in assigned_tracks or candidate_index in assigned_candidates:
                continue
            assigned_tracks.add(track_id)
            assigned_candidates.add(candidate_index)
            matches.append((track_id, candidate_index))

        for track_id, candidate_index in matches:
            track = self._tracks[track_id]
            candidate = candidates[candidate_index]
            predicted = predicted_states.get(track_id)
            if predicted is None:
                predicted_position = track.state.position.copy()
            else:
                predicted_position = predicted.position
            dt = max(timestamp - track.state.timestamp, 1e-3)
            blended_position = predicted_position + self.process_gain * (candidate.footpoint - predicted_position)
            new_velocity = (blended_position - track.state.position) / dt
            track.hits += 1
            track.misses = 0
            lifecycle = "confirmed" if track.hits >= self.confirm_hits else "tentative"
            track.state = TrackState(
                timestamp=timestamp,
                position=blended_position,
                velocity=new_velocity,
                source="observed",
                candidate=candidate,
                track_quality=float(track.hits) / float(max(track.hits + track.misses, 1)),
                track_lifecycle=lifecycle,
            )

        lost_track_ids: list[int] = []
        for track_id in track_ids:
            if track_id in assigned_tracks:
                continue
            track = self._tracks[track_id]
            track.misses += 1
            predicted = predicted_states.get(track_id)
            if predicted is not None and self.allow_prediction:
                predicted.track_lifecycle = track.state.track_lifecycle
                predicted.track_quality = float(track.hits) / float(max(track.hits + track.misses, 1))
                track.state = predicted
            if track.misses > self.forget_misses:
                lost_track_ids.append(track_id)

        for track_id in lost_track_ids:
            self._tracks.pop(track_id, None)

        for candidate_index, candidate in enumerate(candidates):
            if candidate_index not in assigned_candidates:
                self._new_track(timestamp, candidate)

        return self._reported_tracks()

    def _reported_tracks(self) -> list[MultiTrack]:
        tracks = [track for track in self._tracks.values() if track.hits >= self.confirm_hits]
        tracks.sort(key=lambda item: item.track_id)
        return tracks


def build_ground_protect_regions(
    track_positions: list[np.ndarray],
    aruco_priors: list[ArucoPrior],
    ground_cfg: dict[str, Any],
) -> list[dict[str, float]]:
    regions: list[dict[str, float]] = []
    for pos in track_positions:
        regions.append(
            {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "radius_m": float(ground_cfg.get("track_protect_radius_m", 0.4)),
                "z_margin_m": float(ground_cfg.get("track_protect_z_margin_m", 0.8)),
            }
        )
    for aruco_prior in aruco_priors:
        if not bool(aruco_prior.visible):
            continue
        pos = aruco_prior.position_lidar
        regions.append(
            {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "radius_m": float(ground_cfg.get("aruco_protect_radius_m", 0.35)),
                "z_margin_m": float(ground_cfg.get("aruco_protect_z_margin_m", 0.75)),
            }
        )
    return regions


@dataclass
class IePose:
    timestamp: float
    latitude_deg: float
    longitude_deg: float
    height_m: float
    roll_deg: float
    pitch_deg: float
    heading_deg: float


class TargetWorldWriter:
    def __init__(self, output_path: Path, flush_every_frame: bool) -> None:
        self.output_path = output_path
        self.flush_every_frame = bool(flush_every_frame)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self.flush_every_frame:
            self.flush()

    def flush(self) -> None:
        payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in self.records)
        if payload:
            payload += "\n"
        self.output_path.write_text(payload, encoding="utf-8")


def _dms_to_deg(deg_token: str, minute_token: str, second_token: str) -> float:
    deg = float(deg_token)
    minute = float(minute_token)
    second = float(second_token)
    sign = -1.0 if deg < 0 else 1.0
    return sign * (abs(deg) + minute / 60.0 + second / 3600.0)


def load_ie_poses(ie_path: Path) -> list[IePose]:
    if not ie_path.exists():
        return []
    poses: list[IePose] = []
    for raw_line in ie_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or not re.match(r"^[0-9]", line):
            continue
        parts = line.split()
        # Example:
        # ts week gps latD latM latS lonD lonM lonS h ... roll pitch heading Q
        if len(parts) < 18:
            continue
        try:
            timestamp = float(parts[0])
            lat_deg = _dms_to_deg(parts[3], parts[4], parts[5])
            lon_deg = _dms_to_deg(parts[6], parts[7], parts[8])
            height_m = float(parts[9])
            roll_deg = float(parts[-4])
            pitch_deg = float(parts[-3])
            heading_deg = float(parts[-2])
        except ValueError:
            continue
        poses.append(
            IePose(
                timestamp=timestamp,
                latitude_deg=lat_deg,
                longitude_deg=lon_deg,
                height_m=height_m,
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                heading_deg=heading_deg,
            )
        )
    poses.sort(key=lambda item: item.timestamp)
    return poses


class IePoseProvider:
    def __init__(self, poses: list[IePose]) -> None:
        self.poses = poses
        self._timestamps = [pose.timestamp for pose in poses]

    @property
    def available(self) -> bool:
        return bool(self.poses)

    def get_nearest(self, timestamp: float) -> IePose | None:
        if not self.poses:
            return None
        right_index = bisect.bisect_right(self._timestamps, float(timestamp))
        current_index = max(0, min(len(self.poses) - 1, right_index - 1))
        current = self.poses[current_index]
        if current_index + 1 < len(self.poses):
            nxt = self.poses[current_index + 1]
            if abs(nxt.timestamp - timestamp) < abs(current.timestamp - timestamp):
                return nxt
        return current

    @staticmethod
    def _lerp(a: float, b: float, alpha: float) -> float:
        return float(a) + (float(b) - float(a)) * float(alpha)

    @staticmethod
    def _lerp_angle_deg(a_deg: float, b_deg: float, alpha: float) -> float:
        # Interpolate with shortest angular path.
        a = float(a_deg)
        b = float(b_deg)
        delta = (b - a + 180.0) % 360.0 - 180.0
        return a + delta * float(alpha)

    def get_interpolated(self, timestamp: float) -> IePose | None:
        if not self.poses:
            return None
        right_index = bisect.bisect_right(self._timestamps, float(timestamp))
        left_index = max(0, min(len(self.poses) - 1, right_index - 1))
        left = self.poses[left_index]
        if left_index + 1 >= len(self.poses):
            return left
        right = self.poses[left_index + 1]
        dt = float(right.timestamp - left.timestamp)
        if dt <= 1e-9:
            return left
        alpha = float(timestamp - left.timestamp) / dt
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return IePose(
            timestamp=float(timestamp),
            latitude_deg=self._lerp(left.latitude_deg, right.latitude_deg, alpha),
            longitude_deg=self._lerp(left.longitude_deg, right.longitude_deg, alpha),
            height_m=self._lerp(left.height_m, right.height_m, alpha),
            roll_deg=self._lerp_angle_deg(left.roll_deg, right.roll_deg, alpha),
            pitch_deg=self._lerp_angle_deg(left.pitch_deg, right.pitch_deg, alpha),
            heading_deg=self._lerp_angle_deg(left.heading_deg, right.heading_deg, alpha),
        )


def _deg2rad(value_deg: float) -> float:
    return float(value_deg) * math.pi / 180.0


def rotation_matrix_from_ie(roll_deg: float, pitch_deg: float, heading_deg: float) -> np.ndarray:
    # SPAN CPT7 attitude output defines the change-of-basis from ENU to its
    # internal RFU (Right/Forward/Up) body frame:
    #   C_{ENU -> RFU} = R_y(R) * R_x(P) * R_z(-A)
    # where each R_*(theta) is a passive rotation (basis change) about the named
    # axis. The vehicle (base_link) frame in this project is FLU, so after
    # recovering R_{ENU <- RFU} we apply the static RFU -> FLU swap (a 90 deg
    # rotation about z) to return R_{ENU <- FLU}. Downstream code consumes body
    # points in vehicle FLU coords centered at the SPAN origin (see
    # lidar_to_ie_body).
    roll = _deg2rad(roll_deg)
    pitch = _deg2rad(pitch_deg)
    neg_azimuth = -_deg2rad(heading_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(neg_azimuth), math.sin(neg_azimuth)
    # Passive rotations (= active^T) matching the CPT7 formula.
    ry_passive_R = np.array(
        [[cr, 0.0, -sr], [0.0, 1.0, 0.0], [sr, 0.0, cr]],
        dtype=np.float64,
    )
    rx_passive_P = np.array(
        [[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]],
        dtype=np.float64,
    )
    rz_passive_negA = np.array(
        [[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    r_rfu_from_enu = ry_passive_R @ rx_passive_P @ rz_passive_negA
    r_enu_from_rfu = r_rfu_from_enu.T
    # R_{RFU <- FLU}: vehicle FLU (x=F, y=L, z=U) expressed in CPT7 RFU axes.
    # FLU x (Forward) -> RFU [0,1,0]; FLU y (Left) -> RFU [-1,0,0]; FLU z -> [0,0,1].
    r_rfu_from_flu = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return r_enu_from_rfu @ r_rfu_from_flu


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    # WGS84
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = _deg2rad(lat_deg)
    lon = _deg2rad(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def _ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = _deg2rad(lat_deg)
    lon = _deg2rad(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )


def _enu_to_ecef_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    return _ecef_to_enu_matrix(lat_deg, lon_deg).T


def _ecef_to_geodetic(ecef_xyz: np.ndarray) -> tuple[float, float, float]:
    # Bowring iterative solution, WGS84.
    a = 6378137.0
    e2 = 6.69437999014e-3
    x, y, z = float(ecef_xyz[0]), float(ecef_xyz[1]), float(ecef_xyz[2])
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(6):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * n * sin_lat, p)
    sin_lat = math.sin(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    h = p / max(math.cos(lat), 1e-12) - n
    return lat * 180.0 / math.pi, lon * 180.0 / math.pi, h


# Measured extrinsics (FLU convention):
# T_base<-lidar from CAD TF:
# R =
# [[ 0.9063, 0.0,  0.4226],
#  [ 0.0,    1.0,  0.0   ],
#  [-0.4226, 0.0,  0.9063]]
# t = [31.5, 0.0, 131.4] mm
R_BASE_FROM_LIDAR = np.array(
    [
        [0.9063, 0.0, 0.4226],
        [0.0, 1.0, 0.0],
        [-0.4226, 0.0, 0.9063],
    ],
    dtype=np.float64,
)
T_BASE_FROM_LIDAR_M = np.array([0.0315, 0.0, 0.1314], dtype=np.float64)

# T_base<-span from extrinsics document (base_link_to_span_link).
R_BASE_FROM_SPAN = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
T_BASE_FROM_SPAN_M = np.array([-0.2684, -0.0820, -0.1527], dtype=np.float64)

# Inverse to obtain T_span<-base
R_SPAN_FROM_BASE = R_BASE_FROM_SPAN.T
T_SPAN_FROM_BASE_M = -R_SPAN_FROM_BASE @ T_BASE_FROM_SPAN_M

# Compose T_span<-lidar = T_span<-base * T_base<-lidar
R_SPAN_FROM_LIDAR = R_SPAN_FROM_BASE @ R_BASE_FROM_LIDAR
T_SPAN_FROM_LIDAR_M = R_SPAN_FROM_BASE @ T_BASE_FROM_LIDAR_M + T_SPAN_FROM_BASE_M

# Inverse to obtain T_lidar<-span
R_LIDAR_FROM_SPAN = R_SPAN_FROM_LIDAR.T
T_LIDAR_FROM_SPAN_M = -R_LIDAR_FROM_SPAN @ T_SPAN_FROM_LIDAR_M

# Active extrinsics (can be overridden by config at runtime)
R_SPAN_FROM_LIDAR_ACTIVE = R_SPAN_FROM_LIDAR.copy()
T_SPAN_FROM_LIDAR_M_ACTIVE = T_SPAN_FROM_LIDAR_M.copy()
R_LIDAR_FROM_SPAN_ACTIVE = R_LIDAR_FROM_SPAN.copy()
T_LIDAR_FROM_SPAN_M_ACTIVE = T_LIDAR_FROM_SPAN_M.copy()


def _as_3x3_matrix(value: Any, default: np.ndarray) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(3, 3)
        return arr
    except Exception:
        return default


def _as_3_vector(value: Any, default: np.ndarray) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(3)
        return arr
    except Exception:
        return default


def configure_span_lidar_extrinsics(lidar_config: dict[str, Any]) -> None:
    cfg = dict(lidar_config.get("ie_lidar_extrinsics", {}))
    base_from_lidar = dict(cfg.get("base_from_lidar", {}))
    # Prefer base_to_span (T_base<-span), keep span_from_base as legacy fallback.
    base_to_span = dict(cfg.get("base_to_span", {}))
    span_from_base_legacy = dict(cfg.get("span_from_base", {}))

    r_base_from_lidar = _as_3x3_matrix(base_from_lidar.get("rotation_3x3", R_BASE_FROM_LIDAR), R_BASE_FROM_LIDAR)
    t_base_from_lidar = _as_3_vector(base_from_lidar.get("translation_m", T_BASE_FROM_LIDAR_M), T_BASE_FROM_LIDAR_M)
    if base_to_span:
        r_base_from_span = _as_3x3_matrix(base_to_span.get("rotation_3x3", R_BASE_FROM_SPAN), R_BASE_FROM_SPAN)
        t_base_from_span = _as_3_vector(base_to_span.get("translation_m", T_BASE_FROM_SPAN_M), T_BASE_FROM_SPAN_M)
        r_span_from_base = r_base_from_span.T
        t_span_from_base = -r_span_from_base @ t_base_from_span
    else:
        # Backward compatibility: interpret legacy key directly as T_span<-base.
        r_span_from_base = _as_3x3_matrix(span_from_base_legacy.get("rotation_3x3", R_SPAN_FROM_BASE), R_SPAN_FROM_BASE)
        t_span_from_base = _as_3_vector(span_from_base_legacy.get("translation_m", T_SPAN_FROM_BASE_M), T_SPAN_FROM_BASE_M)

    r_span_from_lidar = r_span_from_base @ r_base_from_lidar
    t_span_from_lidar = r_span_from_base @ t_base_from_lidar + t_span_from_base
    r_lidar_from_span = r_span_from_lidar.T
    t_lidar_from_span = -r_lidar_from_span @ t_span_from_lidar

    global R_SPAN_FROM_LIDAR_ACTIVE, T_SPAN_FROM_LIDAR_M_ACTIVE
    global R_LIDAR_FROM_SPAN_ACTIVE, T_LIDAR_FROM_SPAN_M_ACTIVE
    R_SPAN_FROM_LIDAR_ACTIVE = r_span_from_lidar
    T_SPAN_FROM_LIDAR_M_ACTIVE = t_span_from_lidar
    R_LIDAR_FROM_SPAN_ACTIVE = r_lidar_from_span
    T_LIDAR_FROM_SPAN_M_ACTIVE = t_lidar_from_span


def lidar_to_ie_body(point_lidar: np.ndarray) -> np.ndarray:
    # Output: point in vehicle FLU axes, translated so the origin coincides
    # with the SPAN body origin. The intermediate `R_SPAN_FROM_LIDAR @ p + t`
    # produces span FRD coords; the subsequent [x, -y, -z] converts FRD back
    # to vehicle FLU while preserving the span-origin offset.
    p_l = np.asarray(point_lidar, dtype=np.float64).reshape(3)
    p_span_frd = R_SPAN_FROM_LIDAR_ACTIVE @ p_l + T_SPAN_FROM_LIDAR_M_ACTIVE
    return np.array([p_span_frd[0], -p_span_frd[1], -p_span_frd[2]], dtype=np.float64)


def ie_body_to_lidar(point_body: np.ndarray) -> np.ndarray:
    # Inverse of lidar_to_ie_body. Input is vehicle FLU centered at SPAN origin.
    p_body_flu = np.asarray(point_body, dtype=np.float64).reshape(3)
    p_span_frd = np.array([p_body_flu[0], -p_body_flu[1], -p_body_flu[2]], dtype=np.float64)
    return R_LIDAR_FROM_SPAN_ACTIVE @ p_span_frd + T_LIDAR_FROM_SPAN_M_ACTIVE


def ie_pose_to_enu(
    pose: IePose,
    ecef_origin: np.ndarray,
    enu_from_ecef: np.ndarray,
) -> np.ndarray:
    ecef = _geodetic_to_ecef(pose.latitude_deg, pose.longitude_deg, pose.height_m)
    return enu_from_ecef @ (ecef - ecef_origin)


class ArucoPriorProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.context = build_detector_context(config)
        self.frame_source = FrameSource(config)
        tracking_cfg = config.get("tracking", {})
        self.target_id_groups = get_target_id_groups(tracking_cfg)
        self.target_ids = get_target_ids(tracking_cfg)
        self.target_offsets = get_target_offsets(tracking_cfg, self.target_ids)
        self.target_label = target_ids_label(self.target_id_groups)
        self.latest_timestamp: float | None = None
        self.pending_frame: tuple[np.ndarray, float] | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_results: list[dict[str, Any]] = []
        self.latest_tracking_state: Any = None
        self.latest_target_result: dict[str, Any] | None = None
        self.latest_target_results: list[dict[str, Any]] = []
        self.segmentation_predictor = SegmentationPredictor(config)
        self.segmentation_cfg = config.get("segmentation", {})
        self.latest_segmentation: Any = None
        self.latest_target_person_mask: np.ndarray | None = None
        self.latest_selected_person_masks: list[np.ndarray] = []
        self.latest_raw_person_mask_count = 0
        sync_cfg = config.get("sync", {})
        self.sync_max_dt_s = float(sync_cfg.get("camera_sync_max_dt_s", 0.08))
        self.latest_sync_dt_ms: float | None = None

    def _read_next_frame(self) -> tuple[np.ndarray | None, float | None]:
        if self.pending_frame is not None:
            frame, frame_timestamp = self.pending_frame
            self.pending_frame = None
            return frame, frame_timestamp

        ok, frame, frame_timestamp = self.frame_source.read()
        if not ok or frame is None or frame_timestamp is None:
            return None, None
        return frame, float(frame_timestamp)

    def _consume_until(self, timestamp: float) -> None:
        while True:
            frame, frame_timestamp = self._read_next_frame()
            if frame is None or frame_timestamp is None:
                return
            if frame_timestamp > timestamp:
                self.pending_frame = (frame, frame_timestamp)
                return

            self.latest_timestamp = frame_timestamp
            self.latest_frame = frame.copy()
            results = detect_markers(frame, self.context)
            self.latest_results = results
            target_results = select_targets(results, self.target_id_groups, self.target_offsets)
            target = None if not target_results else max(target_results, key=lambda item: (len(item.get("visible_ids", [])), len(item.get("target_ids", []))))
            self.latest_target_result = target
            self.latest_target_results = target_results
            self.latest_segmentation = self.segmentation_predictor.predict(frame)
            (
                self.latest_target_person_mask,
                self.latest_selected_person_masks,
                self.latest_raw_person_mask_count,
            ) = select_target_person_mask(
                frame=frame,
                segmentation=self.latest_segmentation,
                target_results=target_results,
                person_class_id=self.segmentation_predictor.person_class_id,
                min_area_px=int(self.segmentation_cfg.get("min_person_mask_area_px", 3000)),
                max_instances=int(self.segmentation_cfg.get("max_instances_considered", 3)),
            )
            self.latest_tracking_state = None

    def get_prior(self, timestamp: float) -> ArucoPrior | None:
        priors = self.get_priors(timestamp)
        return None if not priors else priors[0]

    def get_priors(self, timestamp: float) -> list[ArucoPrior]:
        self._consume_until(timestamp)
        if self.latest_timestamp is None:
            self.latest_sync_dt_ms = None
            return []
        dt_s = float(self.latest_timestamp - timestamp)
        if abs(dt_s) > self.sync_max_dt_s:
            self.latest_sync_dt_ms = None
            return []
        self.latest_sync_dt_ms = 1000.0 * dt_s

        priors: list[ArucoPrior] = []
        for target in self.latest_target_results:
            position_target = np.asarray(target.get("center_in_target_m", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3)
            priors.append(
                ArucoPrior(
                    timestamp=float(self.latest_timestamp),
                    visible=True,
                    source=f"observed_group_{int(target.get('target_group_index', -1))}",
                    position_lidar=position_target.copy(),
                )
            )
        return priors

    def release(self) -> None:
        self.frame_source.release()

    def get_debug_state(self) -> ArucoDebugState:
        return ArucoDebugState(
            frame=None if self.latest_frame is None else self.latest_frame.copy(),
            frame_timestamp=self.latest_timestamp,
            sync_dt_ms=self.latest_sync_dt_ms,
            results=list(self.latest_results),
            tracking_state=self.latest_tracking_state,
            target_result=self.latest_target_result,
            target_results=[dict(item) for item in self.latest_target_results],
            segmentation=None if self.latest_segmentation is None else self.latest_segmentation.copy(),
            target_person_mask=None if self.latest_target_person_mask is None else self.latest_target_person_mask.copy(),
            selected_person_masks=[mask.copy() for mask in self.latest_selected_person_masks],
            raw_person_mask_count=int(self.latest_raw_person_mask_count),
        )


def select_target_person_mask(
    frame: np.ndarray,
    segmentation: Any,
    target_results: list[dict[str, Any]],
    person_class_id: int = 19,
    min_area_px: int = 0,
    max_instances: int | None = None,
) -> tuple[np.ndarray | None, list[np.ndarray], int]:
    if segmentation is None or not target_results:
        return None, [], 0

    split_mode = "none" if isinstance(segmentation, dict) else "auto"
    person_masks = extract_person_masks(segmentation, person_class_id, split_mode=split_mode)
    raw_person_mask_count = len(person_masks)
    person_masks = select_relevant_person_masks(
        person_masks,
        image_shape=frame.shape,
        min_area_px=min_area_px,
        max_instances=max_instances,
    )
    if not person_masks:
        return None, [], raw_person_mask_count

    h, w = frame.shape[:2]
    target_centers: list[tuple[int, int]] = []
    for target_result in target_results:
        center = np.round(target_result["center_projected_px"]).astype(int)
        cx = int(np.clip(center[0], 0, w - 1))
        cy = int(np.clip(center[1], 0, h - 1))
        target_centers.append((cx, cy))

    selected_masks: list[np.ndarray] = []
    for cx, cy in target_centers:
        best_mask = None
        best_distance = None
        for mask in person_masks:
            mask = np.asarray(mask, dtype=bool)
            if mask[cy, cx]:
                best_mask = mask
                break
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            distances = (xs - cx) ** 2 + (ys - cy) ** 2
            nearest_distance = float(np.min(distances))
            if best_mask is None or nearest_distance < best_distance:
                best_mask = mask
                best_distance = nearest_distance
        if best_mask is not None:
            selected_masks.append(best_mask)

    if not selected_masks:
        return None, person_masks, raw_person_mask_count
    merged_mask = np.any(np.stack(selected_masks, axis=0), axis=0)
    return merged_mask, person_masks, raw_person_mask_count


def overlay_target_mask(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame
    vis = frame.copy()
    overlay = vis.copy()
    overlay[mask] = np.array([0, 255, 255], dtype=np.uint8)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    return vis


class FusionDebugWriter:
    def __init__(self, config: dict[str, Any]) -> None:
        debug_cfg = config.get("debug", {})
        tracking_cfg = config.get("tracking", {})
        self.enabled = bool(debug_cfg.get("save_images", False))
        self.every_n_frames = max(int(debug_cfg.get("every_n_frames", 1)), 1)
        self.output_dir = Path(str(debug_cfg.get("fusion_output_dir", "outputs/fusion_debug")))
        self.tile_width = int(debug_cfg.get("image_width", 1200))
        self.tile_height = int(debug_cfg.get("image_height", 900))
        self.target_label = target_ids_label(get_target_id_groups(tracking_cfg))
        if self.enabled:
            for subdir in [
                "front_camera",
                "segmentation",
                "raw_bev",
                "filtered_bev",
                "side_view",
                "target_crop",
                "overview",
                "raw_pcd",
                "filtered_pcd",
            ]:
                (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _suffix(self, frame_index: int, timestamp: float) -> str:
        return f"frame_{frame_index:06d}_{timestamp:.3f}"

    def _write_pcd(self, path: Path, points: np.ndarray) -> None:
        with path.open("w", encoding="utf-8") as file:
            file.write("# .PCD v0.7 - Point Cloud Data file format\n")
            file.write("VERSION 0.7\n")
            file.write("FIELDS x y z\n")
            file.write("SIZE 4 4 4\n")
            file.write("TYPE F F F\n")
            file.write("COUNT 1 1 1\n")
            file.write(f"WIDTH {points.shape[0]}\n")
            file.write("HEIGHT 1\n")
            file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            file.write(f"POINTS {points.shape[0]}\n")
            file.write("DATA ascii\n")
            for point in points:
                file.write(f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f}\n")

    def _render_front_camera_image(
        self,
        frame_index: int,
        lidar_timestamp: float,
        aruco_prior: ArucoPrior | None,
        aruco_selected_candidate: ClusterCandidate | None,
        aruco_debug: ArucoDebugState,
        aruco_context: Any,
    ) -> np.ndarray:
        if aruco_debug.frame is None:
            canvas = np.full((self.tile_height, self.tile_width, 3), 245, dtype=np.uint8)
            cv2.putText(canvas, "front camera unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 2)
            cv2.putText(canvas, f"lidar_ts={lidar_timestamp:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
            return canvas

        vis = overlay_target_mask(aruco_debug.frame, aruco_debug.target_person_mask)
        vis = draw_results(vis, aruco_debug.results, aruco_context)
        visible_ids = [] if aruco_debug.target_result is None else list(aruco_debug.target_result.get("visible_ids", []))
        selected_target_ids = [] if aruco_debug.target_result is None else list(aruco_debug.target_result.get("target_ids", []))
        vis = draw_tracking_overlay(vis, self.target_label, aruco_debug.tracking_state, visible_ids, selected_target_ids)
        vis = cv2.resize(vis, (self.tile_width, self.tile_height), interpolation=cv2.INTER_LINEAR)
        frame_ts_text = "none" if aruco_debug.frame_timestamp is None else f"{aruco_debug.frame_timestamp:.3f}"
        sync_dt_text = "none" if aruco_debug.sync_dt_ms is None else f"{aruco_debug.sync_dt_ms:.1f}"
        cv2.putText(
            vis,
            f"front camera frame={frame_index} cam_ts={frame_ts_text} lidar_ts={lidar_timestamp:.3f} sync_dt_ms={sync_dt_text}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        prior_text = "aruco_prior=none"
        if aruco_prior is not None:
            prior_text = f"aruco_prior={aruco_prior.source} pos={np.round(aruco_prior.position_lidar, 3).tolist()}"
        cv2.putText(vis, prior_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        match_text = "aruco_match=none"
        if aruco_selected_candidate is not None:
            match_text = f"aruco_match={np.round(aruco_selected_candidate.footpoint, 3).tolist()}"
        cv2.putText(vis, match_text, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if aruco_debug.target_person_mask is not None:
            mask_pixels = int(np.count_nonzero(aruco_debug.target_person_mask))
            cv2.putText(vis, f"target_mask_pixels={mask_pixels}", (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(
            vis,
            f"raw_masks={aruco_debug.raw_person_mask_count} selected_masks={len(aruco_debug.selected_person_masks)}",
            (20, 158),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        return vis

    def _render_overview_info_panel(
        self,
        frame_index: int,
        timestamp: float,
        raw_points: np.ndarray,
        points: np.ndarray,
        candidates: list[ClusterCandidate],
        state: Any,
        config: dict[str, Any],
        aruco_prior: ArucoPrior | None,
        aruco_selected_candidate: ClusterCandidate | None,
        aruco_debug: ArucoDebugState,
    ) -> np.ndarray:
        debug_cfg = config.get("debug", {})
        canvas_width = int(debug_cfg.get("image_width", 1200))
        canvas_height = int(debug_cfg.get("image_height", 900))
        if aruco_debug.frame is not None:
            panel = cv2.resize(aruco_debug.frame, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)
            overlay = panel.copy()
            cv2.rectangle(overlay, (0, 0), (canvas_width, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, panel)
            title = "front camera"
            title_color = (255, 255, 255)
            text_color = (255, 255, 255)
        else:
            panel = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
            title = "fusion overview"
            title_color = (40, 40, 40)
            text_color = (60, 60, 60)

        roi_cfg = config.get("roi", {})
        x_limits = tuple(float(v) for v in roi_cfg.get("x", [0.0, 20.0]))
        y_limits = tuple(float(v) for v in roi_cfg.get("y", [-10.0, 10.0]))
        z_limits = tuple(float(v) for v in roi_cfg.get("z", [-2.0, 3.0]))

        cv2.putText(panel, title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, title_color, 2)
        cv2.putText(panel, f"frame={frame_index} ts={timestamp:.3f}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2)
        cv2.putText(panel, f"raw_points={raw_points.shape[0]} filtered_points={points.shape[0]}", (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2)
        cv2.putText(panel, f"candidates={len(candidates)}", (20, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2)

        sync_text = "camera_ts=none sync_dt_ms=none"
        if aruco_debug.frame_timestamp is not None and aruco_debug.sync_dt_ms is not None:
            sync_text = f"camera_ts={aruco_debug.frame_timestamp:.3f} sync_dt_ms={aruco_debug.sync_dt_ms:.1f}"
        cv2.putText(panel, sync_text, (20, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.68, text_color, 2)

        track_text = "track=none"
        track_color = (0, 0, 180)
        if state is not None:
            track_text = f"track={state.source} pos={np.round(state.position, 3).tolist()}"
            track_color = (0, 180, 0) if state.source == "observed" else (0, 200, 200)
        cv2.putText(panel, track_text, (20, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.72, track_color, 2)

        prior_text = "aruco_prior=none"
        if aruco_prior is not None:
            prior_text = f"aruco_prior={aruco_prior.source} pos={np.round(aruco_prior.position_lidar, 3).tolist()}"
        cv2.putText(panel, prior_text, (20, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)

        match_text = "aruco_match=none"
        if aruco_selected_candidate is not None:
            match_text = f"aruco_match={np.round(aruco_selected_candidate.footpoint, 3).tolist()}"
        cv2.putText(panel, match_text, (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)

        cv2.putText(panel, f"roi x={x_limits} y={y_limits} z={z_limits}", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)
        cv2.putText(
            panel,
            f"raw_masks={aruco_debug.raw_person_mask_count} selected_masks={len(aruco_debug.selected_person_masks)}",
            (20, 398),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            text_color,
            2,
        )

        if aruco_debug.frame is None:
            cv2.putText(panel, "layout: front/info | raw bev | filtered bev", (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 80, 80), 2)
            cv2.putText(panel, "        side view | target crop | segmentation", (20, 506), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 80, 80), 2)
            cv2.putText(panel, "gray=points  orange=boxes  blue=centroid", (20, 578), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (90, 90, 90), 2)
            cv2.putText(panel, "red=footpoint  green/yellow=track", (20, 614), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (90, 90, 90), 2)
        return panel

    def write(
        self,
        frame_index: int,
        timestamp: float,
        raw_points: np.ndarray,
        points: np.ndarray,
        candidates: list[ClusterCandidate],
        state: Any,
        config: dict[str, Any],
        aruco_prior: ArucoPrior | None,
        aruco_selected_candidate: ClusterCandidate | None,
        aruco_debug: ArucoDebugState,
        aruco_context: Any,
        person_class_id: int,
    ) -> None:
        if not self.enabled or frame_index % self.every_n_frames != 0:
            return

        suffix = self._suffix(frame_index, timestamp)
        raw_bev = render_debug_image(
            points=raw_points,
            candidates=[],
            state=None,
            config=config,
            frame_index=frame_index,
            timestamp=timestamp,
            view="bev",
            title="raw bev",
        )
        filtered_bev = render_debug_image(
            points=points,
            candidates=candidates,
            state=state,
            config=config,
            frame_index=frame_index,
            timestamp=timestamp,
            view="bev",
            title="filtered bev",
        )
        side_view = render_debug_image(
            points=points,
            candidates=candidates,
            state=state,
            config=config,
            frame_index=frame_index,
            timestamp=timestamp,
            view="side",
            title="side view",
        )
        target_crop = render_target_crop_image(
            points=points,
            candidates=candidates,
            state=state,
            config=config,
            frame_index=frame_index,
            timestamp=timestamp,
        )
        front_camera = self._render_front_camera_image(
            frame_index=frame_index,
            lidar_timestamp=timestamp,
            aruco_prior=aruco_prior,
            aruco_selected_candidate=aruco_selected_candidate,
            aruco_debug=aruco_debug,
            aruco_context=aruco_context,
        )
        segmentation_preview = render_segmentation_preview(
            frame=aruco_debug.frame,
            segmentation=aruco_debug.segmentation,
            person_mask=aruco_debug.target_person_mask,
            selected_person_masks=aruco_debug.selected_person_masks,
            raw_person_mask_count=aruco_debug.raw_person_mask_count,
            frame_index=frame_index,
            timestamp=timestamp,
            camera_timestamp=aruco_debug.frame_timestamp,
            sync_dt_ms=aruco_debug.sync_dt_ms,
            person_class_id=person_class_id,
            config=config,
        )
        overview = stack_debug_images(
            [
                self._render_overview_info_panel(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    raw_points=raw_points,
                    points=points,
                    candidates=candidates,
                    state=state,
                    config=config,
                    aruco_prior=aruco_prior,
                    aruco_selected_candidate=aruco_selected_candidate,
                    aruco_debug=aruco_debug,
                ),
                raw_bev,
                filtered_bev,
                side_view,
                target_crop,
                segmentation_preview,
            ],
            config,
        )

        cv2.imwrite(str(self.output_dir / "front_camera" / f"{suffix}.png"), front_camera)
        cv2.imwrite(str(self.output_dir / "segmentation" / f"{suffix}.png"), segmentation_preview)
        cv2.imwrite(str(self.output_dir / "raw_bev" / f"{suffix}.png"), raw_bev)
        cv2.imwrite(str(self.output_dir / "filtered_bev" / f"{suffix}.png"), filtered_bev)
        cv2.imwrite(str(self.output_dir / "side_view" / f"{suffix}.png"), side_view)
        cv2.imwrite(str(self.output_dir / "target_crop" / f"{suffix}.png"), target_crop)
        cv2.imwrite(str(self.output_dir / "overview" / f"{suffix}.png"), overview)
        self._write_pcd(self.output_dir / "raw_pcd" / f"{suffix}.pcd", raw_points)
        self._write_pcd(self.output_dir / "filtered_pcd" / f"{suffix}.pcd", points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track target from lidar points projected into the ArUco target mask.")
    parser.add_argument("--aruco-config", default="configs/aruco_config.yaml", help="Path to ArUco YAML config.")
    parser.add_argument("--lidar-config", default="configs/lidar_config.yaml", help="Path to lidar YAML config.")
    parser.add_argument(
        "--aruco-gating-distance",
        type=float,
        default=1.5,
        help="Maximum distance in meters between ArUco prior and lidar candidate footpoint.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/fusion_tracking_log.json",
        help="Path to fused tracking JSON output.",
    )
    parser.add_argument(
        "--ie-path",
        default=None,
        help="Path to IE/SPAN/GT text file for world-frame lidar pose. If omitted, read from lidar config ie_pose.path.",
    )
    parser.add_argument(
        "--output-target-world-json",
        default="outputs/target_world_positions.jsonl",
        help="Path to per-frame world target position JSONL output.",
    )
    return parser.parse_args()


def select_candidate_with_aruco_prior(
    candidates: list[ClusterCandidate],
    aruco_prior: ArucoPrior | None,
    gating_distance_m: float,
    tracker_state: TrackState | None = None,
    tracker_cfg: dict[str, Any] | None = None,
) -> ClusterCandidate | None:
    if aruco_prior is None or not candidates:
        return None
    tracker_cfg = tracker_cfg or {}
    aruco_sigma_m = max(float(tracker_cfg.get("aruco_gate_sigma_m", gating_distance_m * 0.5)), 1e-3)
    hard_gate_multiplier = max(float(tracker_cfg.get("aruco_hard_gate_multiplier", 1.8)), 1.0)
    min_combined_score = float(tracker_cfg.get("aruco_min_combined_score", 0.08))
    motion_sigma_floor_m = max(float(tracker_cfg.get("aruco_motion_sigma_floor_m", 0.5)), 1e-3)

    best_candidate: ClusterCandidate | None = None
    best_combined_score = -1.0
    for candidate in candidates:
        planar_distance = float(np.linalg.norm(candidate.footpoint[:2] - aruco_prior.position_lidar[:2]))
        if planar_distance > gating_distance_m * hard_gate_multiplier:
            continue
        aruco_likelihood = math.exp(-0.5 * (planar_distance / aruco_sigma_m) ** 2)

        motion_likelihood = 1.0
        if tracker_state is not None:
            expected_position = tracker_state.position
            if float(candidate.footpoint.shape[0]) >= 3 and float(expected_position.shape[0]) >= 3:
                residual = float(np.linalg.norm(candidate.footpoint[:2] - expected_position[:2]))
            else:
                residual = float(np.linalg.norm(candidate.footpoint[:2] - expected_position[:2]))
            motion_sigma = motion_sigma_floor_m
            motion_likelihood = math.exp(-0.5 * (residual / motion_sigma) ** 2)

        # Keep geometry/size score in the decision while prioritizing ArUco consistency.
        combined_score = 0.55 * aruco_likelihood + 0.25 * motion_likelihood + 0.20 * float(candidate.score)
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_candidate = candidate

    if best_candidate is None or best_combined_score < min_combined_score:
        return None
    return best_candidate


def build_target_mask_v2_candidates(
    segmented_clusters: list[np.ndarray],
    person_cfg: dict[str, Any],
) -> list[ClusterCandidate]:
    if not segmented_clusters:
        return []

    relaxed_cfg = dict(person_cfg)
    if "min_points_with_aruco" in person_cfg:
        relaxed_cfg["min_points"] = int(person_cfg["min_points_with_aruco"])
    if "height_m_with_aruco" in person_cfg:
        relaxed_cfg["height_m"] = person_cfg["height_m_with_aruco"]
    if "width_m_with_aruco" in person_cfg:
        relaxed_cfg["width_m"] = person_cfg["width_m_with_aruco"]
    if "depth_m_with_aruco" in person_cfg:
        relaxed_cfg["depth_m"] = person_cfg["depth_m_with_aruco"]

    candidates = filter_candidates(segmented_clusters, relaxed_cfg)
    if candidates:
        return candidates

    largest = max(segmented_clusters, key=lambda cluster: int(cluster.shape[0]))
    min_points = int(relaxed_cfg.get("min_points", 3))
    if int(largest.shape[0]) < max(min_points, 3):
        return []
    return [compute_candidate(cluster_id=0, cluster=largest)]


def candidate_key(candidate: ClusterCandidate) -> str:
    foot = np.round(candidate.footpoint, 3).tolist()
    size = np.round(candidate.size, 3).tolist()
    return f"{foot}|{size}"


def candidate_motion_reject_reason(
    candidate: ClusterCandidate,
    last_state: TrackState | None,
    timestamp: float,
    tracker_cfg: dict[str, Any],
    aruco_prior: ArucoPrior | None = None,
) -> str | None:
    aruco_observed = (
        aruco_prior is not None
        and bool(aruco_prior.visible)
        and str(aruco_prior.source).lower() == "observed"
    )
    relax_motion_with_aruco_observed = bool(tracker_cfg.get("aruco_observed_relax_motion", True))
    distance_to_aruco_m = None
    if aruco_prior is not None:
        distance_to_aruco_m = float(np.linalg.norm(candidate.footpoint[:2] - aruco_prior.position_lidar[:2]))

    max_distance_to_aruco_m = tracker_cfg.get("max_distance_to_aruco_m")
    if max_distance_to_aruco_m is not None and distance_to_aruco_m is not None:
        if distance_to_aruco_m > float(max_distance_to_aruco_m):
            return f"aruco_distance>{float(max_distance_to_aruco_m):.2f}"

    max_footpoint_z_m = float(tracker_cfg.get("max_footpoint_z_m", 0.7))
    near_aruco_distance_m = float(tracker_cfg.get("near_aruco_distance_m", 0.8))
    if distance_to_aruco_m is not None and distance_to_aruco_m <= near_aruco_distance_m:
        max_footpoint_z_m = float(tracker_cfg.get("near_aruco_max_footpoint_z_m", max_footpoint_z_m))
    footpoint_z = float(candidate.footpoint[2])
    if footpoint_z > max_footpoint_z_m:
        return f"footpoint_z>{max_footpoint_z_m:.2f}"

    # With observed ArUco prior, trust ROI+prior association first and
    # do not hard-reject by motion continuity (common during re-acquisition).
    if aruco_observed and relax_motion_with_aruco_observed:
        return None

    if last_state is None:
        return None

    dt = float(timestamp - last_state.timestamp)
    if dt < 0.0:
        return "negative_dt"

    delta = candidate.footpoint - last_state.position
    planar_delta = float(np.linalg.norm(delta[:2]))
    vertical_delta = float(abs(delta[2]))
    max_planar_jump_m = float(tracker_cfg.get("max_planar_jump_m", 0.9))
    max_vertical_jump_m = float(tracker_cfg.get("max_vertical_jump_m", 0.5))
    near_aruco_bypass_jump = bool(tracker_cfg.get("near_aruco_bypass_planar_jump", True))
    if distance_to_aruco_m is not None and distance_to_aruco_m <= near_aruco_distance_m and near_aruco_bypass_jump:
        max_planar_jump_m = float(tracker_cfg.get("near_aruco_max_planar_jump_m", max_planar_jump_m))
    if planar_delta > max_planar_jump_m:
        return f"planar_jump>{max_planar_jump_m:.2f}"
    if vertical_delta > max_vertical_jump_m:
        return f"vertical_jump>{max_vertical_jump_m:.2f}"

    if dt > 1e-3:
        planar_speed = planar_delta / dt
        vertical_speed = vertical_delta / dt
        max_planar_speed_mps = float(tracker_cfg.get("max_planar_speed_mps", 3.0))
        max_vertical_speed_mps = float(tracker_cfg.get("max_vertical_speed_mps", 2.0))
        if distance_to_aruco_m is not None and distance_to_aruco_m <= near_aruco_distance_m and near_aruco_bypass_jump:
            max_planar_speed_mps = float(tracker_cfg.get("near_aruco_max_planar_speed_mps", max_planar_speed_mps))
        if planar_speed > max_planar_speed_mps:
            return f"planar_speed>{max_planar_speed_mps:.2f}"
        if vertical_speed > max_vertical_speed_mps:
            return f"vertical_speed>{max_vertical_speed_mps:.2f}"
    return None


def is_candidate_motion_consistent(
    candidate: ClusterCandidate,
    last_state: TrackState | None,
    timestamp: float,
    tracker_cfg: dict[str, Any],
    aruco_prior: ArucoPrior | None = None,
) -> bool:
    return candidate_motion_reject_reason(
        candidate=candidate,
        last_state=last_state,
        timestamp=timestamp,
        tracker_cfg=tracker_cfg,
        aruco_prior=aruco_prior,
    ) is None


def build_motion_cfg(
    tracker_cfg: dict[str, Any],
    aruco_prior: ArucoPrior | None,
    reconnect_streak: int,
    is_rescue_candidate: bool,
) -> dict[str, Any]:
    cfg = dict(tracker_cfg)
    aruco_observed = (
        aruco_prior is not None
        and bool(aruco_prior.visible)
        and str(aruco_prior.source).lower() == "observed"
    )
    if aruco_prior is not None:
        if "max_footpoint_z_with_aruco_m" in tracker_cfg:
            cfg["max_footpoint_z_m"] = float(tracker_cfg["max_footpoint_z_with_aruco_m"])
        if "max_planar_jump_with_aruco_m" in tracker_cfg:
            cfg["max_planar_jump_m"] = float(tracker_cfg["max_planar_jump_with_aruco_m"])
        if "max_planar_speed_with_aruco_mps" in tracker_cfg:
            cfg["max_planar_speed_mps"] = float(tracker_cfg["max_planar_speed_with_aruco_mps"])

        relax_after = int(tracker_cfg.get("reconnect_relax_after_frames", 2))
        if reconnect_streak >= relax_after:
            cfg["max_planar_jump_m"] = float(tracker_cfg.get("reconnect_max_planar_jump_m", cfg.get("max_planar_jump_m", 1.4)))
            cfg["max_planar_speed_mps"] = float(tracker_cfg.get("reconnect_max_planar_speed_mps", cfg.get("max_planar_speed_mps", 5.0)))
            cfg["max_footpoint_z_m"] = float(tracker_cfg.get("reconnect_max_footpoint_z_m", cfg.get("max_footpoint_z_m", 1.2)))
    if aruco_observed:
        cfg["aruco_observed_relax_motion"] = bool(tracker_cfg.get("aruco_observed_relax_motion", True))

    if is_rescue_candidate and aruco_prior is not None:
        cfg["max_footpoint_z_m"] = float(tracker_cfg.get("rescue_max_footpoint_z_m", cfg.get("max_footpoint_z_m", 1.4)))
        cfg["max_planar_jump_m"] = float(tracker_cfg.get("rescue_max_planar_jump_m", cfg.get("max_planar_jump_m", 2.0)))
        cfg["max_distance_to_aruco_m"] = float(tracker_cfg.get("rescue_max_distance_to_aruco_m", 1.5))
    return cfg


def lidar_point_to_camera(point_lidar: np.ndarray, aruco_context: Any) -> np.ndarray:
    rotation_target_from_camera = np.asarray(aruco_context.rotation_target_from_camera, dtype=np.float64)
    translation_target_from_camera = np.asarray(aruco_context.translation_target_from_camera, dtype=np.float64)
    return rotation_target_from_camera.T @ (np.asarray(point_lidar, dtype=np.float64) - translation_target_from_camera)


def select_points_in_target_mask(
    points: np.ndarray,
    aruco_debug: ArucoDebugState,
    aruco_context: Any,
) -> tuple[np.ndarray, int]:
    if points.size == 0 or aruco_debug.target_person_mask is None:
        return np.empty((0, 3), dtype=np.float64), 0

    mask = aruco_debug.target_person_mask
    h, w = mask.shape[:2]
    selected: list[np.ndarray] = []
    projected_count = 0
    for point in points:
        point_camera = lidar_point_to_camera(point, aruco_context)
        if point_camera[2] <= 0:
            continue
        try:
            pixel = project_camera_point_to_image(
                point_camera,
                aruco_context.camera_matrix,
                aruco_context.dist_coeffs,
                camera_model=aruco_context.camera_model,
            )
        except ValueError:
            continue
        x, y = np.round(pixel).astype(int)
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        projected_count += 1
        if mask[y, x]:
            selected.append(point)

    if not selected:
        return np.empty((0, 3), dtype=np.float64), projected_count
    return np.asarray(selected, dtype=np.float64), projected_count


def apply_aruco_local_roi(
    points: np.ndarray,
    aruco_prior: ArucoPrior | None,
    aruco_target_centers: list[np.ndarray] | None,
    roi_cfg: dict[str, Any],
    reconnect_streak: int = 0,
    force_disable: bool = False,
) -> np.ndarray:
    if points.size == 0:
        return points
    local_cfg = roi_cfg.get("local_with_aruco", {})
    if force_disable or not bool(local_cfg.get("enabled", False)):
        return points

    half_x = float(local_cfg.get("half_width_m", 3.0))
    half_y = float(local_cfg.get("half_height_m", 3.0))
    z_min_offset = float(local_cfg.get("z_min_offset_m", -1.2))
    z_max_offset = float(local_cfg.get("z_max_offset_m", 1.8))
    expand_after = int(local_cfg.get("expand_after_reconnect_frames", 2))
    if reconnect_streak >= expand_after:
        half_x = float(local_cfg.get("expanded_half_width_m", half_x))
        half_y = float(local_cfg.get("expanded_half_height_m", half_y))
        z_min_offset = float(local_cfg.get("expanded_z_min_offset_m", z_min_offset))
        z_max_offset = float(local_cfg.get("expanded_z_max_offset_m", z_max_offset))
    centers: list[np.ndarray] = []
    if aruco_target_centers:
        centers.extend([np.asarray(center, dtype=np.float64).reshape(3) for center in aruco_target_centers])
    if aruco_prior is not None:
        centers.append(np.asarray(aruco_prior.position_lidar, dtype=np.float64).reshape(3))
    if not centers:
        return points

    mask = np.zeros((points.shape[0],), dtype=bool)
    for center in centers:
        center_mask = (
            (points[:, 0] >= center[0] - half_x)
            & (points[:, 0] <= center[0] + half_x)
            & (points[:, 1] >= center[1] - half_y)
            & (points[:, 1] <= center[1] + half_y)
            & (points[:, 2] >= center[2] + z_min_offset)
            & (points[:, 2] <= center[2] + z_max_offset)
        )
        mask |= center_mask
    cropped = points[mask]
    min_points_keep = int(local_cfg.get("min_points_after_crop", 180))
    if cropped.shape[0] < min_points_keep:
        # Avoid over-cropping: keep original points if local ROI becomes too sparse.
        return points
    return cropped


def format_fusion_record(
    timestamp: float,
    point_count_raw: int,
    point_count_after_roi: int,
    point_count_after_ego_filter: int,
    point_count_after_ground_removal: int,
    point_count_after_aruco_local_roi: int,
    point_count_filtered: int,
    candidates: list[ClusterCandidate],
    tracker_candidates: list[ClusterCandidate],
    candidate_diagnostics: list[dict[str, Any]],
    segmentation_cluster_count: int,
    aruco_prior: ArucoPrior | None,
    aruco_selected_candidate: ClusterCandidate | None,
    camera_timestamp: float | None,
    sync_dt_ms: float | None,
    segmentation_point_count: int,
    projected_point_count: int,
    aruco_priors: list[ArucoPrior],
    used_aruco_fallback: bool,
    state: Any,
    multi_tracks: list[MultiTrack],
) -> dict[str, Any]:
    iso_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().isoformat()
    return {
        "timestamp": timestamp,
        "datetime": iso_time,
        "point_count_raw": point_count_raw,
        "point_count_after_roi": point_count_after_roi,
        "point_count_after_ego_filter": point_count_after_ego_filter,
        "point_count_after_ground_removal": point_count_after_ground_removal,
        "point_count_after_aruco_local_roi": point_count_after_aruco_local_roi,
        "point_count_filtered": point_count_filtered,
        "camera_timestamp": camera_timestamp,
        "sync_dt_ms": None if sync_dt_ms is None else round(float(sync_dt_ms), 3),
        "point_count_projected_to_front": projected_point_count,
        "point_count_in_target_mask": segmentation_point_count,
        "segmentation_cluster_count": int(segmentation_cluster_count),
        "candidate_count": len(candidates),
        "candidate_count_tracker_input": len(tracker_candidates),
        "candidates": [format_candidate(candidate) for candidate in candidates],
        "tracker_input_candidates": [format_candidate(candidate) for candidate in tracker_candidates],
        "candidate_diagnostics": candidate_diagnostics,
        "aruco_prior": None
        if aruco_prior is None
        else {
            "timestamp": round(float(aruco_prior.timestamp), 6),
            "visible": aruco_prior.visible,
            "source": aruco_prior.source,
            "position_lidar_m": np.round(aruco_prior.position_lidar, 6).tolist(),
            "candidate_count": len(candidates),
        },
        "aruco_priors": [
            {
                "timestamp": round(float(prior.timestamp), 6),
                "visible": bool(prior.visible),
                "source": str(prior.source),
                "position_lidar_m": np.round(prior.position_lidar, 6).tolist(),
            }
            for prior in aruco_priors
        ],
        "aruco_selected_candidate": None if aruco_selected_candidate is None else format_candidate(aruco_selected_candidate),
        "used_aruco_fallback": bool(used_aruco_fallback),
        "track": None
        if state is None
        else {
            "state": state.source,
            "lifecycle": state.track_lifecycle,
            "quality": round(float(state.track_quality), 6),
            "position_lidar_m": np.round(state.position, 6).tolist(),
            "velocity_lidar_mps": np.round(state.velocity, 6).tolist(),
            "selected_candidate": None if state.candidate is None else format_candidate(state.candidate),
        },
        "multi_tracks": [
            {
                "track_id": int(track.track_id),
                "state": track.state.source,
                "lifecycle": track.state.track_lifecycle,
                "quality": round(float(track.state.track_quality), 6),
                "hits": int(track.hits),
                "misses": int(track.misses),
                "position_lidar_m": np.round(track.state.position, 6).tolist(),
                "velocity_lidar_mps": np.round(track.state.velocity, 6).tolist(),
                "selected_candidate": None if track.state.candidate is None else format_candidate(track.state.candidate),
            }
            for track in multi_tracks
        ],
    }


def main() -> None:
    args = parse_args()
    aruco_config = load_aruco_config(args.aruco_config)
    lidar_config = load_aruco_config(args.lidar_config)
    configure_span_lidar_extrinsics(lidar_config)
    aruco_runtime_config = dict(aruco_config)
    aruco_runtime_config["segmentation"] = dict(lidar_config.get("segmentation", {}))
    aruco_runtime_config["sync"] = dict(lidar_config.get("sync", {}))

    aruco_provider = ArucoPriorProvider(aruco_runtime_config)
    lidar_source = RosbagLidarSource(lidar_config)

    clustering_cfg = lidar_config["clustering"]
    tracker_cfg = lidar_config["tracker"]
    logging_cfg = lidar_config.get("logging", {})

    multi_tracker = MultiObjectTracker(
        gating_distance_m=float(tracker_cfg.get("multi_gating_distance_m", tracker_cfg.get("gating_distance_m", 1.0))),
        max_prediction_duration_s=float(
            tracker_cfg.get("multi_max_prediction_duration_s", tracker_cfg.get("max_prediction_duration_s", 0.5))
        ),
        process_gain=float(tracker_cfg.get("multi_process_gain", tracker_cfg.get("process_gain", 0.6))),
        allow_prediction=bool(tracker_cfg.get("multi_allow_prediction", tracker_cfg.get("allow_prediction", False))),
        confirm_hits=int(tracker_cfg.get("multi_confirm_hits", tracker_cfg.get("confirm_hits", 3))),
        forget_misses=int(tracker_cfg.get("multi_forget_misses", tracker_cfg.get("forget_misses", 8))),
    )
    debug_writer = FusionDebugWriter(
        {
            "debug": lidar_config.get("debug", {}),
            "tracking": aruco_config.get("tracking", {}),
        }
    )
    write_fusion_json = bool(logging_cfg.get("write_fusion_json", True))
    json_writer = None if not write_fusion_json else JsonRecordWriter(
        output_path=Path(str(args.output_json)),
        flush_every_frame=bool(logging_cfg.get("flush_every_frame", True)),
    )
    ie_cfg = dict(lidar_config.get("ie_pose", {}))
    ie_path_value = args.ie_path if args.ie_path is not None else ie_cfg.get(
        "path",
        "data/2026-04-14_mongkok-opensky-walk-006/raw/novatel/ie/ie.txt",
    )
    ie_pose_provider = IePoseProvider(load_ie_poses(Path(str(ie_path_value))))
    write_target_world_json = bool(logging_cfg.get("write_target_world_json", True))
    target_world_writer = TargetWorldWriter(
        Path(str(args.output_target_world_json)),
        flush_every_frame=bool(logging_cfg.get("flush_every_frame", True)),
    )
    enu_origin_pose = ie_pose_provider.poses[0] if ie_pose_provider.available else None
    ecef_origin = None if enu_origin_pose is None else _geodetic_to_ecef(
        enu_origin_pose.latitude_deg,
        enu_origin_pose.longitude_deg,
        enu_origin_pose.height_m,
    )
    enu_from_ecef = None if enu_origin_pose is None else _ecef_to_enu_matrix(
        enu_origin_pose.latitude_deg,
        enu_origin_pose.longitude_deg,
    )

    frame_index = 0
    interrupted = False
    local_roi_recovery_countdown = 0
    previous_track_positions: list[np.ndarray] = []
    segmentation_enabled = bool(lidar_config.get("segmentation", {}).get("enabled", False))

    try:
        while True:
            ok, points, timestamp = lidar_source.read()
            if not ok or points is None or timestamp is None:
                break
            timestamp = float(timestamp)
            raw_count = int(points.shape[0])
            roi_points = crop_points(points, lidar_config.get("roi", {}))
            point_count_after_roi = int(roi_points.shape[0])
            no_ego_points = remove_ego_vehicle_points(roi_points, lidar_config.get("ego_vehicle_filter", {}))
            point_count_after_ego_filter = int(no_ego_points.shape[0])
            aruco_priors = aruco_provider.get_priors(timestamp)
            aruco_prior = None if not aruco_priors else aruco_priors[0]
            aruco_debug = aruco_provider.get_debug_state()
            ground_cfg = dict(lidar_config.get("ground_removal", {}))
            protect_regions = build_ground_protect_regions(previous_track_positions, aruco_priors, ground_cfg)
            if protect_regions:
                ground_cfg["protect_regions"] = protect_regions
            processed = remove_ground(no_ego_points, ground_cfg)
            point_count_after_ground_removal = int(processed.shape[0])
            use_local_roi = local_roi_recovery_countdown <= 0
            aruco_target_centers = [
                np.asarray(target["center_in_target_m"], dtype=np.float64)
                for target in aruco_debug.target_results
                if "center_in_target_m" in target
            ]
            processed = apply_aruco_local_roi(
                processed,
                aruco_prior,
                aruco_target_centers,
                lidar_config.get("roi", {}),
                reconnect_streak=0,
                force_disable=not use_local_roi,
            )
            point_count_after_aruco_local_roi = int(processed.shape[0])

            if segmentation_enabled:
                segmented_points, projected_point_count = select_points_in_target_mask(
                    points=processed,
                    aruco_debug=aruco_debug,
                    aruco_context=aruco_provider.context,
                )
            else:
                segmented_points = processed
                projected_point_count = int(processed.shape[0])
            segmented_clusters = euclidean_clusters(
                points=segmented_points,
                tolerance=float(clustering_cfg["tolerance_m"]),
                min_points=int(clustering_cfg["min_cluster_points"]),
                max_points=int(clustering_cfg["max_cluster_points"]),
            ) if segmented_points.shape[0] > 0 else []
            segmented_clusters = merge_vertical_person_clusters(
                segmented_clusters,
                xy_merge_distance_m=float(clustering_cfg.get("merge_xy_distance_m", 0.55)),
                z_gap_m=float(clustering_cfg.get("merge_z_gap_m", 0.9)),
                max_merged_points=int(clustering_cfg.get("merge_max_points", 8000)),
            ) if segmented_clusters else []
            segmentation_cluster_count = len(segmented_clusters)
            candidates = build_target_mask_v2_candidates(segmented_clusters, lidar_config["person_cluster"])

            aruco_selected_candidate = None
            tracker_candidates = list(candidates)
            tracker_candidate_keys = {candidate_key(candidate) for candidate in tracker_candidates}
            aruco_selected_key = None if aruco_selected_candidate is None else candidate_key(aruco_selected_candidate)
            candidate_diagnostics: list[dict[str, Any]] = []
            for candidate in candidates:
                key = candidate_key(candidate)
                reject_reason = None
                distance_to_aruco_m = None
                if aruco_prior is not None:
                    distance_to_aruco_m = float(np.linalg.norm(candidate.footpoint[:2] - aruco_prior.position_lidar[:2]))
                candidate_diagnostics.append(
                    {
                        "candidate": format_candidate(candidate),
                        "source_target_mask_v2": True,
                        "distance_to_aruco_m": None if distance_to_aruco_m is None else round(distance_to_aruco_m, 6),
                        "is_aruco_selected": key == aruco_selected_key,
                        "is_tracker_input": key in tracker_candidate_keys,
                        "motion_reject_reason": reject_reason,
                    }
                )

            state = None
            used_aruco_fallback = False

            if aruco_priors and int(len(tracker_candidates)) == 0:
                local_cfg = lidar_config.get("roi", {}).get("local_with_aruco", {})
                trigger_after = int(local_cfg.get("recovery_trigger_frames", 2))
                recovery_frames = int(local_cfg.get("recovery_disable_frames", 4))
                if trigger_after <= 0:
                    local_roi_recovery_countdown = max(local_roi_recovery_countdown, recovery_frames)
            if local_roi_recovery_countdown > 0:
                local_roi_recovery_countdown -= 1

            multi_tracks = multi_tracker.update(timestamp=timestamp, candidates=candidates)
            previous_track_positions = [track.state.position.copy() for track in multi_tracks]

            debug_writer.write(
                frame_index=frame_index,
                timestamp=timestamp,
                raw_points=points,
                points=processed,
                candidates=candidates,
                state=state,
                config=lidar_config,
                aruco_prior=aruco_prior,
                aruco_selected_candidate=aruco_selected_candidate,
                aruco_debug=aruco_debug,
                aruco_context=aruco_provider.context,
                person_class_id=aruco_provider.segmentation_predictor.person_class_id,
            )
            record = format_fusion_record(
                timestamp=timestamp,
                point_count_raw=raw_count,
                point_count_after_roi=point_count_after_roi,
                point_count_after_ego_filter=point_count_after_ego_filter,
                point_count_after_ground_removal=point_count_after_ground_removal,
                point_count_after_aruco_local_roi=point_count_after_aruco_local_roi,
                point_count_filtered=int(processed.shape[0]),
                candidates=candidates,
                tracker_candidates=tracker_candidates,
                candidate_diagnostics=candidate_diagnostics,
                segmentation_cluster_count=segmentation_cluster_count,
                aruco_prior=aruco_prior,
                aruco_selected_candidate=aruco_selected_candidate,
                camera_timestamp=aruco_debug.frame_timestamp,
                sync_dt_ms=aruco_debug.sync_dt_ms,
                segmentation_point_count=int(segmented_points.shape[0]),
                projected_point_count=int(projected_point_count),
                aruco_priors=aruco_priors,
                used_aruco_fallback=used_aruco_fallback,
                state=state,
                multi_tracks=multi_tracks,
            )
            if json_writer is not None:
                json_writer.append(record)
            ie_pose = ie_pose_provider.get_interpolated(timestamp)
            target_lidar = None
            if aruco_debug.target_results:
                target_name = [target.get("id") for target in aruco_debug.target_results if "id" in target]
            else:
                target_name = [list(target_group) for target_group in aruco_provider.target_id_groups]
            world_record: dict[str, Any] = {
                "timestamp": float(timestamp),
                "datetime": datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().isoformat(),
                "target_name": target_name,
                "target_lidar_m": None if target_lidar is None else np.round(target_lidar, 6).tolist(),
                "target_world_lla": None,
                "target_world_enu_m": None,
                "multi_targets": [],
                "aruco_group_targets": [],
                "ie_pose": None,
            }
            if ie_pose is not None:
                rot_enu_from_body = rotation_matrix_from_ie(
                    roll_deg=ie_pose.roll_deg,
                    pitch_deg=ie_pose.pitch_deg,
                    heading_deg=ie_pose.heading_deg,
                )
                lidar_ecef = _geodetic_to_ecef(ie_pose.latitude_deg, ie_pose.longitude_deg, ie_pose.height_m)
                enu_to_ecef = _enu_to_ecef_matrix(ie_pose.latitude_deg, ie_pose.longitude_deg)
                world_record["ie_pose"] = {
                    "timestamp": round(float(ie_pose.timestamp), 6),
                    "latitude_deg": round(float(ie_pose.latitude_deg), 9),
                    "longitude_deg": round(float(ie_pose.longitude_deg), 9),
                    "height_m": round(float(ie_pose.height_m), 6),
                    "roll_deg": round(float(ie_pose.roll_deg), 6),
                    "pitch_deg": round(float(ie_pose.pitch_deg), 6),
                    "heading_deg": round(float(ie_pose.heading_deg), 6),
                }
                if ecef_origin is not None and enu_from_ecef is not None:
                    for target in aruco_debug.target_results:
                        if "center_in_target_m" not in target:
                            continue
                        aruco_lidar = np.asarray(target["center_in_target_m"], dtype=np.float64).reshape(3)
                        aruco_body = lidar_to_ie_body(aruco_lidar)
                        aruco_offset_enu = rot_enu_from_body @ aruco_body
                        aruco_ecef = lidar_ecef + (enu_to_ecef @ aruco_offset_enu)
                        aruco_lat, aruco_lon, aruco_h = _ecef_to_geodetic(aruco_ecef)
                        aruco_enu = enu_from_ecef @ (aruco_ecef - ecef_origin)
                        world_record["aruco_group_targets"].append(
                            {
                                "target_group_index": int(target.get("target_group_index", -1)),
                                "target_ids": list(target.get("target_ids", [])),
                                "visible_ids": list(target.get("visible_ids", [])),
                                "target_lidar_m": np.round(aruco_lidar, 6).tolist(),
                                "target_world_lla": [
                                    round(float(aruco_lat), 9),
                                    round(float(aruco_lon), 9),
                                    round(float(aruco_h), 6),
                                ],
                                "target_world_enu_m": np.round(aruco_enu, 6).tolist(),
                            }
                        )
                    for track in multi_tracks:
                        target_body_multi = lidar_to_ie_body(track.state.position)
                        target_offset_enu_multi = rot_enu_from_body @ target_body_multi
                        target_ecef_multi = lidar_ecef + (enu_to_ecef @ target_offset_enu_multi)
                        target_lat_multi, target_lon_multi, target_h_multi = _ecef_to_geodetic(target_ecef_multi)
                        target_enu_multi = enu_from_ecef @ (target_ecef_multi - ecef_origin)
                        world_record["multi_targets"].append(
                            {
                                "track_id": int(track.track_id),
                                "state": track.state.source,
                                "lifecycle": track.state.track_lifecycle,
                                "quality": round(float(track.state.track_quality), 6),
                                "hits": int(track.hits),
                                "misses": int(track.misses),
                                "target_lidar_m": np.round(track.state.position, 6).tolist(),
                                "target_world_lla": [
                                    round(float(target_lat_multi), 9),
                                    round(float(target_lon_multi), 9),
                                    round(float(target_h_multi), 6),
                                ],
                                "target_world_enu_m": np.round(target_enu_multi, 6).tolist(),
                            }
                        )
                    # Backward-compatible single-target mirrors use first multi target when available.
                    if world_record["multi_targets"]:
                        first_target = world_record["multi_targets"][0]
                        world_record["target_lidar_m"] = first_target["target_lidar_m"]
                        world_record["target_world_lla"] = first_target["target_world_lla"]
                        world_record["target_world_enu_m"] = first_target["target_world_enu_m"]
                    elif world_record["aruco_group_targets"]:
                        first_group_target = world_record["aruco_group_targets"][0]
                        world_record["target_lidar_m"] = first_group_target["target_lidar_m"]
                        world_record["target_world_lla"] = first_group_target["target_world_lla"]
                        world_record["target_world_enu_m"] = first_group_target["target_world_enu_m"]
            if write_target_world_json:
                target_world_writer.append(world_record)

            if bool(logging_cfg.get("print_summary_every_frame", False)):
                aruco_text = "none"
                if aruco_prior is not None:
                    aruco_text = (
                        f"{aruco_prior.source} pos={np.round(aruco_prior.position_lidar, 3).tolist()} "
                        f"mask_points={segmented_points.shape[0]} "
                        f"sync_dt_ms={aruco_debug.sync_dt_ms if aruco_debug.sync_dt_ms is not None else 'none'}"
                    )
                fused_text = "none"
                selected_text = (
                    "none"
                    if aruco_selected_candidate is None
                    else np.round(aruco_selected_candidate.footpoint, 3).tolist()
                )
                fallback_text = "true" if used_aruco_fallback else "false"
                print(
                    f"time={timestamp:.3f} raw={raw_count} filtered={processed.shape[0]} "
                    f"candidates={len(candidates)} aruco={aruco_text} aruco_match={selected_text} "
                    f"aruco_fallback={fallback_text} fused={fused_text} multi_tracks={len(multi_tracks)}"
                )
            else:
                print(json.dumps(record, ensure_ascii=False))

            frame_index += 1

    except KeyboardInterrupt:
        interrupted = True
    finally:
        lidar_source.release()
        aruco_provider.release()
        if json_writer is not None:
            json_writer.flush()
        if write_target_world_json:
            target_world_writer.flush()

    if interrupted:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
