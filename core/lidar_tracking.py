from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional runtime acceleration
    cKDTree = None

from core.geometry import project_camera_point_to_image
from core.segmentation import INSTANCE_PALETTE, SegmentationPredictor, extract_person_mask, extract_person_masks, select_relevant_person_masks, visualize_instance, visualize_panoptic, visualize_semantic


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class ClusterCandidate:
    cluster_id: int
    point_count: int
    centroid: np.ndarray
    footpoint: np.ndarray
    size: np.ndarray
    min_bound: np.ndarray
    max_bound: np.ndarray
    score: float


@dataclass
class TrackState:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    source: str
    candidate: ClusterCandidate | None
    track_quality: float = 0.0
    track_lifecycle: str = "tentative"


@dataclass
class CameraProjectionContext:
    camera_model: str
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rotation_target_from_camera: np.ndarray
    translation_target_from_camera: np.ndarray


class RosbagLidarSource:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import rosbag  # type: ignore
        except ImportError as exc:
            raise RuntimeError("rosbag support requires ROS1 Python environment with rosbag installed") from exc

        self.rosbag = rosbag
        rosbag_cfg = config["input"]["rosbag"]
        self.bag_path = str(rosbag_cfg["bag_path"])
        self.topic = str(rosbag_cfg["topic"])
        self.start_time_s = rosbag_cfg.get("start_time_s")
        self.end_time_s = rosbag_cfg.get("end_time_s")
        self.loop = bool(rosbag_cfg.get("loop", False))
        self._bag = None
        self._iterator = None
        self._open_bag()

    def _make_ros_time(self, value: float | None) -> Any:
        if value is None:
            return None

        seconds = float(value)

        try:
            import genpy  # type: ignore

            return genpy.Time.from_sec(seconds)
        except ImportError:
            pass

        try:
            import rospy  # type: ignore

            return rospy.Time.from_sec(seconds)
        except ImportError:
            pass

        rosbag_time = getattr(self.rosbag, "Time", None)
        if rosbag_time is not None:
            return rosbag_time.from_sec(seconds)

        raise RuntimeError("Failed to construct ROS time; neither genpy.Time nor rospy.Time is available")

    def _open_bag(self) -> None:
        self._bag = self.rosbag.Bag(self.bag_path, "r")
        start_time = self._make_ros_time(self.start_time_s)
        end_time = self._make_ros_time(self.end_time_s)
        self._iterator = self._bag.read_messages(topics=[self.topic], start_time=start_time, end_time=end_time)

    def _livox_msg_to_points(self, msg: Any) -> np.ndarray:
        if getattr(msg, "_type", "") != "livox_ros_driver2/CustomMsg":
            raise RuntimeError(f"Unsupported lidar message type: {getattr(msg, '_type', '')}")

        points = np.empty((len(msg.points), 3), dtype=np.float64)
        for index, point in enumerate(msg.points):
            points[index, 0] = float(point.x)
            points[index, 1] = float(point.y)
            points[index, 2] = float(point.z)
        return points

    def read(self) -> tuple[bool, np.ndarray | None, float | None]:
        while True:
            try:
                _, msg, stamp = next(self._iterator)
            except StopIteration:
                if not self.loop:
                    return False, None, None
                self.release()
                self._open_bag()
                continue

            points = self._livox_msg_to_points(msg)
            return True, points, stamp.to_sec()

    def release(self) -> None:
        if self._bag is not None:
            self._bag.close()
            self._bag = None
            self._iterator = None


class RecentFrontCameraReader:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import rosbag  # type: ignore
        except ImportError as exc:
            raise RuntimeError("rosbag support requires ROS1 Python environment with rosbag installed") from exc

        self.rosbag = rosbag
        rosbag_cfg = config["input"]["rosbag"]
        debug_cfg = config.get("debug", {})
        self.bag_path = str(rosbag_cfg["bag_path"])
        self.topic = str(debug_cfg.get("front_camera_topic", "/front_camera/image/compressed"))
        self.start_time_s = rosbag_cfg.get("start_time_s")
        self.end_time_s = rosbag_cfg.get("end_time_s")
        self.loop = bool(rosbag_cfg.get("loop", False))
        self._bag = None
        self._iterator = None
        self._pending: tuple[np.ndarray, float] | None = None
        self._latest: tuple[np.ndarray, float] | None = None
        sync_cfg = config.get("sync", {})
        self._sync_max_dt_s = float(sync_cfg.get("camera_sync_max_dt_s", debug_cfg.get("camera_sync_max_dt_s", 0.08)))
        self._open_bag()

    def _make_ros_time(self, value: float | None) -> Any:
        if value is None:
            return None

        seconds = float(value)

        try:
            import genpy  # type: ignore

            return genpy.Time.from_sec(seconds)
        except ImportError:
            pass

        try:
            import rospy  # type: ignore

            return rospy.Time.from_sec(seconds)
        except ImportError:
            pass

        rosbag_time = getattr(self.rosbag, "Time", None)
        if rosbag_time is not None:
            return rosbag_time.from_sec(seconds)

        raise RuntimeError("Failed to construct ROS time; neither genpy.Time nor rospy.Time is available")

    def _open_bag(self) -> None:
        self._bag = self.rosbag.Bag(self.bag_path, "r")
        start_time = self._make_ros_time(self.start_time_s)
        end_time = self._make_ros_time(self.end_time_s)
        self._iterator = self._bag.read_messages(topics=[self.topic], start_time=start_time, end_time=end_time)

    def _decode_image_msg(self, msg: Any) -> np.ndarray:
        msg_type = getattr(msg, "_type", "")
        if msg_type == "sensor_msgs/CompressedImage":
            encoded = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("Failed to decode sensor_msgs/CompressedImage")
            return frame

        if msg_type != "sensor_msgs/Image":
            raise RuntimeError(f"Unsupported rosbag image message type: {msg_type}")

        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding).lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding == "mono8":
            gray = data.reshape(height, width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if encoding in {"bgr8", "rgb8"}:
            frame = data.reshape(height, width, 3)
            if encoding == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame.copy()

        raise RuntimeError(f"Unsupported sensor_msgs/Image encoding: {msg.encoding}")

    def _read_next(self) -> tuple[np.ndarray | None, float | None]:
        if self._pending is not None:
            frame, timestamp = self._pending
            self._pending = None
            return frame, timestamp

        while True:
            try:
                _, msg, stamp = next(self._iterator)
            except StopIteration:
                if not self.loop:
                    return None, None
                self.release()
                self._open_bag()
                continue
            return self._decode_image_msg(msg), float(stamp.to_sec())

    def get_latest_before(self, timestamp: float) -> np.ndarray | None:
        sample = self.get_nearest_sample(timestamp)
        if sample is None:
            return None
        return sample[0]

    def get_latest_sample_before(self, timestamp: float) -> tuple[np.ndarray, float] | None:
        while True:
            frame, frame_timestamp = self._read_next()
            if frame is None or frame_timestamp is None:
                break
            if frame_timestamp > timestamp:
                self._pending = (frame, frame_timestamp)
                break
            self._latest = (frame, frame_timestamp)

        if self._latest is None:
            return None
        return self._latest[0].copy(), float(self._latest[1])

    def get_nearest_sample(self, timestamp: float) -> tuple[np.ndarray, float] | None:
        previous = self.get_latest_sample_before(timestamp)
        next_sample = None
        if self._pending is not None:
            next_sample = (self._pending[0].copy(), float(self._pending[1]))

        best = None
        best_dt = None
        for sample in (previous, next_sample):
            if sample is None:
                continue
            _, sample_ts = sample
            dt = abs(float(sample_ts - timestamp))
            if best is None or dt < best_dt:
                best = sample
                best_dt = dt

        if best is None:
            return None
        if best_dt is not None and best_dt > self._sync_max_dt_s:
            return None
        return best[0].copy(), float(best[1])

    def release(self) -> None:
        if self._bag is not None:
            self._bag.close()
            self._bag = None
            self._iterator = None
            self._pending = None


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_camera_projection_context(config_path: str | Path) -> CameraProjectionContext:
    config = load_config(config_path)
    intrinsics_cfg = config["camera"]["intrinsics"]
    target_frame_cfg = config.get("target_frame", {})
    camera_model = str(intrinsics_cfg.get("camera_model", "pinhole")).lower()
    if camera_model in {"equidistantcamera", "fisheye"}:
        camera_model = "equidistant"
    elif camera_model not in {"pinhole", "equidistant"}:
        raise ValueError(f"Unsupported camera model: {intrinsics_cfg.get('camera_model')}")

    return CameraProjectionContext(
        camera_model=camera_model,
        camera_matrix=np.asarray(intrinsics_cfg["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.asarray(intrinsics_cfg["dist_coeffs"], dtype=np.float64),
        rotation_target_from_camera=np.asarray(target_frame_cfg.get("rotation_matrix", np.eye(3)), dtype=np.float64),
        translation_target_from_camera=np.asarray(target_frame_cfg.get("translation_m", np.zeros(3)), dtype=np.float64),
    )


def overlay_person_mask(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame
    vis = frame.copy()
    overlay = vis.copy()
    overlay[mask] = np.array([0, 255, 255], dtype=np.uint8)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    return vis


def render_segmentation_preview(
    frame: np.ndarray | None,
    segmentation: Any,
    person_mask: np.ndarray | None,
    selected_person_masks: list[np.ndarray] | None,
    raw_person_mask_count: int,
    frame_index: int,
    timestamp: float,
    camera_timestamp: float | None,
    sync_dt_ms: float | None,
    person_class_id: int,
    config: dict[str, Any],
) -> np.ndarray:
    debug_cfg = config.get("debug", {})
    canvas_width = int(debug_cfg.get("image_width", 1200))
    canvas_height = int(debug_cfg.get("image_height", 900))
    if frame is None:
        canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "segmentation unavailable", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 2)
        cv2.putText(canvas, f"frame={frame_index} ts={timestamp:.3f}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 60, 60), 2)
        return canvas

    canvas = cv2.resize(frame, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)
    if selected_person_masks:
        for group_index, mask in enumerate(selected_person_masks):
            mask_resized = cv2.resize(mask.astype(np.uint8), (canvas_width, canvas_height), interpolation=cv2.INTER_NEAREST) > 0
            color = tuple(int(v) for v in INSTANCE_PALETTE[group_index % len(INSTANCE_PALETTE)])
            overlay = canvas.copy()
            overlay[mask_resized] = color
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
            contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2)
    elif segmentation is not None:
        if isinstance(segmentation, dict):
            if "panoptic_seg" in segmentation:
                canvas = cv2.resize(
                    visualize_panoptic(frame, segmentation, person_class_id),
                    (canvas_width, canvas_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            elif "pred_masks" in segmentation:
                canvas = cv2.resize(
                    visualize_instance(frame, segmentation, person_class_id),
                    (canvas_width, canvas_height),
                    interpolation=cv2.INTER_LINEAR,
                )
        else:
            canvas = cv2.resize(
                visualize_semantic(frame, np.asarray(segmentation, dtype=np.int32), person_class_id),
                (canvas_width, canvas_height),
                interpolation=cv2.INTER_LINEAR,
            )
    elif person_mask is not None:
        mask_resized = cv2.resize(person_mask.astype(np.uint8), (canvas_width, canvas_height), interpolation=cv2.INTER_NEAREST) > 0
        overlay = canvas.copy()
        overlay[mask_resized] = np.array([0, 0, 255], dtype=np.uint8)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    band = canvas.copy()
    cv2.rectangle(band, (0, 0), (canvas_width, 128), (0, 0, 0), -1)
    cv2.addWeighted(band, 0.35, canvas, 0.65, 0, canvas)
    cv2.putText(canvas, "segmentation", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cam_text = "none" if camera_timestamp is None else f"{camera_timestamp:.3f}"
    dt_text = "none" if sync_dt_ms is None else f"{sync_dt_ms:.1f}"
    cv2.putText(canvas, f"frame={frame_index} lidar_ts={timestamp:.3f} cam_ts={cam_text}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(canvas, f"person_class_id={person_class_id} sync_dt_ms={dt_text}", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(
        canvas,
        f"raw_masks={raw_person_mask_count} selected_masks={0 if selected_person_masks is None else len(selected_person_masks)}",
        (20, 142),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return canvas


def lidar_point_to_camera(point_lidar: np.ndarray, camera_context: CameraProjectionContext) -> np.ndarray:
    return camera_context.rotation_target_from_camera.T @ (
        np.asarray(point_lidar, dtype=np.float64) - camera_context.translation_target_from_camera
    )


def select_points_in_person_mask(
    points: np.ndarray,
    person_mask: np.ndarray | None,
    camera_context: CameraProjectionContext,
) -> tuple[np.ndarray, int]:
    if points.size == 0 or person_mask is None:
        return np.empty((0, 3), dtype=np.float64), 0

    h, w = person_mask.shape[:2]
    selected: list[np.ndarray] = []
    projected_count = 0
    for point in points:
        point_camera = lidar_point_to_camera(point, camera_context)
        if point_camera[2] <= 0:
            continue
        try:
            pixel = project_camera_point_to_image(
                point_camera,
                camera_context.camera_matrix,
                camera_context.dist_coeffs,
                camera_model=camera_context.camera_model,
            )
        except ValueError:
            continue
        x, y = np.round(pixel).astype(int)
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        projected_count += 1
        if person_mask[y, x]:
            selected.append(point)

    if not selected:
        return np.empty((0, 3), dtype=np.float64), projected_count
    return np.asarray(selected, dtype=np.float64), projected_count


def crop_points(points: np.ndarray, roi_cfg: dict[str, Any]) -> np.ndarray:
    mask = np.ones(points.shape[0], dtype=bool)
    roi_shape = str(roi_cfg.get("shape", "box")).lower()

    if roi_shape == "circle":
        center_xy = np.asarray(roi_cfg.get("center_xy_m", [0.0, 0.0]), dtype=np.float64).reshape(2)
        min_radius_m = float(roi_cfg.get("min_radius_m", 0.0))
        max_radius_m = float(roi_cfg["radius_m"])
        deltas_xy = points[:, :2] - center_xy
        radii = np.sqrt(np.sum(deltas_xy * deltas_xy, axis=1))
        mask &= radii >= min_radius_m
        mask &= radii <= max_radius_m
    elif roi_shape != "box":
        raise ValueError(f"Unsupported roi.shape: {roi_shape}")

    for axis, axis_index in (("x", 0), ("y", 1), ("z", 2)):
        axis_cfg = roi_cfg.get(axis)
        if axis_cfg is None:
            continue
        mask &= points[:, axis_index] >= float(axis_cfg[0])
        mask &= points[:, axis_index] <= float(axis_cfg[1])
    return points[mask]


def remove_ego_vehicle_points(points: np.ndarray, ego_cfg: dict[str, Any]) -> np.ndarray:
    if not ego_cfg.get("enabled", False):
        return points

    mask = np.ones(points.shape[0], dtype=bool)
    x_min, x_max = ego_cfg["x"]
    y_min, y_max = ego_cfg["y"]
    z_min, z_max = ego_cfg["z"]

    in_box = (
        (points[:, 0] >= float(x_min))
        & (points[:, 0] <= float(x_max))
        & (points[:, 1] >= float(y_min))
        & (points[:, 1] <= float(y_max))
        & (points[:, 2] >= float(z_min))
        & (points[:, 2] <= float(z_max))
    )
    mask &= ~in_box
    return points[mask]


def _fit_plane_least_squares(xyz: np.ndarray) -> tuple[float, float, float] | None:
    """
    Fit plane z = ax + by + c
    Returns (a, b, c), or None if not enough points.
    """
    if xyz.shape[0] < 3:
        return None

    A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(xyz.shape[0])]
    b = xyz[:, 2]
    try:
        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return float(coef[0]), float(coef[1]), float(coef[2])
    except np.linalg.LinAlgError:
        return None


def _points_in_protect_regions(
    points_xyz: np.ndarray,
    protect_regions: list[dict[str, Any]],
    default_radius_m: float,
    default_z_margin_m: float,
) -> np.ndarray:
    if points_xyz.size == 0 or not protect_regions:
        return np.zeros(points_xyz.shape[0], dtype=bool)

    mask = np.zeros(points_xyz.shape[0], dtype=bool)
    for region in protect_regions:
        if not isinstance(region, dict):
            continue
        if "x" not in region or "y" not in region:
            continue
        cx = float(region["x"])
        cy = float(region["y"])
        radius = max(float(region.get("radius_m", default_radius_m)), 1e-3)
        dx = points_xyz[:, 0] - cx
        dy = points_xyz[:, 1] - cy
        in_xy = (dx * dx + dy * dy) <= radius * radius

        if "z" in region:
            cz = float(region["z"])
            z_margin = max(float(region.get("z_margin_m", default_z_margin_m)), 0.0)
            in_z = points_xyz[:, 2] <= (cz + z_margin)
            in_xy &= in_z

        mask |= in_xy
    return mask


def remove_ground(points: np.ndarray, ground_cfg: dict[str, Any]) -> np.ndarray:
    if points.size == 0:
        return points

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {points.shape}")

    method = str(ground_cfg.get("method", "z_threshold")).lower()
    z_min = float(ground_cfg.get("z_min", -math.inf))
    z_max = float(ground_cfg.get("z_max", math.inf))
    z = points[:, 2]
    valid_z = (z >= z_min) & (z <= z_max)
    points = points[valid_z]
    if points.size == 0:
        return points

    if method == "z_threshold":
        # Backward-compatible behavior: keep points inside configured z range.
        return points

    if method not in {"adaptive_grid", "adaptive_plane"}:
        raise ValueError(f"Unsupported ground removal method: {method}")

    cell_size_m = max(float(ground_cfg.get("cell_size_m", 0.5)), 1e-3)
    ground_quantile = float(np.clip(ground_cfg.get("ground_quantile", 0.15), 0.0, 1.0))
    clearance_m = float(ground_cfg.get("clearance_m", 0.12))
    min_points_per_cell = max(int(ground_cfg.get("min_points_per_cell", 4)), 1)
    fallback_clearance_m = float(ground_cfg.get("fallback_clearance_m", 0.18))
    neighbor_radius = max(int(ground_cfg.get("neighbor_radius", 1)), 0)
    protect_regions = list(ground_cfg.get("protect_regions", []))
    protect_radius_m = float(ground_cfg.get("protect_radius_m", 0.35))
    protect_z_margin_m = float(ground_cfg.get("protect_z_margin_m", 0.7))
    protect_mask = _points_in_protect_regions(points[:, :3], protect_regions, protect_radius_m, protect_z_margin_m)

    grid_xy = np.floor(points[:, :2] / cell_size_m).astype(np.int32)

    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for idx, (gx, gy) in enumerate(grid_xy):
        key = (int(gx), int(gy))
        cell_to_indices.setdefault(key, []).append(idx)

    # Step 1: estimate one low representative point per cell
    low_repr_map: dict[tuple[int, int], np.ndarray] = {}
    ground_z_map: dict[tuple[int, int], float] = {}
    clearance_map: dict[tuple[int, int], float] = {}

    for key, indices_list in cell_to_indices.items():
        indices = np.asarray(indices_list, dtype=np.int32)
        cell_pts = points[indices]
        cell_z = cell_pts[:, 2]

        if cell_pts.shape[0] < min_points_per_cell:
            local_ground = float(np.min(cell_z))
            local_clearance = fallback_clearance_m
        else:
            local_ground = float(np.quantile(cell_z, ground_quantile))
            local_clearance = clearance_m

        # representative low point near the estimated ground
        target_idx = int(np.argmin(np.abs(cell_z - local_ground)))
        low_repr_map[key] = cell_pts[target_idx, :3]
        ground_z_map[key] = local_ground
        clearance_map[key] = local_clearance

    keep = np.ones(points.shape[0], dtype=bool)

    # Simple adaptive_grid fallback
    if method == "adaptive_grid":
        for key, indices_list in cell_to_indices.items():
            indices = np.asarray(indices_list, dtype=np.int32)
            cell_z = points[indices, 2]
            local_ground = ground_z_map[key]
            local_clearance = clearance_map[key]
            ground_mask = cell_z <= (local_ground + local_clearance)
            if protect_regions:
                ground_mask &= ~protect_mask[indices]
            keep[indices[ground_mask]] = False
        return points[keep]

    # Step 2: adaptive_plane
    for key, indices_list in cell_to_indices.items():
        gx, gy = key
        neigh_pts = []

        for dx in range(-neighbor_radius, neighbor_radius + 1):
            for dy in range(-neighbor_radius, neighbor_radius + 1):
                nkey = (gx + dx, gy + dy)
                if nkey in low_repr_map:
                    neigh_pts.append(low_repr_map[nkey])

        indices = np.asarray(indices_list, dtype=np.int32)
        cell_pts = points[indices, :3]

        if len(neigh_pts) < 3:
            local_ground = ground_z_map[key]
            local_clearance = clearance_map[key]
            ground_mask = cell_pts[:, 2] <= (local_ground + local_clearance)
            if protect_regions:
                ground_mask &= ~protect_mask[indices]
            keep[indices[ground_mask]] = False
            continue

        neigh_pts_arr = np.asarray(neigh_pts, dtype=np.float64)
        plane = _fit_plane_least_squares(neigh_pts_arr)

        if plane is None:
            local_ground = ground_z_map[key]
            local_clearance = clearance_map[key]
            ground_mask = cell_pts[:, 2] <= (local_ground + local_clearance)
            if protect_regions:
                ground_mask &= ~protect_mask[indices]
            keep[indices[ground_mask]] = False
            continue

        a, b, c = plane
        z_pred = a * cell_pts[:, 0] + b * cell_pts[:, 1] + c
        dist = cell_pts[:, 2] - z_pred

        local_clearance = clearance_map[key]
        ground_mask = dist <= local_clearance
        if protect_regions:
            ground_mask &= ~protect_mask[indices]
        keep[indices[ground_mask]] = False

    return points[keep]


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    deltas = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def _cluster_points_ckdtree(
    points: np.ndarray,
    tolerance: float,
    min_points: int,
    max_points: int,
) -> list[np.ndarray]:
    num_points = int(points.shape[0])
    tree = cKDTree(points[:, :3])
    pair_indices = tree.query_pairs(float(tolerance), output_type="ndarray")

    parents = np.arange(num_points, dtype=np.int32)
    ranks = np.zeros(num_points, dtype=np.uint8)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if ranks[root_a] < ranks[root_b]:
            parents[root_a] = root_b
            return
        if ranks[root_a] > ranks[root_b]:
            parents[root_b] = root_a
            return
        parents[root_b] = root_a
        ranks[root_a] += 1

    for a, b in pair_indices:
        union(int(a), int(b))

    root_to_indices: dict[int, list[int]] = {}
    for index in range(num_points):
        root = find(index)
        root_to_indices.setdefault(root, []).append(index)

    ordered_clusters = sorted(root_to_indices.values(), key=lambda indices: indices[0])
    clusters: list[np.ndarray] = []
    for member_indices in ordered_clusters:
        if min_points <= len(member_indices) <= max_points:
            clusters.append(points[np.asarray(member_indices, dtype=np.int32)])
    return clusters


def _cluster_points_spatial_hash(
    points: np.ndarray,
    tolerance: float,
    min_points: int,
    max_points: int,
) -> list[np.ndarray]:
    num_points = int(points.shape[0])
    visited = np.zeros(num_points, dtype=bool)
    tolerance_sq = float(tolerance * tolerance)

    # Spatial hash: map each point to a cubic cell to avoid O(N^2) pair checks.
    inv_cell_size = 1.0 / float(tolerance)
    grid = np.floor(points[:, :3] * inv_cell_size).astype(np.int32)
    cell_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for idx, cell in enumerate(grid):
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        cell_to_indices.setdefault(key, []).append(idx)

    clusters: list[np.ndarray] = []
    for start_index in range(num_points):
        if visited[start_index]:
            continue

        queue = [start_index]
        visited[start_index] = True
        member_indices: list[int] = []
        queue_index = 0

        while queue_index < len(queue):
            current = queue[queue_index]
            queue_index += 1
            member_indices.append(current)

            cx, cy, cz = (int(grid[current, 0]), int(grid[current, 1]), int(grid[current, 2]))
            current_point = points[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nkey = (cx + dx, cy + dy, cz + dz)
                        neighbor_indices = cell_to_indices.get(nkey)
                        if neighbor_indices is None:
                            continue
                        for neighbor in neighbor_indices:
                            if visited[neighbor]:
                                continue
                            diff = points[neighbor] - current_point
                            if float(np.dot(diff, diff)) > tolerance_sq:
                                continue
                            visited[neighbor] = True
                            queue.append(neighbor)

        if min_points <= len(member_indices) <= max_points:
            clusters.append(points[np.asarray(member_indices, dtype=np.int32)])

    return clusters


def euclidean_clusters(points: np.ndarray, tolerance: float, min_points: int, max_points: int) -> list[np.ndarray]:
    if points.size == 0:
        return []
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if cKDTree is not None and int(points.shape[0]) > 1:
        return _cluster_points_ckdtree(points, tolerance, min_points, max_points)
    return _cluster_points_spatial_hash(points, tolerance, min_points, max_points)


def merge_vertical_person_clusters(
    clusters: list[np.ndarray],
    xy_merge_distance_m: float = 0.55,
    z_gap_m: float = 0.9,
    max_merged_points: int = 8000,
) -> list[np.ndarray]:
    if len(clusters) <= 1:
        return clusters

    merged = [cluster.copy() for cluster in clusters]
    changed = True
    while changed:
        changed = False
        next_clusters: list[np.ndarray] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            base = merged[i]
            base_min = np.min(base, axis=0)
            base_max = np.max(base, axis=0)
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                other = merged[j]
                other_min = np.min(other, axis=0)
                other_max = np.max(other, axis=0)
                base_xy = 0.5 * (base_min[:2] + base_max[:2])
                other_xy = 0.5 * (other_min[:2] + other_max[:2])
                xy_distance = float(np.linalg.norm(base_xy - other_xy))
                if xy_distance > xy_merge_distance_m:
                    continue
                # Positive gap means disjoint vertical stacks; negative means overlap.
                z_gap = max(float(other_min[2] - base_max[2]), float(base_min[2] - other_max[2]), 0.0)
                if z_gap > z_gap_m:
                    continue
                candidate = np.concatenate([base, other], axis=0)
                if candidate.shape[0] > int(max_merged_points):
                    continue
                base = candidate
                base_min = np.min(base, axis=0)
                base_max = np.max(base, axis=0)
                used[j] = True
                changed = True
            used[i] = True
            next_clusters.append(base)
        merged = next_clusters
    return merged


def compute_candidate(cluster_id: int, cluster: np.ndarray) -> ClusterCandidate:
    min_bound = np.min(cluster, axis=0)
    max_bound = np.max(cluster, axis=0)
    size = max_bound - min_bound
    centroid = np.mean(cluster, axis=0)

    z_min = float(min_bound[2])
    z_max = float(max_bound[2])
    foot_band_max = z_min + max(0.15, 0.2 * (z_max - z_min))
    foot_points = cluster[cluster[:, 2] <= foot_band_max]
    if foot_points.shape[0] == 0:
        foot_points = cluster

    footpoint_xy = np.median(foot_points[:, :2], axis=0)
    footpoint_z = np.min(foot_points[:, 2])
    footpoint = np.array([footpoint_xy[0], footpoint_xy[1], footpoint_z], dtype=np.float64)

    return ClusterCandidate(
        cluster_id=cluster_id,
        point_count=int(cluster.shape[0]),
        centroid=centroid,
        footpoint=footpoint,
        size=size,
        min_bound=min_bound,
        max_bound=max_bound,
        score=0.0,
    )


def score_candidate(candidate: ClusterCandidate, person_cfg: dict[str, Any]) -> float:
    height = float(candidate.size[2])
    width = float(max(candidate.size[0], candidate.size[1]))
    depth = float(min(candidate.size[0], candidate.size[1]))

    height_center = float(person_cfg.get("height_preferred_m", 1.7))
    width_center = float(person_cfg.get("width_preferred_m", 0.6))
    depth_center = float(person_cfg.get("depth_preferred_m", 0.5))
    point_count_center = float(person_cfg.get("point_count_preferred", 120))

    point_count_score = 1.0 / (1.0 + abs(candidate.point_count - point_count_center) / max(point_count_center, 1.0))
    height_score = 1.0 / (1.0 + abs(height - height_center))
    width_score = 1.0 / (1.0 + abs(width - width_center))
    depth_score = 1.0 / (1.0 + abs(depth - depth_center))
    verticality_score = min(height / max(width, 1e-3), 4.0) / 4.0

    return float(0.4 * height_score + 0.2 * width_score + 0.1 * depth_score + 0.1 * point_count_score + 0.2 * verticality_score)


def filter_candidates(clusters: list[np.ndarray], person_cfg: dict[str, Any]) -> list[ClusterCandidate]:
    filtered: list[ClusterCandidate] = []
    min_height, max_height = person_cfg["height_m"]
    min_width, max_width = person_cfg["width_m"]
    min_depth, max_depth = person_cfg["depth_m"]
    min_points = int(person_cfg["min_points"])

    for cluster_id, cluster in enumerate(clusters):
        candidate = compute_candidate(cluster_id=cluster_id, cluster=cluster)
        height = float(candidate.size[2])
        width = float(max(candidate.size[0], candidate.size[1]))
        depth = float(min(candidate.size[0], candidate.size[1]))

        if candidate.point_count < min_points:
            continue
        if not (float(min_height) <= height <= float(max_height)):
            continue
        if not (float(min_width) <= width <= float(max_width)):
            continue
        if not (float(min_depth) <= depth <= float(max_depth)):
            continue

        candidate.score = score_candidate(candidate, person_cfg)
        filtered.append(candidate)

    return filtered


class SimpleTracker:
    def __init__(
        self,
        gating_distance_m: float,
        max_prediction_duration_s: float,
        process_gain: float,
        allow_prediction: bool = False,
        confirm_hits: int = 3,
        forget_misses: int = 8,
        quality_hit_gain: float = 0.2,
        quality_miss_decay: float = 0.85,
    ) -> None:
        self.gating_distance_m = float(gating_distance_m)
        self.max_prediction_duration_s = float(max_prediction_duration_s)
        self.process_gain = float(process_gain)
        self.allow_prediction = bool(allow_prediction)
        self.confirm_hits = max(int(confirm_hits), 1)
        self.forget_misses = max(int(forget_misses), 1)
        self.quality_hit_gain = float(np.clip(quality_hit_gain, 1e-3, 1.0))
        self.quality_miss_decay = float(np.clip(quality_miss_decay, 0.1, 0.999))
        self.last_state: TrackState | None = None
        self.last_observed_timestamp: float | None = None
        self.track_quality: float = 0.0
        self.track_lifecycle: str = "tentative"
        self.consecutive_hits: int = 0
        self.consecutive_misses: int = 0

    def _on_hit(self) -> None:
        self.consecutive_hits += 1
        self.consecutive_misses = 0
        self.track_quality = self.track_quality + self.quality_hit_gain * (1.0 - self.track_quality)
        if self.consecutive_hits >= self.confirm_hits:
            self.track_lifecycle = "confirmed"
        elif self.track_lifecycle == "lost":
            self.track_lifecycle = "tentative"

    def _on_miss(self) -> None:
        self.consecutive_hits = 0
        self.consecutive_misses += 1
        self.track_quality *= self.quality_miss_decay
        if self.consecutive_misses >= self.forget_misses:
            self.track_lifecycle = "lost"
            self.track_quality = 0.0

    def _predict_state(self, timestamp: float) -> TrackState | None:
        if not self.allow_prediction:
            return None
        if self.last_state is None:
            return None

        if self.last_observed_timestamp is None:
            return None

        dt_since_state = float(timestamp - self.last_state.timestamp)
        dt_since_observed = float(timestamp - self.last_observed_timestamp)
        if dt_since_state < 0.0 or dt_since_observed < 0.0 or dt_since_observed > self.max_prediction_duration_s:
            return None

        predicted_position = self.last_state.position + self.last_state.velocity * dt_since_state
        return TrackState(
            timestamp=timestamp,
            position=predicted_position,
            velocity=self.last_state.velocity.copy(),
            source="predicted",
            candidate=None,
            track_quality=self.track_quality,
            track_lifecycle=self.track_lifecycle,
        )

    def update(self, timestamp: float, candidates: list[ClusterCandidate]) -> TrackState | None:
        predicted = self._predict_state(timestamp)
        selected: ClusterCandidate | None = None

        if candidates:
            if self.last_state is None or predicted is None:
                selected = max(candidates, key=lambda item: item.score)
            else:
                ranked = sorted(
                    candidates,
                    key=lambda item: np.linalg.norm(item.footpoint - predicted.position),
                )
                best = ranked[0]
                if np.linalg.norm(best.footpoint - predicted.position) <= self.gating_distance_m:
                    selected = best

        if selected is None:
            self._on_miss()
            self.last_state = predicted
            if self.last_state is not None:
                self.last_state.track_quality = self.track_quality
                self.last_state.track_lifecycle = self.track_lifecycle
            return predicted

        measurement = selected.footpoint.copy()
        self._on_hit()
        if self.last_state is None:
            new_state = TrackState(
                timestamp=timestamp,
                position=measurement,
                velocity=np.zeros(3, dtype=np.float64),
                source="observed",
                candidate=selected,
                track_quality=self.track_quality,
                track_lifecycle=self.track_lifecycle,
            )
            self.last_state = new_state
            self.last_observed_timestamp = timestamp
            return new_state

        dt = max(float(timestamp - self.last_state.timestamp), 1e-3)
        predicted_position = self.last_state.position + self.last_state.velocity * dt
        blended_position = predicted_position + self.process_gain * (measurement - predicted_position)
        new_velocity = (blended_position - self.last_state.position) / dt
        new_state = TrackState(
            timestamp=timestamp,
            position=blended_position,
            velocity=new_velocity,
            source="observed",
            candidate=selected,
            track_quality=self.track_quality,
            track_lifecycle=self.track_lifecycle,
        )
        self.last_state = new_state
        self.last_observed_timestamp = timestamp
        return new_state


def format_candidate(candidate: ClusterCandidate) -> dict[str, Any]:
    return {
        "cluster_id": candidate.cluster_id,
        "point_count": candidate.point_count,
        "score": round(float(candidate.score), 6),
        "centroid_lidar_m": np.round(candidate.centroid, 6).tolist(),
        "footpoint_lidar_m": np.round(candidate.footpoint, 6).tolist(),
        "size_lidar_m": np.round(candidate.size, 6).tolist(),
        "min_bound_lidar_m": np.round(candidate.min_bound, 6).tolist(),
        "max_bound_lidar_m": np.round(candidate.max_bound, 6).tolist(),
    }


def format_record(
    timestamp: float,
    point_count_raw: int,
    point_count_filtered: int,
    candidates: list[ClusterCandidate],
    camera_timestamp: float | None,
    sync_dt_ms: float | None,
    point_count_projected_to_front: int,
    point_count_in_person_mask: int,
    segmentation_candidate_count: int,
    state: TrackState | None,
) -> dict[str, Any]:
    iso_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().isoformat()
    return {
        "timestamp": timestamp,
        "datetime": iso_time,
        "point_count_raw": point_count_raw,
        "point_count_filtered": point_count_filtered,
        "camera_timestamp": camera_timestamp,
        "sync_dt_ms": None if sync_dt_ms is None else round(float(sync_dt_ms), 3),
        "point_count_projected_to_front": point_count_projected_to_front,
        "point_count_in_person_mask": point_count_in_person_mask,
        "candidate_count": len(candidates),
        "segmentation_candidate_count": segmentation_candidate_count,
        "candidates": [format_candidate(candidate) for candidate in candidates],
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
    }


def point_to_canvas(
    point_xy: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    canvas_size: tuple[int, int],
) -> tuple[int, int]:
    width, height = canvas_size
    x_min, x_max = x_limits
    y_min, y_max = y_limits

    x_norm = (float(point_xy[0]) - x_min) / max(x_max - x_min, 1e-6)
    y_norm = (float(point_xy[1]) - y_min) / max(y_max - y_min, 1e-6)
    px = int(np.clip(x_norm * (width - 1), 0, width - 1))
    py = int(np.clip((1.0 - y_norm) * (height - 1), 0, height - 1))
    return px, py


def render_grid(
    canvas: np.ndarray,
    axis_a_limits: tuple[float, float],
    axis_b_limits: tuple[float, float],
) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    for a_meter in np.arange(math.floor(axis_a_limits[0]), math.ceil(axis_a_limits[1]) + 1.0, 1.0):
        p0 = point_to_canvas(np.array([a_meter, axis_b_limits[0]]), axis_a_limits, axis_b_limits, (canvas_width, canvas_height))
        p1 = point_to_canvas(np.array([a_meter, axis_b_limits[1]]), axis_a_limits, axis_b_limits, (canvas_width, canvas_height))
        cv2.line(canvas, p0, p1, (230, 230, 230), 1)
    for b_meter in np.arange(math.floor(axis_b_limits[0]), math.ceil(axis_b_limits[1]) + 1.0, 1.0):
        p0 = point_to_canvas(np.array([axis_a_limits[0], b_meter]), axis_a_limits, axis_b_limits, (canvas_width, canvas_height))
        p1 = point_to_canvas(np.array([axis_a_limits[1], b_meter]), axis_a_limits, axis_b_limits, (canvas_width, canvas_height))
        cv2.line(canvas, p0, p1, (230, 230, 230), 1)

    if axis_a_limits[0] <= 0.0 <= axis_a_limits[1]:
        p0 = point_to_canvas(np.array([0.0, axis_b_limits[0]]), axis_a_limits, axis_b_limits, (canvas_width, canvas_height))
        p1 = point_to_canvas(np.array([0.0, axis_b_limits[1]]), axis_a_limits, axis_b_limits, (canvas_width, canvas_height))
        cv2.line(canvas, p0, p1, (120, 120, 120), 2)
    if axis_b_limits[0] <= 0.0 <= axis_b_limits[1]:
        p0 = point_to_canvas(np.array([axis_a_limits[0], 0.0]), axis_a_limits, axis_b_limits, (canvas_width, canvas_height))
        p1 = point_to_canvas(np.array([axis_a_limits[1], 0.0]), axis_a_limits, axis_b_limits, (canvas_width, canvas_height))
        cv2.line(canvas, p0, p1, (120, 120, 120), 2)


def z_to_bgr(z_value: float, z_limits: tuple[float, float]) -> tuple[int, int, int]:
    z_min, z_max = z_limits
    ratio = float(np.clip((z_value - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0))
    if ratio < 0.25:
        t = ratio / 0.25
        b, g, r = 255, int(255 * t), 0
    elif ratio < 0.5:
        t = (ratio - 0.25) / 0.25
        b, g, r = int(255 * (1 - t)), 255, 0
    elif ratio < 0.75:
        t = (ratio - 0.5) / 0.25
        b, g, r = 0, 255, int(255 * t)
    else:
        t = (ratio - 0.75) / 0.25
        b, g, r = 0, int(255 * (1 - t)), 255
    return int(b), int(g), int(r)


def draw_points_2d(
    canvas: np.ndarray,
    points: np.ndarray,
    axis_a_index: int,
    axis_b_index: int,
    axis_a_limits: tuple[float, float],
    axis_b_limits: tuple[float, float],
    z_limits: tuple[float, float],
    point_radius: int,
) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    if points.shape[0] == 0:
        return

    for point in points:
        draw_point = np.array([point[axis_a_index], point[axis_b_index]], dtype=np.float64)
        px, py = point_to_canvas(draw_point, axis_a_limits, axis_b_limits, (canvas_width, canvas_height))
        color = z_to_bgr(float(point[2]), z_limits)
        cv2.circle(canvas, (px, py), point_radius, color, -1, lineType=cv2.LINE_AA)


def draw_highlight_groups_2d(
    canvas: np.ndarray,
    point_groups: list[np.ndarray],
    axis_a_index: int,
    axis_b_index: int,
    axis_a_limits: tuple[float, float],
    axis_b_limits: tuple[float, float],
    point_radius: int,
) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    for group_index, points in enumerate(point_groups):
        if points.shape[0] == 0:
            continue
        color = tuple(int(v) for v in INSTANCE_PALETTE[group_index % len(INSTANCE_PALETTE)])
        for point in points:
            draw_point = np.array([point[axis_a_index], point[axis_b_index]], dtype=np.float64)
            px, py = point_to_canvas(draw_point, axis_a_limits, axis_b_limits, (canvas_width, canvas_height))
            cv2.circle(canvas, (px, py), point_radius + 1, color, -1, lineType=cv2.LINE_AA)


def candidate_color(cluster_id: int) -> tuple[int, int, int]:
    palette = [
        (50, 120, 255),
        (0, 180, 255),
        (80, 200, 120),
        (220, 160, 60),
        (180, 80, 220),
        (220, 80, 120),
    ]
    return palette[cluster_id % len(palette)]


def draw_candidate_boxes_bev(
    canvas: np.ndarray,
    candidates: list[ClusterCandidate],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    canvas_height, canvas_width = canvas.shape[:2]

    for candidate in candidates:
        color = candidate_color(candidate.cluster_id)
        min_xy = candidate.min_bound[:2]
        max_xy = candidate.max_bound[:2]
        top_left = point_to_canvas(np.array([min_xy[0], max_xy[1]]), x_limits, y_limits, (canvas_width, canvas_height))
        bottom_right = point_to_canvas(np.array([max_xy[0], min_xy[1]]), x_limits, y_limits, (canvas_width, canvas_height))
        overlay = canvas.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)
        cv2.rectangle(canvas, top_left, bottom_right, color, 2)

        foot_px = point_to_canvas(candidate.footpoint[:2], x_limits, y_limits, (canvas_width, canvas_height))
        center_px = point_to_canvas(candidate.centroid[:2], x_limits, y_limits, (canvas_width, canvas_height))
        cv2.circle(canvas, foot_px, 5, (0, 0, 255), -1)
        cv2.circle(canvas, center_px, 4, (255, 0, 0), -1)
        label = f"id={candidate.cluster_id} s={candidate.score:.2f} n={candidate.point_count}"
        cv2.putText(canvas, label, (top_left[0] + 4, max(top_left[1] - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)


def draw_candidate_boxes_side(
    canvas: np.ndarray,
    candidates: list[ClusterCandidate],
    x_limits: tuple[float, float],
    z_limits: tuple[float, float],
) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    for candidate in candidates:
        color = candidate_color(candidate.cluster_id)
        min_xz = np.array([candidate.min_bound[0], candidate.min_bound[2]], dtype=np.float64)
        max_xz = np.array([candidate.max_bound[0], candidate.max_bound[2]], dtype=np.float64)
        top_left = point_to_canvas(np.array([min_xz[0], max_xz[1]]), x_limits, z_limits, (canvas_width, canvas_height))
        bottom_right = point_to_canvas(np.array([max_xz[0], min_xz[1]]), x_limits, z_limits, (canvas_width, canvas_height))
        overlay = canvas.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)
        cv2.rectangle(canvas, top_left, bottom_right, color, 2)

        foot_px = point_to_canvas(candidate.footpoint[[0, 2]], x_limits, z_limits, (canvas_width, canvas_height))
        center_px = point_to_canvas(candidate.centroid[[0, 2]], x_limits, z_limits, (canvas_width, canvas_height))
        cv2.circle(canvas, foot_px, 5, (0, 0, 255), -1)
        cv2.circle(canvas, center_px, 4, (255, 0, 0), -1)


def draw_track_bev(
    canvas: np.ndarray,
    state: TrackState | None,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    if state is not None:
        track_px = point_to_canvas(state.position[:2], x_limits, y_limits, (canvas_width, canvas_height))
        color = (0, 180, 0) if state.source == "observed" else (0, 200, 200)
        cv2.circle(canvas, track_px, 8, color, -1)
        cv2.circle(canvas, track_px, 14, color, 2)


def draw_track_side(
    canvas: np.ndarray,
    state: TrackState | None,
    x_limits: tuple[float, float],
    z_limits: tuple[float, float],
) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    if state is not None:
        track_px = point_to_canvas(state.position[[0, 2]], x_limits, z_limits, (canvas_width, canvas_height))
        color = (0, 180, 0) if state.source == "observed" else (0, 200, 200)
        cv2.circle(canvas, track_px, 8, color, -1)
        cv2.circle(canvas, track_px, 14, color, 2)


def render_debug_image(
    points: np.ndarray,
    candidates: list[ClusterCandidate],
    state: TrackState | None,
    config: dict[str, Any],
    frame_index: int,
    timestamp: float,
    view: str,
    title: str,
    highlight_groups: list[np.ndarray] | None = None,
) -> np.ndarray:
    debug_cfg = config.get("debug", {})
    canvas_width = int(debug_cfg.get("image_width", 1200))
    canvas_height = int(debug_cfg.get("image_height", 900))
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)

    roi_cfg = config.get("roi", {})
    x_limits = tuple(float(v) for v in roi_cfg.get("x", [0.0, 20.0]))
    y_limits = tuple(float(v) for v in roi_cfg.get("y", [-10.0, 10.0]))
    z_limits = tuple(float(v) for v in roi_cfg.get("z", [-2.0, 3.0]))
    point_radius = max(int(debug_cfg.get("point_radius_px", 2)), 1)

    if view == "bev":
        render_grid(canvas, x_limits, y_limits)
        draw_points_2d(canvas, points, 0, 1, x_limits, y_limits, z_limits, point_radius)
        if highlight_groups:
            draw_highlight_groups_2d(canvas, highlight_groups, 0, 1, x_limits, y_limits, point_radius)
        draw_candidate_boxes_bev(canvas, candidates, x_limits, y_limits)
        draw_track_bev(canvas, state, x_limits, y_limits)
    elif view == "side":
        render_grid(canvas, x_limits, z_limits)
        draw_points_2d(canvas, points, 0, 2, x_limits, z_limits, z_limits, point_radius)
        draw_candidate_boxes_side(canvas, candidates, x_limits, z_limits)
        draw_track_side(canvas, state, x_limits, z_limits)
    else:
        raise ValueError(f"Unsupported debug view: {view}")

    if state is not None:
        color = (0, 180, 0) if state.source == "observed" else (0, 200, 200)
        cv2.putText(canvas, f"track={state.source} pos={np.round(state.position, 3).tolist()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    else:
        cv2.putText(canvas, "track=none", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 180), 2)
    cv2.putText(
        canvas,
        f"{title} frame={frame_index} ts={timestamp:.3f} points={points.shape[0]} candidates={len(candidates)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 40, 40),
        2,
    )
    cv2.putText(
        canvas,
        f"color=height z[{z_limits[0]:.1f},{z_limits[1]:.1f}] point_radius={point_radius}px",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
    )
    cv2.putText(
        canvas,
        f"range a[{axis_label(view)[0]}]={axis_limits_text(view, x_limits, y_limits, z_limits)[0]}  b[{axis_label(view)[1]}]={axis_limits_text(view, x_limits, y_limits, z_limits)[1]}",
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
    )
    return canvas


def axis_label(view: str) -> tuple[str, str]:
    if view == "bev":
        return "x", "y"
    return "x", "z"


def axis_limits_text(
    view: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
) -> tuple[str, str]:
    if view == "bev":
        return f"[{x_limits[0]:.1f},{x_limits[1]:.1f}]", f"[{y_limits[0]:.1f},{y_limits[1]:.1f}]"
    return f"[{x_limits[0]:.1f},{x_limits[1]:.1f}]", f"[{z_limits[0]:.1f},{z_limits[1]:.1f}]"


def crop_points_around_state(
    points: np.ndarray,
    state: TrackState | None,
    config: dict[str, Any],
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    debug_cfg = config.get("debug", {})
    half_width = float(debug_cfg.get("target_crop_half_width_m", 2.5))
    half_height = float(debug_cfg.get("target_crop_half_height_m", 2.5))
    roi_cfg = config.get("roi", {})
    default_x_limits = tuple(float(v) for v in roi_cfg.get("x", [0.0, 20.0]))
    default_y_limits = tuple(float(v) for v in roi_cfg.get("y", [-10.0, 10.0]))

    if state is None:
        return points, default_x_limits, default_y_limits

    x_limits = (float(state.position[0] - half_width), float(state.position[0] + half_width))
    y_limits = (float(state.position[1] - half_height), float(state.position[1] + half_height))
    mask = (
        (points[:, 0] >= x_limits[0])
        & (points[:, 0] <= x_limits[1])
        & (points[:, 1] >= y_limits[0])
        & (points[:, 1] <= y_limits[1])
    )
    return points[mask], x_limits, y_limits


def render_target_crop_image(
    points: np.ndarray,
    candidates: list[ClusterCandidate],
    state: TrackState | None,
    config: dict[str, Any],
    frame_index: int,
    timestamp: float,
) -> np.ndarray:
    debug_cfg = config.get("debug", {})
    canvas_width = int(debug_cfg.get("image_width", 1200))
    canvas_height = int(debug_cfg.get("image_height", 900))
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    cropped_points, x_limits, y_limits = crop_points_around_state(points, state, config)
    z_limits = tuple(float(v) for v in config.get("roi", {}).get("z", [-2.0, 3.0]))
    point_radius = max(int(debug_cfg.get("point_radius_px", 2)), 1)
    render_grid(canvas, x_limits, y_limits)
    draw_points_2d(canvas, cropped_points, 0, 1, x_limits, y_limits, z_limits, point_radius)

    visible_candidates = [
        candidate
        for candidate in candidates
        if x_limits[0] <= candidate.footpoint[0] <= x_limits[1] and y_limits[0] <= candidate.footpoint[1] <= y_limits[1]
    ]
    draw_candidate_boxes_bev(canvas, visible_candidates, x_limits, y_limits)
    draw_track_bev(canvas, state, x_limits, y_limits)
    cv2.putText(canvas, f"target crop frame={frame_index} ts={timestamp:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2)
    cv2.putText(canvas, "zoomed local BEV around current track", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    cv2.putText(canvas, f"color=height point_radius={point_radius}px", (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    return canvas


def render_overview_info_panel(
    front_frame: np.ndarray | None,
    camera_timestamp: float | None,
    sync_dt_ms: float | None,
    frame_index: int,
    timestamp: float,
    raw_points: np.ndarray,
    points: np.ndarray,
    candidates: list[ClusterCandidate],
    state: TrackState | None,
    config: dict[str, Any],
) -> np.ndarray:
    debug_cfg = config.get("debug", {})
    canvas_width = int(debug_cfg.get("image_width", 1200))
    canvas_height = int(debug_cfg.get("image_height", 900))
    if front_frame is not None:
        panel = cv2.resize(front_frame, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (canvas_width, 128), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, panel)
    else:
        panel = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    roi_cfg = config.get("roi", {})
    x_limits = tuple(float(v) for v in roi_cfg.get("x", [0.0, 20.0]))
    y_limits = tuple(float(v) for v in roi_cfg.get("y", [-10.0, 10.0]))
    z_limits = tuple(float(v) for v in roi_cfg.get("z", [-2.0, 3.0]))

    title = "front camera" if front_frame is not None else "lidar overview"
    title_color = (255, 255, 255) if front_frame is not None else (40, 40, 40)
    text_color = (255, 255, 255) if front_frame is not None else (60, 60, 60)
    cv2.putText(panel, title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, title_color, 2)
    cv2.putText(panel, f"frame={frame_index} ts={timestamp:.3f}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 60, 60), 2)
    cv2.putText(panel, f"raw_points={raw_points.shape[0]} filtered_points={points.shape[0]}", (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2)
    cv2.putText(panel, f"candidates={len(candidates)}", (20, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2)
    sync_text = "camera_ts=none sync_dt_ms=none"
    if camera_timestamp is not None and sync_dt_ms is not None:
        sync_text = f"camera_ts={camera_timestamp:.3f} sync_dt_ms={sync_dt_ms:.1f}"
    cv2.putText(panel, sync_text, (20, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.68, text_color, 2)

    track_text = "track=none"
    track_color = (0, 0, 180)
    if state is not None:
        track_text = f"track={state.source} pos={np.round(state.position, 3).tolist()}"
        track_color = (0, 180, 0) if state.source == "observed" else (0, 200, 200)
    cv2.putText(panel, track_text, (20, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.72, track_color, 2)

    cv2.putText(panel, f"roi x={x_limits} y={y_limits} z={z_limits}", (20, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)
    if front_frame is None:
        cv2.putText(panel, "layout: front/info | raw bev | filtered bev", (20, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 80, 80), 2)
        cv2.putText(panel, "        side view | target crop | empty", (20, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 80, 80), 2)
        cv2.putText(panel, "gray=points  orange=boxes  blue=centroid", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (90, 90, 90), 2)
        cv2.putText(panel, "red=footpoint  green/yellow=track", (20, 456), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (90, 90, 90), 2)
    return panel


def stack_debug_images(images: list[np.ndarray], config: dict[str, Any]) -> np.ndarray:
    debug_cfg = config.get("debug", {})
    canvas_width = int(debug_cfg.get("image_width", 1200))
    canvas_height = int(debug_cfg.get("image_height", 900))
    blank = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    cv2.putText(blank, "empty", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)

    padded = [cv2.resize(image, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR) for image in images]
    while len(padded) < 6:
        padded.append(blank.copy())
    top = np.hstack(padded[:3])
    bottom = np.hstack(padded[3:6])
    return np.vstack([top, bottom])


class DebugWriter:
    def __init__(self, config: dict[str, Any]) -> None:
        debug_cfg = config.get("debug", {})
        self.enabled = bool(debug_cfg.get("save_images", False))
        self.every_n_frames = max(int(debug_cfg.get("every_n_frames", 1)), 1)
        self.output_dir = Path(str(debug_cfg.get("output_dir", "outputs/lidar_debug")))
        self.save_raw_bev = bool(debug_cfg.get("save_raw_bev", True))
        self.save_filtered_bev = bool(debug_cfg.get("save_filtered_bev", True))
        self.save_side_view = bool(debug_cfg.get("save_side_view", True))
        self.save_target_crop = bool(debug_cfg.get("save_target_crop", True))
        self.save_overview = bool(debug_cfg.get("save_overview", True))
        self.save_segmentation = bool(debug_cfg.get("save_segmentation", True))
        self.front_camera_reader = RecentFrontCameraReader(config) if self.enabled and self.save_overview else None
        if self.enabled:
            for subdir in ["raw_bev", "filtered_bev", "side_view", "target_crop", "segmentation", "overview"]:
                (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def write(
        self,
        frame_index: int,
        timestamp: float,
        raw_points: np.ndarray,
        points: np.ndarray,
        candidates: list[ClusterCandidate],
        state: TrackState | None,
        config: dict[str, Any],
        front_frame: np.ndarray | None = None,
        segmentation: Any = None,
        person_mask: np.ndarray | None = None,
        selected_person_masks: list[np.ndarray] | None = None,
        raw_person_mask_count: int = 0,
        highlight_groups: list[np.ndarray] | None = None,
        person_class_id: int = 19,
        camera_timestamp: float | None = None,
        sync_dt_ms: float | None = None,
    ) -> None:
        if not self.enabled or frame_index % self.every_n_frames != 0:
            return

        suffix = f"frame_{frame_index:06d}_{timestamp:.3f}.png"
        if front_frame is None:
            front_frame = None if self.front_camera_reader is None else self.front_camera_reader.get_latest_before(timestamp)
        overview_images: list[np.ndarray] = [
            render_overview_info_panel(
                front_frame=front_frame,
                camera_timestamp=camera_timestamp,
                sync_dt_ms=sync_dt_ms,
                frame_index=frame_index,
                timestamp=timestamp,
                raw_points=raw_points,
                points=points,
                candidates=candidates,
                state=state,
                config=config,
            )
        ]

        if self.save_raw_bev:
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
            cv2.imwrite(str(self.output_dir / "raw_bev" / suffix), raw_bev)
            overview_images.append(raw_bev)

        if self.save_filtered_bev:
            filtered_bev = render_debug_image(
                points=points,
                candidates=candidates,
                state=state,
                config=config,
                frame_index=frame_index,
                timestamp=timestamp,
                view="bev",
                title="filtered bev",
                highlight_groups=highlight_groups,
            )
            cv2.imwrite(str(self.output_dir / "filtered_bev" / suffix), filtered_bev)
            overview_images.append(filtered_bev)

        if self.save_side_view:
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
            cv2.imwrite(str(self.output_dir / "side_view" / suffix), side_view)
            overview_images.append(side_view)

        if self.save_target_crop:
            target_crop = render_target_crop_image(
                points=points,
                candidates=candidates,
                state=state,
                config=config,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            cv2.imwrite(str(self.output_dir / "target_crop" / suffix), target_crop)
            overview_images.append(target_crop)

        if self.save_segmentation:
            segmentation_preview = render_segmentation_preview(
                frame=front_frame,
                segmentation=segmentation,
                person_mask=person_mask,
                selected_person_masks=selected_person_masks,
                raw_person_mask_count=raw_person_mask_count,
                frame_index=frame_index,
                timestamp=timestamp,
                camera_timestamp=camera_timestamp,
                sync_dt_ms=sync_dt_ms,
                person_class_id=person_class_id,
                config=config,
            )
            cv2.imwrite(str(self.output_dir / "segmentation" / suffix), segmentation_preview)
            overview_images.append(segmentation_preview)

        if self.save_overview:
            overview = stack_debug_images(overview_images, config)
            cv2.imwrite(str(self.output_dir / "overview" / suffix), overview)

    def release(self) -> None:
        if self.front_camera_reader is not None:
            self.front_camera_reader.release()


class JsonRecordWriter:
    def __init__(self, output_path: str | Path, flush_every_frame: bool) -> None:
        self.output_path = Path(output_path)
        self.flush_every_frame = flush_every_frame
        self.records: list[dict[str, Any]] = []
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self.flush_every_frame:
            self.flush()

    def flush(self) -> None:
        self.output_path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a person-like cluster from lidar rosbag data.")
    parser.add_argument("--config", default="configs/lidar_config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    source = RosbagLidarSource(config)

    clustering_cfg = config["clustering"]
    tracker_cfg = config["tracker"]
    logging_cfg = config["logging"]
    segmentation_cfg = config.get("segmentation", {})
    output_path = Path(str(logging_cfg.get("output_json", "outputs/lidar_tracking_log.json")))

    tracker = SimpleTracker(
        gating_distance_m=float(tracker_cfg.get("gating_distance_m", 1.0)),
        max_prediction_duration_s=float(tracker_cfg.get("max_prediction_duration_s", 0.5)),
        process_gain=float(tracker_cfg.get("process_gain", 0.6)),
        allow_prediction=bool(tracker_cfg.get("allow_prediction", False)),
        confirm_hits=int(tracker_cfg.get("confirm_hits", 3)),
        forget_misses=int(tracker_cfg.get("forget_misses", 8)),
        quality_hit_gain=float(tracker_cfg.get("quality_hit_gain", 0.2)),
        quality_miss_decay=float(tracker_cfg.get("quality_miss_decay", 0.85)),
    )
    front_camera_reader = RecentFrontCameraReader(config)
    segmentation_predictor = SegmentationPredictor(config)
    camera_context = load_camera_projection_context(segmentation_cfg.get("camera_config", "configs/aruco_config.yaml"))
    debug_writer = DebugWriter(config)
    json_writer = JsonRecordWriter(
        output_path=output_path,
        flush_every_frame=bool(logging_cfg.get("flush_every_frame", True)),
    )

    if segmentation_predictor.enabled:
        if segmentation_predictor.available:
            print("Segmentation guidance enabled for lidar_tracking.")
        else:
            print(f"Segmentation guidance unavailable; falling back to geometry-only tracking. error={segmentation_predictor.load_error}")

    frame_index = 0
    interrupted = False

    try:
        while True:
            ok, points, timestamp = source.read()
            if not ok or points is None or timestamp is None:
                break

            raw_count = int(points.shape[0])
            processed = crop_points(points, config.get("roi", {}))
            processed = remove_ego_vehicle_points(processed, config.get("ego_vehicle_filter", {}))
            processed = remove_ground(processed, config.get("ground_removal", {}))

            nearest_sample = front_camera_reader.get_nearest_sample(timestamp)
            front_frame = None if nearest_sample is None else nearest_sample[0]
            camera_timestamp = None if nearest_sample is None else nearest_sample[1]
            sync_dt_ms = None if camera_timestamp is None else 1000.0 * float(camera_timestamp - timestamp)
            segmentation = None if front_frame is None else segmentation_predictor.predict(front_frame)
            split_mode = "none" if segmentation_predictor.task in {"instance", "panoptic"} else "auto"
            person_masks = extract_person_masks(segmentation, segmentation_predictor.person_class_id, split_mode=split_mode)
            raw_person_mask_count = len(person_masks)
            min_person_mask_area_px = int(segmentation_cfg.get("min_person_mask_area_px", 3000))
            max_instances_considered = int(segmentation_cfg.get("max_instances_considered", 3))
            person_masks = select_relevant_person_masks(
                person_masks,
                image_shape=front_frame.shape if front_frame is not None else (0, 0, 0),
                min_area_px=min_person_mask_area_px,
                max_instances=max_instances_considered,
            )
            person_mask = extract_person_mask(segmentation, segmentation_predictor.person_class_id, split_mode=split_mode)
            if person_masks:
                person_mask = np.any(np.stack(person_masks, axis=0), axis=0)
            masked_points, projected_point_count = select_points_in_person_mask(
                points=processed,
                person_mask=person_mask,
                camera_context=camera_context,
            )
            highlight_groups: list[np.ndarray] = []

            clusters = euclidean_clusters(
                points=processed,
                tolerance=float(clustering_cfg["tolerance_m"]),
                min_points=int(clustering_cfg["min_cluster_points"]),
                max_points=int(clustering_cfg["max_cluster_points"]),
            )
            candidates = filter_candidates(clusters, config["person_cluster"])
            segmentation_candidates: list[ClusterCandidate] = []
            for instance_mask in person_masks:
                instance_points, _ = select_points_in_person_mask(
                    points=processed,
                    person_mask=instance_mask,
                    camera_context=camera_context,
                )
                if instance_points.shape[0] == 0:
                    continue
                highlight_groups.append(instance_points)
                instance_clusters = euclidean_clusters(
                    points=instance_points,
                    tolerance=float(clustering_cfg["tolerance_m"]),
                    min_points=int(clustering_cfg["min_cluster_points"]),
                    max_points=int(clustering_cfg["max_cluster_points"]),
                )
                segmentation_candidates.extend(filter_candidates(instance_clusters, config["person_cluster"]))
            prioritized_candidates = segmentation_candidates if segmentation_candidates else candidates
            state = tracker.update(timestamp=timestamp, candidates=prioritized_candidates)
            debug_writer.write(
                frame_index=frame_index,
                timestamp=timestamp,
                raw_points=points,
                points=processed,
                candidates=prioritized_candidates,
                state=state,
                config=config,
                front_frame=front_frame,
                segmentation=segmentation,
                person_mask=person_mask,
                selected_person_masks=person_masks,
                raw_person_mask_count=raw_person_mask_count,
                highlight_groups=highlight_groups,
                person_class_id=segmentation_predictor.person_class_id,
                camera_timestamp=camera_timestamp,
                sync_dt_ms=sync_dt_ms,
            )
            json_writer.append(
                format_record(
                    timestamp=timestamp,
                    point_count_raw=raw_count,
                    point_count_filtered=int(processed.shape[0]),
                    candidates=prioritized_candidates,
                    camera_timestamp=camera_timestamp,
                    sync_dt_ms=sync_dt_ms,
                    point_count_projected_to_front=int(projected_point_count),
                    point_count_in_person_mask=int(masked_points.shape[0]),
                    segmentation_candidate_count=len(segmentation_candidates),
                    state=state,
                )
            )

            if bool(logging_cfg.get("print_summary_every_frame", False)):
                track_text = "none" if state is None else f"{state.source} pos={np.round(state.position, 3).tolist()}"
                print(
                    f"time={timestamp:.3f} raw={raw_count} filtered={processed.shape[0]} "
                    f"candidates={len(candidates)} seg_candidates={len(segmentation_candidates)} "
                    f"mask_points={masked_points.shape[0]} masks={len(person_masks)} "
                    f"sync_dt_ms={sync_dt_ms if sync_dt_ms is not None else 'none'} track={track_text}"
                )
            frame_index += 1
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user, flushing partial results...")
    finally:
        source.release()
        front_camera_reader.release()
        debug_writer.release()
        json_writer.flush()

    if interrupted:
        print(f"Stopped early after {len(json_writer.records)} records. Partial log saved to {output_path}")
    else:
        print(f"Wrote {len(json_writer.records)} records to {output_path}")


if __name__ == "__main__":
    main()
