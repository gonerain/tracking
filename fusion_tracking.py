from __future__ import annotations

import argparse
import json
import sys
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
    SimpleTracker,
    crop_points,
    euclidean_clusters,
    filter_candidates,
    format_candidate,
    remove_ego_vehicle_points,
    remove_ground,
    render_debug_image,
    render_target_crop_image,
)
from core.segmentation import SegmentationPredictor
from core.tracking import TargetTracker, draw_tracking_overlay, select_target

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    segmentation: np.ndarray | None
    target_person_mask: np.ndarray | None


class ArucoPriorProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.context = build_detector_context(config)
        self.frame_source = FrameSource(config)
        tracking_cfg = config.get("tracking", {})
        self.target_id = int(tracking_cfg["target_id"])
        self.tracker = TargetTracker(
            history_size=int(tracking_cfg.get("history_size", 10)),
            max_prediction_duration_s=float(tracking_cfg.get("max_prediction_duration_s", 1.0)),
        )
        self.latest_timestamp: float | None = None
        self.latest_visible = False
        self.pending_frame: tuple[np.ndarray, float] | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_results: list[dict[str, Any]] = []
        self.latest_tracking_state: Any = None
        self.latest_target_result: dict[str, Any] | None = None
        self.segmentation_predictor = SegmentationPredictor(config)
        self.latest_segmentation: np.ndarray | None = None
        self.latest_target_person_mask: np.ndarray | None = None
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
            target = select_target(results, self.target_id)
            self.latest_target_result = target
            self.latest_segmentation = self.segmentation_predictor.predict(frame)
            self.latest_target_person_mask = select_target_person_mask(
                frame=frame,
                segmentation=self.latest_segmentation,
                target_result=target,
                person_class_id=self.segmentation_predictor.person_class_id,
            )
            if target is not None:
                self.latest_tracking_state = self.tracker.update(
                    timestamp=frame_timestamp,
                    position_camera=np.asarray(target["center_in_camera_m"], dtype=np.float64),
                    position_target=np.asarray(target["center_in_target_m"], dtype=np.float64),
                    corners=np.asarray(target["corners"], dtype=np.float64),
                )
                self.latest_visible = True
            else:
                self.latest_tracking_state = self.tracker.predict(frame_timestamp)
                self.latest_visible = False

    def _choose_nearest_timestamp(self, timestamp: float) -> tuple[np.ndarray, float] | None:
        self._consume_until(timestamp)
        previous = None if self.latest_frame is None or self.latest_timestamp is None else (self.latest_frame.copy(), float(self.latest_timestamp))
        next_sample = None
        if self.pending_frame is not None:
            next_sample = (self.pending_frame[0].copy(), float(self.pending_frame[1]))

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
            self.latest_sync_dt_ms = None
            return None
        if best_dt is not None and best_dt > self.sync_max_dt_s:
            self.latest_sync_dt_ms = None
            return None
        self.latest_sync_dt_ms = 1000.0 * float(best[1] - timestamp)
        return best

    def get_prior(self, timestamp: float) -> ArucoPrior | None:
        nearest = self._choose_nearest_timestamp(timestamp)
        if nearest is None:
            return None
        _, nearest_timestamp = nearest
        if self.latest_timestamp is None:
            return None

        if abs(nearest_timestamp - self.latest_timestamp) <= 1e-6 and self.latest_visible and self.tracker.last_observed is not None:
            observed = self.tracker.last_observed
            return ArucoPrior(
                timestamp=float(observed.timestamp),
                visible=True,
                source="observed",
                position_lidar=np.asarray(observed.position_target, dtype=np.float64).copy(),
            )

        predicted = self.tracker.predict(timestamp)
        if predicted is None:
            return None

        return ArucoPrior(
            timestamp=float(predicted.timestamp),
            visible=False,
            source=predicted.source,
            position_lidar=np.asarray(predicted.position_target, dtype=np.float64).copy(),
        )

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
            segmentation=None if self.latest_segmentation is None else self.latest_segmentation.copy(),
            target_person_mask=None if self.latest_target_person_mask is None else self.latest_target_person_mask.copy(),
        )


def select_target_person_mask(
    frame: np.ndarray,
    segmentation: Any,
    target_result: dict[str, Any] | None,
    person_class_id: int = 19,
) -> np.ndarray | None:
    if segmentation is None or target_result is None:
        return None

    if isinstance(segmentation, dict):
        if "pred_masks" in segmentation:
            masks = segmentation.get("pred_masks", np.zeros((0, frame.shape[0], frame.shape[1]), dtype=bool))
            classes = segmentation.get("pred_classes", np.zeros((0,), dtype=np.int64))
            person_masks = [np.asarray(mask, dtype=bool) for mask, cls in zip(masks, classes) if int(cls) == int(person_class_id)]
            if not person_masks:
                return None
            person_mask = np.any(np.stack(person_masks, axis=0), axis=0)
        elif "panoptic_seg" in segmentation:
            panoptic_seg = segmentation.get("panoptic_seg")
            segments_info = segmentation.get("segments_info", [])
            if panoptic_seg is None:
                return None
            person_mask = np.zeros_like(panoptic_seg, dtype=bool)
            for segment in segments_info:
                if int(segment.get("category_id", -1)) == int(person_class_id):
                    person_mask |= panoptic_seg == int(segment["id"])
        else:
            return None
    else:
        person_mask = np.asarray(segmentation, dtype=np.int32) == int(person_class_id)

    if not np.any(person_mask):
        return None

    center = np.round(target_result["center_projected_px"]).astype(int)
    h, w = person_mask.shape[:2]
    cx = int(np.clip(center[0], 0, w - 1))
    cy = int(np.clip(center[1], 0, h - 1))

    if not person_mask[cy, cx]:
        ys, xs = np.where(person_mask)
        if xs.size == 0:
            return None
        distances = (xs - cx) ** 2 + (ys - cy) ** 2
        nearest_idx = int(np.argmin(distances))
        cx = int(xs[nearest_idx])
        cy = int(ys[nearest_idx])

    num_labels, labels, _, _ = cv2.connectedComponents(person_mask.astype(np.uint8))
    if num_labels <= 1:
        return person_mask
    target_label = int(labels[cy, cx])
    if target_label == 0:
        return None
    return labels == target_label


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
        self.target_id = int(tracking_cfg.get("target_id", 0))
        if self.enabled:
            for subdir in [
                "front_camera",
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
        vis = draw_tracking_overlay(vis, self.target_id, aruco_debug.tracking_state)
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
        return vis

    def _pad_tile(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.tile_width, self.tile_height), interpolation=cv2.INTER_LINEAR)

    def _compose_overview(self, images: list[np.ndarray]) -> np.ndarray:
        blank = np.full((self.tile_height, self.tile_width, 3), 245, dtype=np.uint8)
        cv2.putText(blank, "empty", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
        padded = [self._pad_tile(image) for image in images]
        while len(padded) < 6:
            padded.append(blank.copy())
        top = np.hstack(padded[:3])
        bottom = np.hstack(padded[3:6])
        return np.vstack([top, bottom])

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
        overview = self._compose_overview([front_camera, raw_bev, filtered_bev, side_view, target_crop])

        cv2.imwrite(str(self.output_dir / "front_camera" / f"{suffix}.png"), front_camera)
        cv2.imwrite(str(self.output_dir / "raw_bev" / f"{suffix}.png"), raw_bev)
        cv2.imwrite(str(self.output_dir / "filtered_bev" / f"{suffix}.png"), filtered_bev)
        cv2.imwrite(str(self.output_dir / "side_view" / f"{suffix}.png"), side_view)
        cv2.imwrite(str(self.output_dir / "target_crop" / f"{suffix}.png"), target_crop)
        cv2.imwrite(str(self.output_dir / "overview" / f"{suffix}.png"), overview)
        self._write_pcd(self.output_dir / "raw_pcd" / f"{suffix}.pcd", raw_points)
        self._write_pcd(self.output_dir / "filtered_pcd" / f"{suffix}.pcd", points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse ArUco target prior with lidar person-cluster tracking.")
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
    return parser.parse_args()


def select_candidate_with_aruco_prior(
    candidates: list[ClusterCandidate],
    aruco_prior: ArucoPrior | None,
    gating_distance_m: float,
) -> ClusterCandidate | None:
    if aruco_prior is None or not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda item: np.linalg.norm(item.footpoint[:2] - aruco_prior.position_lidar[:2]),
    )
    best = ranked[0]
    best_distance = float(np.linalg.norm(best.footpoint[:2] - aruco_prior.position_lidar[:2]))
    if best_distance > gating_distance_m:
        return None
    return best


def nearest_candidate_distance(
    candidates: list[ClusterCandidate],
    aruco_prior: ArucoPrior | None,
) -> float | None:
    if aruco_prior is None or not candidates:
        return None
    return min(float(np.linalg.norm(candidate.footpoint[:2] - aruco_prior.position_lidar[:2])) for candidate in candidates)


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


def format_fusion_record(
    timestamp: float,
    point_count_raw: int,
    point_count_filtered: int,
    candidates: list[ClusterCandidate],
    aruco_prior: ArucoPrior | None,
    aruco_selected_candidate: ClusterCandidate | None,
    aruco_nearest_distance_m: float | None,
    camera_timestamp: float | None,
    sync_dt_ms: float | None,
    segmentation_point_count: int,
    projected_point_count: int,
    segmentation_candidate_count: int,
    state: Any,
) -> dict[str, Any]:
    iso_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().isoformat()
    return {
        "timestamp": timestamp,
        "datetime": iso_time,
        "point_count_raw": point_count_raw,
        "point_count_filtered": point_count_filtered,
        "camera_timestamp": camera_timestamp,
        "sync_dt_ms": None if sync_dt_ms is None else round(float(sync_dt_ms), 3),
        "point_count_projected_to_front": projected_point_count,
        "point_count_in_target_mask": segmentation_point_count,
        "candidate_count": len(candidates),
        "candidates": [format_candidate(candidate) for candidate in candidates],
        "aruco_prior": None
        if aruco_prior is None
        else {
            "timestamp": round(float(aruco_prior.timestamp), 6),
            "visible": aruco_prior.visible,
            "source": aruco_prior.source,
            "position_lidar_m": np.round(aruco_prior.position_lidar, 6).tolist(),
            "nearest_candidate_distance_m": None if aruco_nearest_distance_m is None else round(float(aruco_nearest_distance_m), 6),
            "segmentation_candidate_count": segmentation_candidate_count,
        },
        "aruco_selected_candidate": None if aruco_selected_candidate is None else format_candidate(aruco_selected_candidate),
        "track": None
        if state is None
        else {
            "state": state.source,
            "position_lidar_m": np.round(state.position, 6).tolist(),
            "velocity_lidar_mps": np.round(state.velocity, 6).tolist(),
            "selected_candidate": None if state.candidate is None else format_candidate(state.candidate),
        },
    }


def main() -> None:
    args = parse_args()
    aruco_config = load_aruco_config(args.aruco_config)
    lidar_config = load_aruco_config(args.lidar_config)

    aruco_provider = ArucoPriorProvider(aruco_config)
    lidar_source = RosbagLidarSource(lidar_config)

    clustering_cfg = lidar_config["clustering"]
    tracker_cfg = lidar_config["tracker"]
    logging_cfg = lidar_config.get("logging", {})

    tracker = SimpleTracker(
        gating_distance_m=float(tracker_cfg.get("gating_distance_m", 1.0)),
        max_prediction_duration_s=float(tracker_cfg.get("max_prediction_duration_s", 0.5)),
        process_gain=float(tracker_cfg.get("process_gain", 0.6)),
    )
    debug_writer = FusionDebugWriter(
        {
            "debug": lidar_config.get("debug", {}),
            "tracking": aruco_config.get("tracking", {}),
        }
    )
    json_writer = JsonRecordWriter(
        output_path=Path(str(args.output_json)),
        flush_every_frame=bool(logging_cfg.get("flush_every_frame", True)),
    )

    frame_index = 0
    interrupted = False

    try:
        while True:
            ok, points, timestamp = lidar_source.read()
            if not ok or points is None or timestamp is None:
                break

            raw_count = int(points.shape[0])
            processed = crop_points(points, lidar_config.get("roi", {}))
            processed = remove_ego_vehicle_points(processed, lidar_config.get("ego_vehicle_filter", {}))
            processed = remove_ground(processed, lidar_config.get("ground_removal", {}))

            clusters = euclidean_clusters(
                points=processed,
                tolerance=float(clustering_cfg["tolerance_m"]),
                min_points=int(clustering_cfg["min_cluster_points"]),
                max_points=int(clustering_cfg["max_cluster_points"]),
            )
            candidates = filter_candidates(clusters, lidar_config["person_cluster"])

            aruco_prior = aruco_provider.get_prior(timestamp)
            aruco_debug = aruco_provider.get_debug_state()
            segmented_points, projected_point_count = select_points_in_target_mask(
                points=processed,
                aruco_debug=aruco_debug,
                aruco_context=aruco_provider.context,
            )
            segmented_clusters = euclidean_clusters(
                points=segmented_points,
                tolerance=float(clustering_cfg["tolerance_m"]),
                min_points=int(clustering_cfg["min_cluster_points"]),
                max_points=int(clustering_cfg["max_cluster_points"]),
            ) if segmented_points.shape[0] > 0 else []
            segmentation_candidates = filter_candidates(segmented_clusters, lidar_config["person_cluster"])
            aruco_nearest_distance_m = nearest_candidate_distance(candidates, aruco_prior)
            prioritized_candidates = segmentation_candidates if segmentation_candidates else candidates
            aruco_selected_candidate = select_candidate_with_aruco_prior(
                candidates=prioritized_candidates,
                aruco_prior=aruco_prior,
                gating_distance_m=float(args.aruco_gating_distance),
            )

            tracker_candidates = [aruco_selected_candidate] if aruco_selected_candidate is not None else prioritized_candidates
            state = tracker.update(timestamp=timestamp, candidates=tracker_candidates)

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
            )
            record = format_fusion_record(
                timestamp=timestamp,
                point_count_raw=raw_count,
                point_count_filtered=int(processed.shape[0]),
                candidates=candidates,
                aruco_prior=aruco_prior,
                aruco_selected_candidate=aruco_selected_candidate,
                aruco_nearest_distance_m=aruco_nearest_distance_m,
                camera_timestamp=aruco_debug.frame_timestamp,
                sync_dt_ms=aruco_debug.sync_dt_ms,
                segmentation_point_count=int(segmented_points.shape[0]),
                projected_point_count=int(projected_point_count),
                segmentation_candidate_count=len(segmentation_candidates),
                state=state,
            )
            json_writer.append(record)

            if bool(logging_cfg.get("print_summary_every_frame", False)):
                aruco_text = "none"
                if aruco_prior is not None:
                    distance_text = "none" if aruco_nearest_distance_m is None else f"{aruco_nearest_distance_m:.3f}"
                    aruco_text = (
                        f"{aruco_prior.source} pos={np.round(aruco_prior.position_lidar, 3).tolist()} "
                        f"nearest={distance_text} mask_points={segmented_points.shape[0]} "
                        f"sync_dt_ms={aruco_debug.sync_dt_ms if aruco_debug.sync_dt_ms is not None else 'none'}"
                    )
                fused_text = "none" if state is None else f"{state.source} pos={np.round(state.position, 3).tolist()}"
                selected_text = (
                    "none"
                    if aruco_selected_candidate is None
                    else np.round(aruco_selected_candidate.footpoint, 3).tolist()
                )
                print(
                    f"time={timestamp:.3f} raw={raw_count} filtered={processed.shape[0]} "
                    f"candidates={len(candidates)} aruco={aruco_text} aruco_match={selected_text} fused={fused_text}"
                )
            else:
                print(json.dumps(record, ensure_ascii=False))

            frame_index += 1
    except KeyboardInterrupt:
        interrupted = True
    finally:
        lidar_source.release()
        aruco_provider.release()
        json_writer.flush()

    if interrupted:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
