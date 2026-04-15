from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
    TrackState,
    crop_points,
    euclidean_clusters,
    filter_candidates,
    format_candidate,
    remove_ego_vehicle_points,
    remove_ground,
    render_debug_image,
    render_segmentation_preview,
    render_target_crop_image,
    stack_debug_images,
)
from core.segmentation import SegmentationPredictor, extract_person_masks, select_relevant_person_masks
from core.tracking import TargetTracker, draw_tracking_overlay, get_target_ids, get_target_offsets, select_target, target_ids_label


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
    segmentation: Any
    target_person_mask: np.ndarray | None
    selected_person_masks: list[np.ndarray]
    raw_person_mask_count: int


class CenterPointCandidateProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        detector_cfg = config.get("detector", {})
        centerpoint_cfg = detector_cfg.get("centerpoint", {})
        self.enabled = bool(detector_cfg.get("enabled", False)) and str(detector_cfg.get("source", "cluster")) in {"centerpoint", "hybrid"}
        self.source_mode = str(detector_cfg.get("source", "cluster"))
        self.provider = str(centerpoint_cfg.get("provider", "none")).lower()
        self.person_label_ids = {int(v) for v in centerpoint_cfg.get("person_label_ids", [0])}
        self.score_threshold = float(centerpoint_cfg.get("score_threshold", 0.2))
        self._module_fn = None
        self._mmdet_model = None
        self.load_error: str | None = None

        if not self.enabled:
            return
        try:
            if self.provider == "python_module":
                module_path = Path(str(centerpoint_cfg.get("module_path", "")))
                function_name = str(centerpoint_cfg.get("function_name", "infer"))
                if not module_path.exists():
                    raise RuntimeError(f"centerpoint module not found: {module_path}")
                spec = importlib.util.spec_from_file_location("centerpoint_external_module", str(module_path))
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"failed to load module spec: {module_path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                fn = getattr(module, function_name, None)
                if fn is None:
                    raise RuntimeError(f"function '{function_name}' not found in {module_path}")
                self._module_fn = fn
            elif self.provider == "mmdet3d":
                from mmdet3d.apis import inference_detector, init_model  # type: ignore

                config_path = str(centerpoint_cfg.get("config_path", ""))
                checkpoint_path = str(centerpoint_cfg.get("checkpoint_path", ""))
                device = str(centerpoint_cfg.get("device", "cuda:0"))
                if not config_path or not checkpoint_path:
                    raise RuntimeError("centerpoint mmdet3d requires config_path and checkpoint_path")
                self._mmdet_model = {
                    "model": init_model(config_path, checkpoint_path, device=device),
                    "infer": inference_detector,
                }
            else:
                raise RuntimeError(f"unsupported centerpoint provider: {self.provider}")
        except Exception as exc:
            self.load_error = str(exc)
            self.enabled = False

    def _from_box(self, index: int, center_xyz: np.ndarray, size_xyz: np.ndarray, score: float) -> ClusterCandidate:
        center = np.asarray(center_xyz, dtype=np.float64).reshape(3)
        size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
        half = size / 2.0
        min_bound = center - half
        max_bound = center + half
        footpoint = center.copy()
        footpoint[2] = center[2] - half[2]
        return ClusterCandidate(
            cluster_id=100000 + int(index),
            point_count=0,
            centroid=center,
            footpoint=footpoint,
            size=size,
            min_bound=min_bound,
            max_bound=max_bound,
            score=float(score),
        )

    def _parse_mmdet3d_result(self, result: Any) -> list[ClusterCandidate]:
        data = result
        if isinstance(result, tuple) and result:
            data = result[0]
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict) and "pred_instances_3d" in data:
            pred = data["pred_instances_3d"]
            boxes = getattr(pred, "bboxes_3d", None)
            scores = getattr(pred, "scores_3d", None)
            labels = getattr(pred, "labels_3d", None)
            if boxes is None:
                return []
            box_tensor = boxes.tensor.detach().cpu().numpy()
            score_arr = scores.detach().cpu().numpy() if scores is not None else np.ones((box_tensor.shape[0],), dtype=np.float64)
            label_arr = labels.detach().cpu().numpy() if labels is not None else np.zeros((box_tensor.shape[0],), dtype=np.int32)
        elif isinstance(data, dict) and "pts_bbox" in data:
            pts = data["pts_bbox"]
            boxes = pts.get("boxes_3d")
            scores = pts.get("scores_3d")
            labels = pts.get("labels_3d")
            if boxes is None:
                return []
            box_tensor = boxes.tensor.detach().cpu().numpy()
            score_arr = scores.detach().cpu().numpy() if scores is not None else np.ones((box_tensor.shape[0],), dtype=np.float64)
            label_arr = labels.detach().cpu().numpy() if labels is not None else np.zeros((box_tensor.shape[0],), dtype=np.int32)
        else:
            return []

        out: list[ClusterCandidate] = []
        for i in range(box_tensor.shape[0]):
            score = float(score_arr[i])
            label = int(label_arr[i])
            if score < self.score_threshold or label not in self.person_label_ids:
                continue
            center = box_tensor[i, 0:3]
            size = box_tensor[i, 3:6]
            out.append(self._from_box(i, center, size, score))
        return out

    def detect(self, points_xyz: np.ndarray, timestamp: float) -> list[ClusterCandidate]:
        if not self.enabled or points_xyz.size == 0:
            return []
        try:
            if self.provider == "python_module":
                rows = self._module_fn(points_xyz, timestamp)  # type: ignore[misc]
                candidates: list[ClusterCandidate] = []
                for i, row in enumerate(rows or []):
                    score = float(row.get("score", 1.0))
                    label = int(row.get("label", 0))
                    if score < self.score_threshold or label not in self.person_label_ids:
                        continue
                    center = np.asarray(row["center_lidar_m"], dtype=np.float64)
                    size = np.asarray(row["size_lidar_m"], dtype=np.float64)
                    candidates.append(self._from_box(i, center, size, score))
                return candidates
            if self.provider == "mmdet3d":
                infer = self._mmdet_model["infer"]
                model = self._mmdet_model["model"]
                points = np.concatenate([points_xyz, np.zeros((points_xyz.shape[0], 1), dtype=np.float32)], axis=1).astype(np.float32)
                result = infer(model, points)
                return self._parse_mmdet3d_result(result)
        except Exception as exc:
            self.load_error = str(exc)
        return []


class ArucoPriorProvider:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.context = build_detector_context(config)
        self.frame_source = FrameSource(config)
        tracking_cfg = config.get("tracking", {})
        self.target_ids = get_target_ids(tracking_cfg)
        self.target_offsets = get_target_offsets(tracking_cfg, self.target_ids)
        self.target_label = target_ids_label(self.target_ids)
        self.tracker = TargetTracker(
            history_size=int(tracking_cfg.get("history_size", 10)),
            max_prediction_duration_s=float(tracking_cfg.get("max_prediction_duration_s", 1.0)),
            allow_prediction=bool(tracking_cfg.get("allow_prediction", False)),
        )
        self.latest_timestamp: float | None = None
        self.latest_visible = False
        self.pending_frame: tuple[np.ndarray, float] | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_results: list[dict[str, Any]] = []
        self.latest_tracking_state: Any = None
        self.latest_target_result: dict[str, Any] | None = None
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
            target = select_target(results, self.target_ids, self.target_offsets)
            self.latest_target_result = target
            self.latest_segmentation = self.segmentation_predictor.predict(frame)
            (
                self.latest_target_person_mask,
                self.latest_selected_person_masks,
                self.latest_raw_person_mask_count,
            ) = select_target_person_mask(
                frame=frame,
                segmentation=self.latest_segmentation,
                target_result=target,
                person_class_id=self.segmentation_predictor.person_class_id,
                min_area_px=int(self.segmentation_cfg.get("min_person_mask_area_px", 3000)),
                max_instances=int(self.segmentation_cfg.get("max_instances_considered", 3)),
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
                self.latest_tracking_state = None
                self.latest_visible = False

    def get_prior(self, timestamp: float) -> ArucoPrior | None:
        self._consume_until(timestamp)
        if self.latest_timestamp is None:
            self.latest_sync_dt_ms = None
            return None
        dt_s = float(self.latest_timestamp - timestamp)
        if abs(dt_s) > self.sync_max_dt_s:
            self.latest_sync_dt_ms = None
            return None
        self.latest_sync_dt_ms = 1000.0 * dt_s

        if self.latest_visible and self.tracker.last_observed is not None:
            observed = self.tracker.last_observed
            return ArucoPrior(
                timestamp=float(observed.timestamp),
                visible=True,
                source="observed",
                position_lidar=np.asarray(observed.position_target, dtype=np.float64).copy(),
            )
        return None

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
            selected_person_masks=[mask.copy() for mask in self.latest_selected_person_masks],
            raw_person_mask_count=int(self.latest_raw_person_mask_count),
        )


def select_target_person_mask(
    frame: np.ndarray,
    segmentation: Any,
    target_result: dict[str, Any] | None,
    person_class_id: int = 19,
    min_area_px: int = 0,
    max_instances: int | None = None,
) -> tuple[np.ndarray | None, list[np.ndarray], int]:
    if segmentation is None or target_result is None:
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

    center = np.round(target_result["center_projected_px"]).astype(int)
    h, w = frame.shape[:2]
    cx = int(np.clip(center[0], 0, w - 1))
    cy = int(np.clip(center[1], 0, h - 1))
    best_mask = None
    best_distance = None
    for mask in person_masks:
        mask = np.asarray(mask, dtype=bool)
        if mask[cy, cx]:
            return mask, person_masks, raw_person_mask_count
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        distances = (xs - cx) ** 2 + (ys - cy) ** 2
        nearest_distance = float(np.min(distances))
        if best_mask is None or nearest_distance < best_distance:
            best_mask = mask
            best_distance = nearest_distance
    return best_mask, person_masks, raw_person_mask_count


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
        self.target_label = target_ids_label(get_target_ids(tracking_cfg))
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
        vis = draw_tracking_overlay(vis, self.target_label, aruco_debug.tracking_state, visible_ids)
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


def nearest_candidate_distance(
    candidates: list[ClusterCandidate],
    aruco_prior: ArucoPrior | None,
) -> float | None:
    if aruco_prior is None or not candidates:
        return None
    return min(float(np.linalg.norm(candidate.footpoint[:2] - aruco_prior.position_lidar[:2])) for candidate in candidates)


def merge_prioritized_candidates(
    segmentation_candidates: list[ClusterCandidate],
    candidates: list[ClusterCandidate],
    dedup_distance_m: float = 0.2,
) -> list[ClusterCandidate]:
    merged: list[ClusterCandidate] = list(segmentation_candidates)
    for candidate in candidates:
        duplicate = any(
            float(np.linalg.norm(existing.footpoint[:2] - candidate.footpoint[:2])) <= dedup_distance_m
            for existing in merged
        )
        if not duplicate:
            merged.append(candidate)
    return merged


def select_lidar_candidates(
    source_mode: str,
    geometric_candidates: list[ClusterCandidate],
    centerpoint_candidates: list[ClusterCandidate],
    dedup_distance_m: float = 0.4,
) -> tuple[list[ClusterCandidate], str]:
    mode = source_mode.lower()
    if mode == "centerpoint":
        if centerpoint_candidates:
            return centerpoint_candidates, "centerpoint"
        return geometric_candidates, "cluster_fallback"
    if mode == "hybrid":
        merged = merge_prioritized_candidates(
            segmentation_candidates=centerpoint_candidates,
            candidates=geometric_candidates,
            dedup_distance_m=dedup_distance_m,
        )
        return merged, "hybrid"
    return geometric_candidates, "cluster"


def build_aruco_rescue_candidates(
    clusters: list[np.ndarray],
    person_cfg: dict[str, Any],
    aruco_prior: ArucoPrior | None,
) -> list[ClusterCandidate]:
    if aruco_prior is None:
        return []
    rescue_cfg = dict(person_cfg)
    if "min_points_with_aruco" in person_cfg:
        rescue_cfg["min_points"] = int(person_cfg["min_points_with_aruco"])
    if "height_m_with_aruco" in person_cfg:
        rescue_cfg["height_m"] = person_cfg["height_m_with_aruco"]
    if "width_m_with_aruco" in person_cfg:
        rescue_cfg["width_m"] = person_cfg["width_m_with_aruco"]
    if "depth_m_with_aruco" in person_cfg:
        rescue_cfg["depth_m"] = person_cfg["depth_m_with_aruco"]

    relaxed = filter_candidates(clusters, rescue_cfg)
    max_distance_m = float(person_cfg.get("aruco_rescue_distance_m", 1.2))
    return [
        candidate
        for candidate in relaxed
        if float(np.linalg.norm(candidate.footpoint[:2] - aruco_prior.position_lidar[:2])) <= max_distance_m
    ]


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
    roi_cfg: dict[str, Any],
    reconnect_streak: int = 0,
    force_disable: bool = False,
) -> np.ndarray:
    if points.size == 0 or aruco_prior is None:
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
    center = aruco_prior.position_lidar
    mask = (
        (points[:, 0] >= center[0] - half_x)
        & (points[:, 0] <= center[0] + half_x)
        & (points[:, 1] >= center[1] - half_y)
        & (points[:, 1] <= center[1] + half_y)
        & (points[:, 2] >= center[2] + z_min_offset)
        & (points[:, 2] <= center[2] + z_max_offset)
    )
    return points[mask]


def format_fusion_record(
    timestamp: float,
    point_count_raw: int,
    point_count_after_roi: int,
    point_count_after_ego_filter: int,
    point_count_after_ground_removal: int,
    point_count_after_aruco_local_roi: int,
    point_count_filtered: int,
    candidates: list[ClusterCandidate],
    geometric_candidates: list[ClusterCandidate],
    centerpoint_candidates: list[ClusterCandidate],
    candidate_source: str,
    segmentation_candidates: list[ClusterCandidate],
    aruco_rescue_candidates: list[ClusterCandidate],
    prioritized_candidates: list[ClusterCandidate],
    tracker_candidates: list[ClusterCandidate],
    candidate_diagnostics: list[dict[str, Any]],
    cluster_count: int,
    segmentation_cluster_count: int,
    aruco_prior: ArucoPrior | None,
    aruco_selected_candidate: ClusterCandidate | None,
    aruco_nearest_distance_m: float | None,
    camera_timestamp: float | None,
    sync_dt_ms: float | None,
    segmentation_point_count: int,
    projected_point_count: int,
    segmentation_candidate_count: int,
    used_aruco_fallback: bool,
    state: Any,
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
        "cluster_count": int(cluster_count),
        "segmentation_cluster_count": int(segmentation_cluster_count),
        "candidate_source": str(candidate_source),
        "candidate_count": len(candidates),
        "candidate_count_geometric": len(geometric_candidates),
        "candidate_count_centerpoint": len(centerpoint_candidates),
        "candidate_count_segmentation": len(segmentation_candidates),
        "candidate_count_aruco_rescue": len(aruco_rescue_candidates),
        "candidate_count_prioritized": len(prioritized_candidates),
        "candidate_count_tracker_input": len(tracker_candidates),
        "candidates": [format_candidate(candidate) for candidate in candidates],
        "geometric_candidates": [format_candidate(candidate) for candidate in geometric_candidates],
        "centerpoint_candidates": [format_candidate(candidate) for candidate in centerpoint_candidates],
        "segmentation_candidates": [format_candidate(candidate) for candidate in segmentation_candidates],
        "aruco_rescue_candidates": [format_candidate(candidate) for candidate in aruco_rescue_candidates],
        "prioritized_candidates": [format_candidate(candidate) for candidate in prioritized_candidates],
        "tracker_input_candidates": [format_candidate(candidate) for candidate in tracker_candidates],
        "candidate_diagnostics": candidate_diagnostics,
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
    }


def main() -> None:
    args = parse_args()
    aruco_config = load_aruco_config(args.aruco_config)
    lidar_config = load_aruco_config(args.lidar_config)
    aruco_runtime_config = dict(aruco_config)
    aruco_runtime_config["segmentation"] = dict(lidar_config.get("segmentation", {}))
    aruco_runtime_config["sync"] = dict(lidar_config.get("sync", {}))

    aruco_provider = ArucoPriorProvider(aruco_runtime_config)
    lidar_source = RosbagLidarSource(lidar_config)

    clustering_cfg = lidar_config["clustering"]
    tracker_cfg = lidar_config["tracker"]
    logging_cfg = lidar_config.get("logging", {})

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
    centerpoint_provider = CenterPointCandidateProvider(lidar_config)
    detector_cfg = lidar_config.get("detector", {})
    detector_source_mode = str(detector_cfg.get("source", "cluster"))
    detector_dedup_distance_m = float(detector_cfg.get("dedup_distance_m", 0.4))
    if bool(logging_cfg.get("print_summary_every_frame", False)) and centerpoint_provider.load_error is not None:
        print(f"CenterPoint provider unavailable; fallback to cluster candidates. reason={centerpoint_provider.load_error}")
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
    reconnect_streak = 0
    local_roi_recovery_countdown = 0

    try:
        while True:
            ok, points, timestamp = lidar_source.read()
            if not ok or points is None or timestamp is None:
                break

            raw_count = int(points.shape[0])
            roi_points = crop_points(points, lidar_config.get("roi", {}))
            point_count_after_roi = int(roi_points.shape[0])
            no_ego_points = remove_ego_vehicle_points(roi_points, lidar_config.get("ego_vehicle_filter", {}))
            point_count_after_ego_filter = int(no_ego_points.shape[0])
            processed = remove_ground(no_ego_points, lidar_config.get("ground_removal", {}))
            point_count_after_ground_removal = int(processed.shape[0])
            aruco_prior = aruco_provider.get_prior(timestamp)
            aruco_debug = aruco_provider.get_debug_state()
            use_local_roi = local_roi_recovery_countdown <= 0
            processed = apply_aruco_local_roi(
                processed,
                aruco_prior,
                lidar_config.get("roi", {}),
                reconnect_streak=reconnect_streak,
                force_disable=not use_local_roi,
            )
            point_count_after_aruco_local_roi = int(processed.shape[0])

            clusters = euclidean_clusters(
                points=processed,
                tolerance=float(clustering_cfg["tolerance_m"]),
                min_points=int(clustering_cfg["min_cluster_points"]),
                max_points=int(clustering_cfg["max_cluster_points"]),
            )
            cluster_count = len(clusters)
            geometric_candidates = filter_candidates(clusters, lidar_config["person_cluster"])
            centerpoint_candidates = centerpoint_provider.detect(processed, timestamp)
            candidates, candidate_source = select_lidar_candidates(
                source_mode=detector_source_mode,
                geometric_candidates=geometric_candidates,
                centerpoint_candidates=centerpoint_candidates,
                dedup_distance_m=detector_dedup_distance_m,
            )

            aruco_rescue_candidates = build_aruco_rescue_candidates(
                clusters=clusters,
                person_cfg=lidar_config["person_cluster"],
                aruco_prior=aruco_prior,
            )
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
            segmentation_cluster_count = len(segmented_clusters)
            segmentation_candidates = filter_candidates(segmented_clusters, lidar_config["person_cluster"])
            aruco_nearest_distance_m = nearest_candidate_distance(candidates, aruco_prior)
            prioritized_candidates = []
            if aruco_prior is not None:
                prioritized_candidates = merge_prioritized_candidates(
                    segmentation_candidates=segmentation_candidates,
                    candidates=merge_prioritized_candidates(
                        segmentation_candidates=aruco_rescue_candidates,
                        candidates=candidates,
                        dedup_distance_m=float(lidar_config.get("segmentation", {}).get("candidate_dedup_distance_m", 0.2)),
                    ),
                    dedup_distance_m=float(lidar_config.get("segmentation", {}).get("candidate_dedup_distance_m", 0.2)),
                )
            rescue_candidate_keys = {candidate_key(candidate) for candidate in aruco_rescue_candidates}

            def motion_cfg_for(candidate: ClusterCandidate) -> dict[str, Any]:
                return build_motion_cfg(
                    tracker_cfg=tracker_cfg,
                    aruco_prior=aruco_prior,
                    reconnect_streak=reconnect_streak,
                    is_rescue_candidate=candidate_key(candidate) in rescue_candidate_keys,
                )

            aruco_selected_candidate = select_candidate_with_aruco_prior(
                candidates=prioritized_candidates,
                aruco_prior=aruco_prior,
                gating_distance_m=float(args.aruco_gating_distance),
                tracker_state=tracker.last_state,
                tracker_cfg=tracker_cfg,
            )
            if aruco_selected_candidate is not None and not is_candidate_motion_consistent(
                candidate=aruco_selected_candidate,
                last_state=tracker.last_state,
                timestamp=timestamp,
                tracker_cfg=motion_cfg_for(aruco_selected_candidate),
                aruco_prior=aruco_prior,
            ):
                aruco_selected_candidate = None

            if aruco_selected_candidate is not None:
                tracker_candidates = [aruco_selected_candidate]
            else:
                tracker_candidates = [
                    candidate
                    for candidate in prioritized_candidates
                    if is_candidate_motion_consistent(
                        candidate=candidate,
                        last_state=tracker.last_state,
                        timestamp=timestamp,
                        tracker_cfg=motion_cfg_for(candidate),
                        aruco_prior=aruco_prior,
                    )
                ]
            tracker_candidate_keys = {candidate_key(candidate) for candidate in tracker_candidates}
            aruco_selected_key = None if aruco_selected_candidate is None else candidate_key(aruco_selected_candidate)
            geo_candidate_keys = {candidate_key(candidate) for candidate in candidates}
            seg_candidate_keys = {candidate_key(candidate) for candidate in segmentation_candidates}
            candidate_diagnostics: list[dict[str, Any]] = []
            for candidate in prioritized_candidates:
                key = candidate_key(candidate)
                reject_reason = candidate_motion_reject_reason(
                    candidate=candidate,
                    last_state=tracker.last_state,
                    timestamp=timestamp,
                    tracker_cfg=motion_cfg_for(candidate),
                    aruco_prior=aruco_prior,
                )
                distance_to_aruco_m = None
                if aruco_prior is not None:
                    distance_to_aruco_m = float(np.linalg.norm(candidate.footpoint[:2] - aruco_prior.position_lidar[:2]))
                candidate_diagnostics.append(
                    {
                        "candidate": format_candidate(candidate),
                        "source_geometric": key in geo_candidate_keys,
                        "source_segmentation": key in seg_candidate_keys,
                        "source_aruco_rescue": key in rescue_candidate_keys,
                        "distance_to_aruco_m": None if distance_to_aruco_m is None else round(distance_to_aruco_m, 6),
                        "is_aruco_selected": key == aruco_selected_key,
                        "is_tracker_input": key in tracker_candidate_keys,
                        "motion_reject_reason": reject_reason,
                    }
                )

            state = tracker.update(timestamp=timestamp, candidates=tracker_candidates)
            used_aruco_fallback = False
            if state is None and tracker.last_state is not None:
                hold_duration_s = float(tracker_cfg.get("hold_without_match_duration_s", 0.35))
                dt = float(timestamp - tracker.last_state.timestamp)
                if 0.0 <= dt <= hold_duration_s:
                    predicted_position = tracker.last_state.position + tracker.last_state.velocity * dt
                    state = TrackState(
                        timestamp=timestamp,
                        position=predicted_position,
                        velocity=tracker.last_state.velocity.copy(),
                        source="predicted",
                        candidate=None,
                        track_quality=tracker.track_quality,
                        track_lifecycle=tracker.track_lifecycle,
                    )
                    tracker.last_state = state
            if state is None and aruco_prior is not None:
                fallback_velocity = np.zeros(3, dtype=np.float64)
                if tracker.last_state is not None:
                    dt = float(timestamp - tracker.last_state.timestamp)
                    if dt > 1e-3:
                        fallback_velocity = (aruco_prior.position_lidar - tracker.last_state.position) / dt
                state = TrackState(
                    timestamp=timestamp,
                    position=aruco_prior.position_lidar.copy(),
                    velocity=fallback_velocity,
                    source="aruco_fallback",
                    candidate=None,
                    track_quality=tracker.track_quality,
                    track_lifecycle=tracker.track_lifecycle,
                )
                tracker.last_state = state
                used_aruco_fallback = True

            if aruco_prior is not None and int(len(tracker_candidates)) == 0:
                local_cfg = lidar_config.get("roi", {}).get("local_with_aruco", {})
                trigger_after = int(local_cfg.get("recovery_trigger_frames", 2))
                recovery_frames = int(local_cfg.get("recovery_disable_frames", 4))
                if reconnect_streak >= trigger_after:
                    local_roi_recovery_countdown = max(local_roi_recovery_countdown, recovery_frames)
            if local_roi_recovery_countdown > 0:
                local_roi_recovery_countdown -= 1

            if aruco_prior is not None:
                if state is not None and state.source == "observed":
                    reconnect_streak = 0
                else:
                    reconnect_streak += 1
            else:
                reconnect_streak = 0

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
                geometric_candidates=geometric_candidates,
                centerpoint_candidates=centerpoint_candidates,
                candidate_source=candidate_source,
                segmentation_candidates=segmentation_candidates,
                aruco_rescue_candidates=aruco_rescue_candidates,
                prioritized_candidates=prioritized_candidates,
                tracker_candidates=tracker_candidates,
                candidate_diagnostics=candidate_diagnostics,
                cluster_count=cluster_count,
                segmentation_cluster_count=segmentation_cluster_count,
                aruco_prior=aruco_prior,
                aruco_selected_candidate=aruco_selected_candidate,
                aruco_nearest_distance_m=aruco_nearest_distance_m,
                camera_timestamp=aruco_debug.frame_timestamp,
                sync_dt_ms=aruco_debug.sync_dt_ms,
                segmentation_point_count=int(segmented_points.shape[0]),
                projected_point_count=int(projected_point_count),
                segmentation_candidate_count=len(segmentation_candidates),
                used_aruco_fallback=used_aruco_fallback,
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
                fallback_text = "true" if used_aruco_fallback else "false"
                print(
                    f"time={timestamp:.3f} raw={raw_count} filtered={processed.shape[0]} "
                    f"candidates={len(candidates)} source={candidate_source} aruco={aruco_text} aruco_match={selected_text} "
                    f"aruco_fallback={fallback_text} fused={fused_text}"
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
