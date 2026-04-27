from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.detect_aruco import FrameSource, build_detector_context, detect_markers, draw_results, load_config


@dataclass
class TrackState:
    timestamp: float
    position_camera: np.ndarray
    position_target: np.ndarray
    corners: np.ndarray | None
    source: str


class TargetTracker:
    def __init__(self, history_size: int, max_prediction_duration_s: float, allow_prediction: bool = False) -> None:
        self.history: deque[TrackState] = deque(maxlen=history_size)
        self.max_prediction_duration_s = float(max_prediction_duration_s)
        self.allow_prediction = bool(allow_prediction)
        self.last_observed: TrackState | None = None

    def update(
        self,
        timestamp: float,
        position_camera: np.ndarray,
        position_target: np.ndarray,
        corners: np.ndarray,
    ) -> TrackState:
        state = TrackState(
            timestamp=timestamp,
            position_camera=np.asarray(position_camera, dtype=np.float64).copy(),
            position_target=np.asarray(position_target, dtype=np.float64).copy(),
            corners=np.asarray(corners, dtype=np.float64).copy(),
            source="observed",
        )
        self.history.append(state)
        self.last_observed = state
        return state

    def predict(self, timestamp: float) -> TrackState | None:
        if not self.allow_prediction:
            return None
        if self.last_observed is None:
            return None

        dt_since_observed = timestamp - self.last_observed.timestamp
        if dt_since_observed > self.max_prediction_duration_s:
            return None

        velocity_camera = np.zeros(3, dtype=np.float64)
        velocity_target = np.zeros(3, dtype=np.float64)
        velocity_corners = None

        if len(self.history) >= 2:
            previous_state = self.history[-2]
            last_state = self.history[-1]
            dt = last_state.timestamp - previous_state.timestamp
            if dt > 1e-6:
                velocity_camera = (last_state.position_camera - previous_state.position_camera) / dt
                velocity_target = (last_state.position_target - previous_state.position_target) / dt
                if previous_state.corners is not None and last_state.corners is not None:
                    velocity_corners = (last_state.corners - previous_state.corners) / dt

        predicted_corners = None
        if self.last_observed.corners is not None:
            predicted_corners = self.last_observed.corners.copy()
            if velocity_corners is not None:
                predicted_corners = predicted_corners + velocity_corners * dt_since_observed

        return TrackState(
            timestamp=timestamp,
            position_camera=self.last_observed.position_camera + velocity_camera * dt_since_observed,
            position_target=self.last_observed.position_target + velocity_target * dt_since_observed,
            corners=predicted_corners,
            source="predicted",
        )


class JsonLogger:
    def __init__(self, output_path: str | Path, flush_every_frame: bool) -> None:
        self.output_path = Path(output_path)
        self.flush_every_frame = flush_every_frame
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self.flush_every_frame:
            self.flush()

    def flush(self) -> None:
        self.output_path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")


class VideoLogger:
    def __init__(self, config: dict[str, Any], frame_shape: tuple[int, int, int]) -> None:
        video_cfg = config.get("video", {})
        self.writer: cv2.VideoWriter | None = None
        if not bool(video_cfg.get("save_enabled", False)):
            return

        output_path = Path(str(video_cfg.get("output_path", "tracking_output.mp4")))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(video_cfg.get("fps", config.get("input", {}).get("camera", {}).get("fps", 30)))
        fourcc_text = str(video_cfg.get("fourcc", "mp4v"))
        if len(fourcc_text) != 4:
            raise ValueError(f"video.fourcc must be 4 characters, got: {fourcc_text}")

        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a specified ArUco id from camera or rosbag.")
    parser.add_argument("--config", default="configs/aruco_config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def _normalize_target_id_group(target_ids: Any) -> list[int]:
    if not isinstance(target_ids, (list, tuple)):
        return [int(target_ids)]
    return [int(target_id) for target_id in target_ids]


def get_target_id_groups(tracking_cfg: dict[str, Any]) -> list[list[int]]:
    target_id_groups_cfg = tracking_cfg.get("target_id_groups")
    if target_id_groups_cfg is not None:
        return [_normalize_target_id_group(target_ids) for target_ids in target_id_groups_cfg]

    target_ids_cfg = tracking_cfg.get("target_ids")
    if target_ids_cfg is not None:
        if isinstance(target_ids_cfg, (list, tuple)) and target_ids_cfg and isinstance(target_ids_cfg[0], (list, tuple)):
            return [_normalize_target_id_group(target_ids) for target_ids in target_ids_cfg]
        return [_normalize_target_id_group(target_ids_cfg)]
    return [[int(tracking_cfg["target_id"])]]


def get_target_ids(tracking_cfg: dict[str, Any]) -> list[int]:
    flattened: list[int] = []
    for target_group in get_target_id_groups(tracking_cfg):
        for target_id in target_group:
            if target_id not in flattened:
                flattened.append(target_id)
    return flattened


def _target_group_label(target_ids: list[int]) -> str:
    return str(target_ids[0]) if len(target_ids) == 1 else "[" + ",".join(str(target_id) for target_id in target_ids) + "]"


def target_ids_label(target_ids: list[int] | list[list[int]]) -> str:
    if target_ids and isinstance(target_ids[0], (list, tuple)):
        return " | ".join(_target_group_label([int(target_id) for target_id in target_group]) for target_group in target_ids)
    return ",".join(str(target_id) for target_id in target_ids)


def get_target_offsets(tracking_cfg: dict[str, Any], target_ids: list[int]) -> dict[int, np.ndarray]:
    offsets_cfg = tracking_cfg.get("target_offsets_m", {})
    offsets: dict[int, np.ndarray] = {}
    for target_id in target_ids:
        raw_value = offsets_cfg.get(str(target_id), offsets_cfg.get(target_id))
        if raw_value is None:
            continue
        offsets[target_id] = np.asarray(raw_value, dtype=np.float64).reshape(3)
    return offsets


def estimate_group_center_from_markers(marker_centers: list[np.ndarray]) -> np.ndarray | None:
    if len(marker_centers) >= 4:
        return np.mean(np.stack(marker_centers, axis=0), axis=0)
    if len(marker_centers) == 3:
        # For three visible corners of a square, the longest pair is the diagonal;
        # its midpoint is the square center.
        best_pair: tuple[int, int] | None = None
        best_distance = -1.0
        for i in range(3):
            for j in range(i + 1, 3):
                distance = float(np.linalg.norm(marker_centers[i] - marker_centers[j]))
                if distance > best_distance:
                    best_distance = distance
                    best_pair = (i, j)
        if best_pair is None:
            return None
        return 0.5 * (marker_centers[best_pair[0]] + marker_centers[best_pair[1]])
    return None


def estimate_square_layout_from_visible_markers(
    marker_centers_by_id: dict[int, np.ndarray],
    target_id_list: list[int],
) -> dict[Any, np.ndarray]:
    visible_ids = list(marker_centers_by_id.keys())
    if len(visible_ids) >= 4:
        visible_ids = visible_ids[:4]
        center = np.mean(np.stack([marker_centers_by_id[target_id] for target_id in visible_ids], axis=0), axis=0)
        layout: dict[Any, np.ndarray] = {target_id: marker_centers_by_id[target_id] for target_id in visible_ids}
        layout["__center__"] = center
        return layout

    if len(visible_ids) == 3 and len(target_id_list) == 4:
        best_pair: tuple[int, int] | None = None
        best_distance = -1.0
        for i in range(3):
            for j in range(i + 1, 3):
                distance = float(np.linalg.norm(marker_centers_by_id[visible_ids[i]] - marker_centers_by_id[visible_ids[j]]))
                if distance > best_distance:
                    best_distance = distance
                    best_pair = (i, j)
        if best_pair is None:
            return {}
        diagonal_ids = {visible_ids[best_pair[0]], visible_ids[best_pair[1]]}
        right_angle_id = next(target_id for target_id in visible_ids if target_id not in diagonal_ids)
        missing_ids = [target_id for target_id in target_id_list if target_id not in marker_centers_by_id]
        if len(missing_ids) != 1:
            return {}
        center = 0.5 * (
            marker_centers_by_id[visible_ids[best_pair[0]]] + marker_centers_by_id[visible_ids[best_pair[1]]]
        )
        missing_center = 2.0 * center - marker_centers_by_id[right_angle_id]
        layout = dict(marker_centers_by_id)
        layout[missing_ids[0]] = missing_center
        layout["__center__"] = center  # type: ignore[index]
        return layout

    return {}


def estimate_target_offsets_from_results(
    results: list[dict[str, Any]],
    target_id_groups: list[list[int]],
    min_visible_markers: int = 3,
) -> dict[int, np.ndarray]:
    offsets: dict[int, np.ndarray] = {}
    min_visible_markers = max(int(min_visible_markers), 3)
    for target_id_list in target_id_groups:
        matched = [
            result
            for result in results
            if int(result["id"]) in set(target_id_list) and "center_in_target_m" in result
        ]
        if len(matched) < min_visible_markers:
            continue
        marker_centers_by_id = {
            int(result["id"]): np.asarray(result["center_in_target_m"], dtype=np.float64).reshape(3)
            for result in matched
        }
        layout = estimate_square_layout_from_visible_markers(marker_centers_by_id, target_id_list)
        group_center = layout.pop("__center__", None)
        if group_center is None:
            continue
        for target_id, marker_center in layout.items():
            offsets[int(target_id)] = np.asarray(group_center, dtype=np.float64) - marker_center
    return offsets


def _normalize_target_groups(target_ids: int | list[int] | list[list[int]]) -> list[list[int]]:
    if isinstance(target_ids, int):
        return [[int(target_ids)]]
    if not target_ids:
        return []
    if isinstance(target_ids[0], (list, tuple)):
        return [[int(target_id) for target_id in target_group] for target_group in target_ids]
    return [[int(target_id) for target_id in target_ids]]


def _build_target_result(
    matched: list[dict[str, Any]],
    target_id_list: list[int],
    group_index: int,
    target_offsets: dict[int, np.ndarray] | None,
) -> dict[str, Any]:
    offsets = target_offsets or {}
    camera_centers = [np.asarray(result["center_in_camera_m"], dtype=np.float64) for result in matched]
    visible_marker_centers = {
        int(result["id"]): np.asarray(result["center_in_target_m"], dtype=np.float64).reshape(3)
        for result in matched
    }
    visible_ids = [int(result["id"]) for result in matched]
    known_offset_ids = [target_id for target_id in target_id_list if target_id in offsets]
    can_complete_group = len(known_offset_ids) == len(target_id_list) and any(target_id in offsets for target_id in visible_ids)
    inferred_ids: list[int] = []
    completed_marker_centers: dict[int, np.ndarray] = dict(visible_marker_centers)

    if can_complete_group:
        center_estimates = [
            visible_marker_centers[target_id] + offsets[target_id]
            for target_id in visible_ids
            if target_id in offsets
        ]
        target_center = np.mean(np.stack(center_estimates, axis=0), axis=0)
        for target_id in target_id_list:
            if target_id not in completed_marker_centers:
                completed_marker_centers[target_id] = target_center - offsets[target_id]
                inferred_ids.append(int(target_id))
    else:
        target_centers = [
            visible_marker_centers[target_id] + offsets.get(target_id, np.zeros(3, dtype=np.float64))
            for target_id in visible_ids
        ]
        target_center = np.mean(np.stack(target_centers, axis=0), axis=0)

    projected_centers = [np.asarray(result["center_projected_px"], dtype=np.float64) for result in matched]
    corners_list = [np.asarray(result["corners"], dtype=np.float64) for result in matched]
    corner_points = np.concatenate(corners_list, axis=0)
    min_corner = corner_points.min(axis=0)
    max_corner = corner_points.max(axis=0)
    combined_corners = np.array(
        [
            [min_corner[0], min_corner[1]],
            [max_corner[0], min_corner[1]],
            [max_corner[0], max_corner[1]],
            [min_corner[0], max_corner[1]],
        ],
        dtype=np.float64,
    )

    return {
        "id": target_id_list[0] if len(target_id_list) == 1 else target_id_list,
        "target_ids": target_id_list,
        "target_group_index": group_index,
        "visible_ids": visible_ids,
        "inferred_ids": inferred_ids,
        "corners": combined_corners,
        "center_in_camera_m": np.mean(np.stack(camera_centers, axis=0), axis=0),
        "center_in_target_m": target_center,
        "center_projected_px": np.mean(np.stack(projected_centers, axis=0), axis=0),
        "completed_marker_centers_in_target_m": {
            str(target_id): completed_marker_centers[target_id]
            for target_id in sorted(completed_marker_centers)
        },
        "offsets_applied": bool(can_complete_group),
    }


def _target_selection_key(target: dict[str, Any]) -> tuple[float, float, float]:
    target_ids = list(target.get("target_ids", []))
    visible_ids = list(target.get("visible_ids", []))
    corners = np.asarray(target["corners"], dtype=np.float64)
    span = corners.max(axis=0) - corners.min(axis=0)
    area_px = float(span[0] * span[1])
    coverage = float(len(visible_ids)) / float(max(len(target_ids), 1))
    return (float(len(visible_ids)), coverage, area_px)


def select_target(results: list[dict[str, Any]], target_ids: int | list[int] | list[list[int]], target_offsets: dict[int, np.ndarray] | None = None) -> dict[str, Any] | None:
    candidates = select_targets(results, target_ids, target_offsets)
    if not candidates:
        return None
    return max(candidates, key=_target_selection_key)

def select_targets(
    results: list[dict[str, Any]],
    target_ids: int | list[int] | list[list[int]],
    target_offsets: dict[int, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    target_groups = _normalize_target_groups(target_ids)
    candidates: list[dict[str, Any]] = []
    for group_index, target_id_list in enumerate(target_groups):
        target_id_set = set(target_id_list)
        matched = [result for result in results if int(result["id"]) in target_id_set]
        if not matched:
            continue
        candidates.append(_build_target_result(matched, target_id_list, group_index, target_offsets))
    return candidates


def format_record(
    timestamp: float,
    target_id_groups: list[list[int]],
    selected_target_ids: list[int],
    state: TrackState | None,
    visible: bool,
    visible_ids: list[int],
) -> dict[str, Any]:
    iso_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().isoformat()
    flat_target_ids = [target_id for target_group in target_id_groups for target_id in target_group]
    if state is None:
        return {
            "timestamp": timestamp,
            "datetime": iso_time,
            "target_ids": flat_target_ids,
            "target_id_groups": target_id_groups,
            "selected_target_ids": selected_target_ids,
            "visible_ids": visible_ids,
            "visible": visible,
            "tracking_state": "lost",
            "position_camera_m": None,
            "position_target_m": None,
            "corners_px": None,
        }

    return {
        "timestamp": timestamp,
        "datetime": iso_time,
        "target_ids": flat_target_ids,
        "target_id_groups": target_id_groups,
        "selected_target_ids": selected_target_ids,
        "visible_ids": visible_ids,
        "visible": visible,
        "tracking_state": state.source,
        "position_camera_m": np.round(state.position_camera, 6).tolist(),
        "position_target_m": np.round(state.position_target, 6).tolist(),
        "corners_px": None if state.corners is None else np.round(state.corners, 3).tolist(),
    }


def draw_tracking_overlay(
    frame: np.ndarray,
    target_label: str,
    state: TrackState | None,
    visible_ids: list[int] | None = None,
    selected_target_ids: list[int] | None = None,
) -> np.ndarray:
    output = frame.copy()
    visible_text = "none" if not visible_ids else ",".join(str(target_id) for target_id in visible_ids)
    selected_text = "none" if not selected_target_ids else _target_group_label(selected_target_ids)
    if state is None:
        cv2.putText(
            output,
            f"targets={target_label} selected={selected_text} visible_ids={visible_text} status=lost",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2,
        )
        return output

    color = (0, 255, 0) if state.source == "observed" else (0, 255, 255)
    text = (
        f"targets={target_label} selected={selected_text} visible_ids={visible_text} "
        f"status={state.source} pos={np.round(state.position_target, 3).tolist()}"
    )
    cv2.putText(output, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if state.source == "predicted" and state.corners is not None:
        corners = np.round(state.corners).astype(np.int32).reshape(-1, 1, 2)
        center = np.round(state.corners.mean(axis=0)).astype(int)
        cv2.polylines(output, [corners], isClosed=True, color=color, thickness=2)
        cv2.circle(output, tuple(center.tolist()), 5, color, -1)
        cv2.putText(output, "predicted", (int(center[0]) + 8, int(center[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return output


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    context = build_detector_context(config)
    frame_source = FrameSource(config)

    tracking_cfg = config.get("tracking", {})
    logging_cfg = config.get("logging", {})
    display_cfg = config.get("display", {})
    target_id_groups = get_target_id_groups(tracking_cfg)
    target_offsets = get_target_offsets(tracking_cfg, get_target_ids(tracking_cfg))
    target_label = target_ids_label(target_id_groups)
    tracker = TargetTracker(
        history_size=int(tracking_cfg.get("history_size", 10)),
        max_prediction_duration_s=float(tracking_cfg.get("max_prediction_duration_s", 1.0)),
        allow_prediction=bool(tracking_cfg.get("allow_prediction", False)),
    )
    logger = JsonLogger(
        output_path=logging_cfg.get("output_json", "tracking_log.json"),
        flush_every_frame=bool(logging_cfg.get("flush_every_frame", True)),
    )
    display_enabled = bool(display_cfg.get("enabled", False))
    window_name = str(display_cfg.get("window_name", "tracking"))

    ok, first_frame, first_timestamp = frame_source.read()
    if not ok or first_frame is None or first_timestamp is None:
        frame_source.release()
        raise RuntimeError("Failed to read initial frame from input source")
    video_logger = VideoLogger(config, first_frame.shape)

    if display_enabled:
        print("Press q to quit.")
    else:
        print("Display disabled. Press Ctrl+C to quit.")

    try:
        frame = first_frame
        timestamp = first_timestamp
        while True:
            results = detect_markers(frame, context)
            target = select_target(results, target_id_groups, target_offsets)
            visible_ids = [] if target is None else list(target.get("visible_ids", []))
            selected_target_ids = [] if target is None else list(target.get("target_ids", []))

            if target is not None:
                state = tracker.update(
                    timestamp=timestamp,
                    position_camera=np.asarray(target["center_in_camera_m"], dtype=np.float64),
                    position_target=np.asarray(target["center_in_target_m"], dtype=np.float64),
                    corners=np.asarray(target["corners"], dtype=np.float64),
                )
                visible = True
            else:
                state = None
                visible = False

            record = format_record(timestamp, target_id_groups, selected_target_ids, state, visible, visible_ids)
            logger.append(record)
            print(json.dumps(record, ensure_ascii=False))

            vis = draw_results(frame, results, context)
            vis = draw_tracking_overlay(vis, target_label, state, visible_ids, selected_target_ids)
            video_logger.write(vis)

            if display_enabled:
                cv2.imshow(window_name, vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            ok, frame, next_timestamp = frame_source.read()
            if not ok or frame is None or next_timestamp is None:
                break
            timestamp = next_timestamp
    finally:
        frame_source.release()
        video_logger.release()
        logger.flush()
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
