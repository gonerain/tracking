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

from detect_aruco import FrameSource, build_detector_context, detect_markers, draw_results, load_config


@dataclass
class TrackState:
    timestamp: float
    position_camera: np.ndarray
    position_target: np.ndarray
    corners: np.ndarray | None
    source: str


class TargetTracker:
    def __init__(self, history_size: int, max_prediction_duration_s: float) -> None:
        self.history: deque[TrackState] = deque(maxlen=history_size)
        self.max_prediction_duration_s = float(max_prediction_duration_s)
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
    parser.add_argument("--config", default="aruco_config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def select_target(results: list[dict[str, Any]], target_id: int) -> dict[str, Any] | None:
    for result in results:
        if int(result["id"]) == int(target_id):
            return result
    return None


def format_record(timestamp: float, target_id: int, state: TrackState | None, visible: bool) -> dict[str, Any]:
    iso_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().isoformat()
    if state is None:
        return {
            "timestamp": timestamp,
            "datetime": iso_time,
            "target_id": target_id,
            "visible": visible,
            "tracking_state": "lost",
            "position_camera_m": None,
            "position_target_m": None,
            "corners_px": None,
        }

    return {
        "timestamp": timestamp,
        "datetime": iso_time,
        "target_id": target_id,
        "visible": visible,
        "tracking_state": state.source,
        "position_camera_m": np.round(state.position_camera, 6).tolist(),
        "position_target_m": np.round(state.position_target, 6).tolist(),
        "corners_px": None if state.corners is None else np.round(state.corners, 3).tolist(),
    }


def draw_tracking_overlay(frame: np.ndarray, target_id: int, state: TrackState | None) -> np.ndarray:
    output = frame.copy()
    if state is None:
        cv2.putText(output, f"target_id={target_id} status=lost", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return output

    color = (0, 255, 0) if state.source == "observed" else (0, 255, 255)
    text = f"target_id={target_id} status={state.source} pos={np.round(state.position_target, 3).tolist()}"
    cv2.putText(output, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
    target_id = int(tracking_cfg["target_id"])
    tracker = TargetTracker(
        history_size=int(tracking_cfg.get("history_size", 10)),
        max_prediction_duration_s=float(tracking_cfg.get("max_prediction_duration_s", 1.0)),
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
            target = select_target(results, target_id)

            if target is not None:
                state = tracker.update(
                    timestamp=timestamp,
                    position_camera=np.asarray(target["center_in_camera_m"], dtype=np.float64),
                    position_target=np.asarray(target["center_in_target_m"], dtype=np.float64),
                    corners=np.asarray(target["corners"], dtype=np.float64),
                )
                visible = True
            else:
                state = tracker.predict(timestamp)
                visible = False

            record = format_record(timestamp, target_id, state, visible)
            logger.append(record)
            print(json.dumps(record, ensure_ascii=False))

            vis = draw_results(frame, results, context)
            vis = draw_tracking_overlay(vis, target_id, state)
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
