from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detect_aruco import build_detector, detect_markers, draw_results, load_config, open_camera


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
            prev_state = self.history[-2]
            last_state = self.history[-1]
            dt = last_state.timestamp - prev_state.timestamp
            if dt > 1e-6:
                velocity_camera = (last_state.position_camera - prev_state.position_camera) / dt
                velocity_target = (last_state.position_target - prev_state.position_target) / dt
                if prev_state.corners is not None and last_state.corners is not None:
                    velocity_corners = (last_state.corners - prev_state.corners) / dt

        predicted_corners = None
        if self.last_observed.corners is not None:
            if velocity_corners is None:
                predicted_corners = self.last_observed.corners.copy()
            else:
                predicted_corners = self.last_observed.corners + velocity_corners * dt_since_observed

        predicted_camera = self.last_observed.position_camera + velocity_camera * dt_since_observed
        predicted_target = self.last_observed.position_target + velocity_target * dt_since_observed
        return TrackState(
            timestamp=timestamp,
            position_camera=predicted_camera,
            position_target=predicted_target,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a specified ArUco id with short-term prediction and JSON logging.")
    parser.add_argument("--config", default="aruco_config.yaml", help="Path to YAML config.")
    return parser.parse_args()


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
        "corners_px": None if state.corners is None else np.round(state.corners.reshape(-1, 2), 3).tolist(),
    }


def select_target(results: list[dict[str, Any]], target_id: int) -> dict[str, Any] | None:
    for result in results:
        if int(result["id"]) == int(target_id):
            return result
    return None


def draw_predicted_box(frame: np.ndarray, state: TrackState | None, target_id: int) -> np.ndarray:
    output = frame.copy()
    if state is None:
        text = f"target_id={target_id} status=lost"
        cv2.putText(output, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return output

    color = (0, 255, 0) if state.source == "observed" else (0, 255, 255)
    text = f"target_id={target_id} status={state.source} pos={np.round(state.position_target, 3).tolist()}"
    cv2.putText(output, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if state.source == "predicted" and state.corners is not None:
        corners = np.round(state.corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [corners], isClosed=True, color=color, thickness=2)
        center = np.round(state.corners.reshape(-1, 2).mean(axis=0)).astype(int)
        cv2.circle(output, tuple(center.tolist()), 5, color, -1)
        cv2.putText(output, "predicted", (int(center[0]) + 8, int(center[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return output


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tracking_cfg = config.get("tracking", {})
    logging_cfg = config.get("logging", {})
    display_cfg = config.get("display", {})
    target_id = int(tracking_cfg["target_id"])
    history_size = int(tracking_cfg.get("history_size", 10))
    max_prediction_duration_s = float(tracking_cfg.get("max_prediction_duration_s", 1.0))
    display_enabled = bool(display_cfg.get("enabled", False))
    window_name = str(display_cfg.get("window_name", "tracking"))

    tracker = TargetTracker(history_size=history_size, max_prediction_duration_s=max_prediction_duration_s)
    logger = JsonLogger(
        output_path=logging_cfg.get("output_json", "tracking_log.json"),
        flush_every_frame=bool(logging_cfg.get("flush_every_frame", True)),
    )

    _, _, camera_matrix, dist_coeffs = build_detector(config)
    capture = open_camera(config)

    if display_enabled:
        print("Press q to quit.")
    else:
        print("Display disabled. Press Ctrl+C to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            frame_timestamp = time.time()
            results = detect_markers(frame, config)
            target = select_target(results, target_id)

            if target is not None:
                state = tracker.update(
                    timestamp=frame_timestamp,
                    position_camera=np.asarray(target["center_in_camera_m"], dtype=np.float64),
                    position_target=np.asarray(target["center_in_target_m"], dtype=np.float64),
                    corners=np.asarray(target["corners"], dtype=np.float64).reshape(-1, 2),
                )
                visible = True
            else:
                state = tracker.predict(frame_timestamp)
                visible = False

            record = format_record(frame_timestamp, target_id, state, visible)
            logger.append(record)
            print(json.dumps(record, ensure_ascii=False))

            if display_enabled:
                vis = draw_results(frame, results, camera_matrix, dist_coeffs)
                vis = draw_predicted_box(vis, state, target_id)
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    finally:
        capture.release()
        logger.flush()
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
