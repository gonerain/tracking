from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from core.geometry import camera_point_to_target_frame, project_camera_point_to_image, rvec_tvec_to_matrix


ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


@dataclass
class DetectorContext:
    detector: cv2.aruco.ArucoDetector
    marker_length_m: float
    camera_model: str
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rotation_target_from_camera: np.ndarray
    translation_target_from_camera: np.ndarray


class BaseFrameSource:
    def read(self) -> tuple[bool, np.ndarray | None, float | None]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class CameraFrameSource(BaseFrameSource):
    def __init__(self, config: dict[str, Any]) -> None:
        camera_cfg = config["input"]["camera"]
        self.capture = cv2.VideoCapture(int(camera_cfg.get("device_id", 0)))
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open camera device {camera_cfg.get('device_id', 0)}")

        if "width" in camera_cfg:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg["width"]))
        if "height" in camera_cfg:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg["height"]))
        if "fps" in camera_cfg:
            self.capture.set(cv2.CAP_PROP_FPS, int(camera_cfg["fps"]))

    def read(self) -> tuple[bool, np.ndarray | None, float | None]:
        ok, frame = self.capture.read()
        timestamp = time.time() if ok else None
        return ok, frame if ok else None, timestamp

    def release(self) -> None:
        self.capture.release()


class RosbagFrameSource(BaseFrameSource):
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import rosbag  # type: ignore
        except ImportError as exc:
            raise RuntimeError("rosbag support requires ROS1 Python environment with rosbag installed") from exc

        rosbag_cfg = config["input"]["rosbag"]
        self.rosbag = rosbag
        self.bag_path = str(rosbag_cfg["bag_path"])
        self.topic = str(rosbag_cfg["topic"])
        self.start_time_s = rosbag_cfg.get("start_time_s", 0.0)
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

            frame = self._decode_image_msg(msg)
            return True, frame, stamp.to_sec()

    def release(self) -> None:
        if self._bag is not None:
            self._bag.close()
            self._bag = None
            self._iterator = None


class FrameSource:
    def __init__(self, config: dict[str, Any]) -> None:
        input_type = str(config.get("input", {}).get("type", "camera")).lower()
        if input_type == "camera":
            self.impl: BaseFrameSource = CameraFrameSource(config)
        elif input_type == "rosbag":
            self.impl = RosbagFrameSource(config)
        else:
            raise ValueError(f"Unsupported input.type: {input_type}")

    def read(self) -> tuple[bool, np.ndarray | None, float | None]:
        return self.impl.read()

    def release(self) -> None:
        self.impl.release()


class ArucoDebugWriter:
    def __init__(self, config: dict[str, Any]) -> None:
        debug_cfg = config.get("debug", {})
        self.enabled = bool(debug_cfg.get("save_images", False))
        self.save_jsonl = bool(debug_cfg.get("save_jsonl", False))
        self.every_n_frames = max(int(debug_cfg.get("every_n_frames", 1)), 1)
        self.output_dir = Path(str(debug_cfg.get("aruco_output_dir", "outputs/aruco_debug")))
        self.images_dir = self.output_dir / "images"
        self.jsonl_path = self.output_dir / "records.jsonl"
        self.records: list[dict[str, Any]] = []
        if self.enabled or self.save_jsonl:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.enabled:
            self.images_dir.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        frame_index: int,
        timestamp: float,
        frame: np.ndarray,
        vis: np.ndarray,
        results: list[dict[str, Any]],
        selected_target: dict[str, Any] | None,
    ) -> None:
        if frame_index % self.every_n_frames != 0:
            return
        suffix = f"frame_{frame_index:06d}_{timestamp:.3f}"
        if self.enabled:
            cv2.imwrite(str(self.images_dir / f"{suffix}.png"), vis)
        if self.save_jsonl:
            record = {
                "frame_index": int(frame_index),
                "timestamp": float(timestamp),
                "datetime": datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().isoformat(),
                "marker_count": len(results),
                "markers": [
                    {
                        "id": int(result["id"]),
                        "center_in_camera_m": np.round(np.asarray(result["center_in_camera_m"], dtype=np.float64), 6).tolist(),
                        "center_in_target_m": np.round(np.asarray(result["center_in_target_m"], dtype=np.float64), 6).tolist(),
                        "center_projected_px": np.round(np.asarray(result["center_projected_px"], dtype=np.float64), 3).tolist(),
                        "corners_px": np.round(np.asarray(result["corners"], dtype=np.float64), 3).tolist(),
                    }
                    for result in results
                ],
                "selected_target": None
                if selected_target is None
                else {
                    "target_ids": list(selected_target.get("target_ids", [])),
                    "visible_ids": list(selected_target.get("visible_ids", [])),
                    "center_in_target_m": np.round(np.asarray(selected_target["center_in_target_m"], dtype=np.float64), 6).tolist(),
                    "center_projected_px": np.round(np.asarray(selected_target["center_projected_px"], dtype=np.float64), 3).tolist(),
                    "corners_px": np.round(np.asarray(selected_target["corners"], dtype=np.float64), 3).tolist(),
                },
            }
            self.records.append(record)

    def flush(self) -> None:
        if not self.save_jsonl:
            return
        payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in self.records)
        if payload:
            payload += "\n"
        self.jsonl_path.write_text(payload, encoding="utf-8")


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_detector_context(config: dict[str, Any]) -> DetectorContext:
    aruco_cfg = config["aruco"]
    intrinsics_cfg = config["camera"]["intrinsics"]
    target_frame_cfg = config.get("target_frame", {})
    camera_model = str(intrinsics_cfg.get("camera_model", "pinhole")).lower()
    if camera_model in {"equidistantcamera", "fisheye"}:
        camera_model = "equidistant"
    elif camera_model not in {"pinhole", "equidistant"}:
        raise ValueError(f"Unsupported camera model: {intrinsics_cfg.get('camera_model')}")

    dictionary_name = aruco_cfg["dictionary"]
    if dictionary_name not in ARUCO_DICT_MAP:
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dictionary_name])
    detector_params = cv2.aruco.DetectorParameters()
    refinement_map = {
        "NONE": cv2.aruco.CORNER_REFINE_NONE,
        "SUBPIX": cv2.aruco.CORNER_REFINE_SUBPIX,
        "CONTOUR": cv2.aruco.CORNER_REFINE_CONTOUR,
        "APRILTAG": cv2.aruco.CORNER_REFINE_APRILTAG,
    }

    for key, value in aruco_cfg.get("detector", {}).items():
        if key == "cornerRefinementMethod":
            value = refinement_map[value]
        setattr(detector_params, key, value)

    return DetectorContext(
        detector=cv2.aruco.ArucoDetector(dictionary, detector_params),
        marker_length_m=float(aruco_cfg["marker_length_m"]),
        camera_model=camera_model,
        camera_matrix=np.asarray(intrinsics_cfg["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.asarray(intrinsics_cfg["dist_coeffs"], dtype=np.float64),
        rotation_target_from_camera=np.asarray(
            target_frame_cfg.get("rotation_matrix", np.eye(3)),
            dtype=np.float64,
        ),
        translation_target_from_camera=np.asarray(
            target_frame_cfg.get("translation_m", np.zeros(3)),
            dtype=np.float64,
        ),
    )


def estimate_marker_poses(
    corners: list[np.ndarray],
    marker_length_m: float,
    camera_model: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    half_size = marker_length_m / 2.0
    object_points = np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float64,
    )

    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for corner in corners:
        image_points = corner.reshape(-1, 2).astype(np.float64)
        solve_dist_coeffs = dist_coeffs
        if camera_model == "equidistant":
            image_points = cv2.fisheye.undistortPoints(
                image_points.reshape(1, -1, 2),
                K=camera_matrix,
                D=dist_coeffs.reshape(-1, 1),
                P=camera_matrix,
            ).reshape(-1, 2)
            solve_dist_coeffs = np.zeros((4,), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            solve_dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            raise RuntimeError("solvePnP failed for detected marker")
        poses.append((rvec.reshape(3), tvec.reshape(3)))
    return poses


def detect_markers(image: np.ndarray, context: DetectorContext) -> list[dict[str, Any]]:
    corners, ids, _ = context.detector.detectMarkers(image)
    if ids is None:
        return []

    poses = estimate_marker_poses(
        corners,
        context.marker_length_m,
        context.camera_model,
        context.camera_matrix,
        context.dist_coeffs,
    )

    results = []
    for marker_id, corner, (rvec, tvec) in zip(ids.flatten().tolist(), corners, poses):
        rotation_marker_from_camera, translation_marker_from_camera = rvec_tvec_to_matrix(rvec, tvec)
        center_camera = translation_marker_from_camera
        center_target = camera_point_to_target_frame(
            center_camera,
            context.rotation_target_from_camera,
            context.translation_target_from_camera,
        )
        center_pixel = project_camera_point_to_image(
            center_camera,
            context.camera_matrix,
            context.dist_coeffs,
            camera_model=context.camera_model,
        )
        results.append(
            {
                "id": marker_id,
                "corners": corner.reshape(-1, 2),
                "rvec_marker_from_camera": rvec,
                "tvec_marker_from_camera_m": tvec,
                "rotation_marker_from_camera": rotation_marker_from_camera,
                "center_in_camera_m": center_camera,
                "center_in_target_m": center_target,
                "center_projected_px": center_pixel,
            }
        )
    return results


def draw_results(frame: np.ndarray, results: list[dict[str, Any]], context: DetectorContext) -> np.ndarray:
    output = frame.copy()
    for result in results:
        corners = np.round(result["corners"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        if context.camera_model != "equidistant":
            cv2.drawFrameAxes(
                output,
                context.camera_matrix,
                context.dist_coeffs,
                result["rvec_marker_from_camera"].reshape(3, 1),
                result["tvec_marker_from_camera_m"].reshape(3, 1),
                0.03,
            )

        center_px = tuple(np.round(result["center_projected_px"]).astype(int).tolist())
        cv2.circle(output, center_px, 4, (0, 0, 255), -1)

        top_left = result["corners"].min(axis=0).astype(int)
        text_1 = f"id={result['id']} cam={np.round(result['center_in_camera_m'], 3).tolist()}"
        text_2 = f"target={np.round(result['center_in_target_m'], 3).tolist()}"
        cv2.putText(output, text_1, (int(top_left[0]), int(top_left[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(output, text_2, (int(top_left[0]), int(top_left[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read camera or rosbag, detect ArUco, and convert to target frame.")
    parser.add_argument("--config", default="configs/aruco_config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    context = build_detector_context(config)
    frame_source = FrameSource(config)
    debug_writer = ArucoDebugWriter(config)
    display_cfg = config.get("display", {})
    display_enabled = bool(display_cfg.get("enabled", False))
    window_name = str(display_cfg.get("window_name", "aruco_detection"))
    tracking_cfg = config.get("tracking", {})
    target_id_groups: list[list[int]] = []
    target_label = ""
    if tracking_cfg:
        from core.tracking import get_target_id_groups, select_target, target_ids_label

        target_id_groups = get_target_id_groups(tracking_cfg)
        target_label = target_ids_label(target_id_groups)

    if display_enabled:
        print("Press q to quit.")
    else:
        print("Display disabled. Press Ctrl+C to quit.")

    frame_index = 0
    interrupted = False

    try:
        while True:
            ok, frame, timestamp = frame_source.read()
            if not ok or frame is None or timestamp is None:
                break

            results = detect_markers(frame, context)
            selected_target = None
            if target_id_groups:
                selected_target = select_target(results, target_id_groups)
            for result in results:
                print(
                    f"marker_id={result['id']} center_in_camera_m={np.round(result['center_in_camera_m'], 4).tolist()} "
                    f"center_in_target_m={np.round(result['center_in_target_m'], 4).tolist()}"
                )
            if target_id_groups:
                if selected_target is None:
                    print(f"target_groups={target_label} selected_target=none")
                else:
                    selected_ids = list(selected_target.get("target_ids", []))
                    selected_text = str(selected_ids[0]) if len(selected_ids) == 1 else "[" + ",".join(str(target_id) for target_id in selected_ids) + "]"
                    print(
                        f"target_groups={target_label} selected_target={selected_text} "
                        f"visible_ids={selected_target.get('visible_ids', [])} "
                        f"center_in_target_m={np.round(selected_target['center_in_target_m'], 4).tolist()}"
                    )

            vis = draw_results(frame, results, context)
            if target_id_groups:
                selected_text = "none"
                visible_text = "none"
                if selected_target is not None:
                    selected_ids = list(selected_target.get("target_ids", []))
                    selected_text = str(selected_ids[0]) if len(selected_ids) == 1 else "[" + ",".join(str(target_id) for target_id in selected_ids) + "]"
                    visible_ids = list(selected_target.get("visible_ids", []))
                    visible_text = ",".join(str(target_id) for target_id in visible_ids) if visible_ids else "none"
                    corners = np.round(np.asarray(selected_target["corners"], dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
                    center = tuple(np.round(np.asarray(selected_target["center_projected_px"], dtype=np.float64)).astype(int).tolist())
                    cv2.polylines(vis, [corners], isClosed=True, color=(0, 255, 255), thickness=2)
                    cv2.circle(vis, center, 5, (0, 255, 255), -1)
                cv2.putText(
                    vis,
                    f"targets={target_label} selected={selected_text} visible_ids={visible_text}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

            debug_writer.append(
                frame_index=frame_index,
                timestamp=float(timestamp),
                frame=frame,
                vis=vis,
                results=results,
                selected_target=selected_target,
            )

            if display_enabled:
                cv2.imshow(window_name, vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            frame_index += 1
    except KeyboardInterrupt:
        interrupted = True
    finally:
        frame_source.release()
        debug_writer.flush()
        if display_enabled:
            cv2.destroyAllWindows()
    if interrupted:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
