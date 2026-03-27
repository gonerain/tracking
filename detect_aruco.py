from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from geometry import camera_point_to_target_frame, project_camera_point_to_image, rvec_tvec_to_matrix


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


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_detector(config: dict[str, Any]) -> tuple[cv2.aruco.ArucoDetector, float, np.ndarray, np.ndarray]:
    aruco_cfg = config["aruco"]
    intrinsics_cfg = config["camera"]["intrinsics"]

    dictionary_name = aruco_cfg["dictionary"]
    if dictionary_name not in ARUCO_DICT_MAP:
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dictionary_name])
    detector_params = cv2.aruco.DetectorParameters()

    detector_cfg = aruco_cfg.get("detector", {})
    refinement_map = {
        "NONE": cv2.aruco.CORNER_REFINE_NONE,
        "SUBPIX": cv2.aruco.CORNER_REFINE_SUBPIX,
        "CONTOUR": cv2.aruco.CORNER_REFINE_CONTOUR,
        "APRILTAG": cv2.aruco.CORNER_REFINE_APRILTAG,
    }
    for key, value in detector_cfg.items():
        if key == "cornerRefinementMethod":
            value = refinement_map[value]
        setattr(detector_params, key, value)

    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    marker_length_m = float(aruco_cfg["marker_length_m"])
    camera_matrix = np.asarray(intrinsics_cfg["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(intrinsics_cfg["dist_coeffs"], dtype=np.float64)
    return detector, marker_length_m, camera_matrix, dist_coeffs


def open_camera(config: dict[str, Any]) -> cv2.VideoCapture:
    source_cfg = config["camera"]["source"]
    capture = cv2.VideoCapture(int(source_cfg.get("device_id", 0)))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open camera device {source_cfg.get('device_id', 0)}")

    if "width" in source_cfg:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(source_cfg["width"]))
    if "height" in source_cfg:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(source_cfg["height"]))
    if "fps" in source_cfg:
        capture.set(cv2.CAP_PROP_FPS, int(source_cfg["fps"]))
    return capture


def estimate_marker_poses(
    corners: list[np.ndarray],
    marker_length_m: float,
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
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            corner.reshape(-1, 2),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            raise RuntimeError("solvePnP failed for detected marker")
        poses.append((rvec.reshape(3), tvec.reshape(3)))
    return poses


def detect_markers(image: np.ndarray, config: dict[str, Any]) -> list[dict[str, Any]]:
    detector, marker_length_m, camera_matrix, dist_coeffs = build_detector(config)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return []

    poses = estimate_marker_poses(corners, marker_length_m, camera_matrix, dist_coeffs)
    target_frame_cfg = config.get("target_frame", {})
    rotation_target_from_camera = np.asarray(target_frame_cfg.get("rotation_matrix", np.eye(3)), dtype=np.float64)
    translation_target_from_camera = np.asarray(target_frame_cfg.get("translation_m", np.zeros(3)), dtype=np.float64)

    results = []
    for marker_id, corner, (rvec, tvec) in zip(ids.flatten().tolist(), corners, poses):
        rotation_marker_from_camera, translation_marker_from_camera = rvec_tvec_to_matrix(rvec, tvec)
        center_camera = translation_marker_from_camera
        center_target = camera_point_to_target_frame(
            center_camera,
            rotation_target_from_camera,
            translation_target_from_camera,
        )
        center_pixel = project_camera_point_to_image(center_camera, camera_matrix, dist_coeffs)

        results.append(
            {
                "id": marker_id,
                "corners": corner,
                "rvec_marker_from_camera": rvec,
                "tvec_marker_from_camera_m": tvec,
                "rotation_marker_from_camera": rotation_marker_from_camera,
                "center_in_camera_m": center_camera,
                "center_in_target_m": center_target,
                "center_projected_px": center_pixel,
            }
        )
    return results


def draw_results(frame: np.ndarray, results: list[dict[str, Any]], camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    output = frame.copy()
    for result in results:
        cv2.polylines(
            output,
            [result["corners"].astype(np.int32)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.drawFrameAxes(
            output,
            camera_matrix,
            dist_coeffs,
            result["rvec_marker_from_camera"].reshape(3, 1),
            result["tvec_marker_from_camera_m"].reshape(3, 1),
            0.03,
        )

        center_px = tuple(np.round(result["center_projected_px"]).astype(int).tolist())
        cv2.circle(output, center_px, 4, (0, 0, 255), -1)

        text_1 = f"id={result['id']} cam={np.round(result['center_in_camera_m'], 3).tolist()}"
        text_2 = f"target={np.round(result['center_in_target_m'], 3).tolist()}"
        top_left = result["corners"].reshape(-1, 2).min(axis=0).astype(int)
        cv2.putText(output, text_1, (int(top_left[0]), int(top_left[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(output, text_2, (int(top_left[0]), int(top_left[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read camera, detect ArUco, and convert to target frame.")
    parser.add_argument(
        "--config",
        default="aruco_config.yaml",
        help="Path to YAML config with camera source, intrinsics, and target frame transform.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, _, camera_matrix, dist_coeffs = build_detector(config)
    capture = open_camera(config)
    display_cfg = config.get("display", {})
    display_enabled = bool(display_cfg.get("enabled", False))
    window_name = str(display_cfg.get("window_name", "aruco_detection"))

    if display_enabled:
        print("Press q to quit.")
    else:
        print("Display disabled. Press Ctrl+C to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            results = detect_markers(frame, config)
            for result in results:
                print(
                    f"marker_id={result['id']} center_in_camera_m={np.round(result['center_in_camera_m'], 4).tolist()} "
                    f"center_in_target_m={np.round(result['center_in_target_m'], 4).tolist()}"
                )

            if display_enabled:
                vis = draw_results(frame, results, camera_matrix, dist_coeffs)
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    finally:
        capture.release()
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
