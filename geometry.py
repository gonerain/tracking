from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def as_point3(point: Iterable[float]) -> np.ndarray:
    point_array = np.asarray(point, dtype=np.float64).reshape(-1)
    if point_array.size != 3:
        raise ValueError(f"Expected a 3D point, got shape {point_array.shape}")
    return point_array


def as_rotation_matrix(rotation: Iterable[Iterable[float]]) -> np.ndarray:
    rotation_matrix = np.asarray(rotation, dtype=np.float64)
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {rotation_matrix.shape}")
    return rotation_matrix


def as_translation(translation: Iterable[float]) -> np.ndarray:
    translation_vec = np.asarray(translation, dtype=np.float64).reshape(-1)
    if translation_vec.size != 3:
        raise ValueError(f"Expected a 3D translation, got shape {translation_vec.shape}")
    return translation_vec


def rvec_tvec_to_matrix(rvec: Iterable[float], tvec: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    rvec_array = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec_array = np.asarray(tvec, dtype=np.float64).reshape(3)
    rotation_matrix, _ = cv2.Rodrigues(rvec_array)
    return rotation_matrix, tvec_array


def transform_point(point: Iterable[float], rotation: Iterable[Iterable[float]], translation: Iterable[float]) -> np.ndarray:
    point_vec = as_point3(point)
    rotation_matrix = as_rotation_matrix(rotation)
    translation_vec = as_translation(translation)
    return rotation_matrix @ point_vec + translation_vec


def camera_point_to_target_frame(
    point_camera: Iterable[float],
    rotation_target_from_camera: Iterable[Iterable[float]],
    translation_target_from_camera: Iterable[float],
) -> np.ndarray:
    return transform_point(
        point=point_camera,
        rotation=rotation_target_from_camera,
        translation=translation_target_from_camera,
    )


def project_camera_point_to_image(
    point_camera: Iterable[float],
    camera_matrix: Iterable[Iterable[float]],
    dist_coeffs: Iterable[float] | None = None,
) -> np.ndarray:
    point = as_point3(point_camera).reshape(1, 1, 3)
    if point[0, 0, 2] <= 0:
        raise ValueError("Point must be in front of the camera (z > 0)")

    camera_matrix_np = np.asarray(camera_matrix, dtype=np.float64)
    if camera_matrix_np.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 camera matrix, got {camera_matrix_np.shape}")

    if dist_coeffs is None:
        dist_coeffs_np = np.zeros((5, 1), dtype=np.float64)
    else:
        dist_coeffs_np = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    image_points, _ = cv2.projectPoints(
        objectPoints=point,
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.zeros((3, 1), dtype=np.float64),
        cameraMatrix=camera_matrix_np,
        distCoeffs=dist_coeffs_np,
    )
    return image_points.reshape(2)
