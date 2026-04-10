from __future__ import annotations

import argparse
import sys
from typing import Callable

from core.detect_aruco import main as detect_aruco_main
from core.lidar_tracking import main as lidar_tracking_main
from core.segmentation import main as segment_image_main
from core.tracking import main as tracking_main
from fusion_tracking import main as fusion_tracking_main


EntryPoint = Callable[[], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified entrypoint for camera and lidar tracking tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect_aruco", help="Run ArUco detection on camera or rosbag input.")
    detect_parser.add_argument("--config", default="configs/aruco_config.yaml", help="Path to YAML config.")

    tracking_parser = subparsers.add_parser("tracking", help="Track a specified ArUco marker.")
    tracking_parser.add_argument("--config", default="configs/aruco_config.yaml", help="Path to YAML config.")

    lidar_parser = subparsers.add_parser("lidar_tracking", help="Track a person-like cluster from lidar rosbag data.")
    lidar_parser.add_argument("--config", default="configs/lidar_config.yaml", help="Path to YAML config.")

    segment_parser = subparsers.add_parser("segment_image", help="Run the configured segmentation model on a single image.")
    segment_parser.add_argument("--config", default="configs/lidar_config.yaml", help="Path to YAML config.")
    segment_parser.add_argument("--image", required=True, help="Input image path.")
    segment_parser.add_argument("--output", default="outputs/segmentation_preview.png", help="Output preview image path.")

    fusion_parser = subparsers.add_parser("fusion_tracking", help="Fuse ArUco target prior with lidar person tracking.")
    fusion_parser.add_argument("--aruco-config", default="configs/aruco_config.yaml", help="Path to ArUco YAML config.")
    fusion_parser.add_argument("--lidar-config", default="configs/lidar_config.yaml", help="Path to lidar YAML config.")
    fusion_parser.add_argument(
        "--aruco-gating-distance",
        type=float,
        default=1.5,
        help="Maximum distance in meters between ArUco prior and lidar candidate footpoint.",
    )
    fusion_parser.add_argument(
        "--output-json",
        default="outputs/fusion_tracking_log.json",
        help="Path to fused tracking JSON output.",
    )

    return parser


def dispatch(command: str) -> EntryPoint:
    if command == "detect_aruco":
        return detect_aruco_main
    if command == "tracking":
        return tracking_main
    if command == "lidar_tracking":
        return lidar_tracking_main
    if command == "segment_image":
        return segment_image_main
    if command == "fusion_tracking":
        return fusion_tracking_main
    raise ValueError(f"Unsupported command: {command}")


def main() -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args()
    entrypoint = dispatch(args.command)

    # Rebuild argv so the delegated entrypoint can keep using its own argparse.
    if args.command == "fusion_tracking":
        sys.argv = [
            f"{parser.prog} {args.command}",
            "--aruco-config",
            args.aruco_config,
            "--lidar-config",
            args.lidar_config,
            "--aruco-gating-distance",
            str(args.aruco_gating_distance),
            "--output-json",
            args.output_json,
            *remaining,
        ]
    elif args.command == "segment_image":
        sys.argv = [
            f"{parser.prog} {args.command}",
            "--config",
            args.config,
            "--image",
            args.image,
            "--output",
            args.output,
            *remaining,
        ]
    else:
        sys.argv = [f"{parser.prog} {args.command}", "--config", args.config, *remaining]
    entrypoint()


if __name__ == "__main__":
    main()
