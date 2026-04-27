from __future__ import annotations

import argparse
import sys
from typing import Callable

from core.detect_aruco import main as detect_aruco_main
from core.fusion_tracking import main as fusion_tracking_main
from core.ground_filter_lab import main as ground_filter_lab_main
from core.lidar_tracking import main as lidar_tracking_main
from core.tracking import main as tracking_main


EntryPoint = Callable[[], None]
ENTRYPOINTS: dict[str, EntryPoint] = {
    "detect_aruco": detect_aruco_main,
    "tracking": tracking_main,
    "lidar_tracking": lidar_tracking_main,
    "fusion_tracking": fusion_tracking_main,
    "ground_filter_lab": ground_filter_lab_main,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified entrypoint for camera and lidar tracking tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect_aruco", help="Run ArUco detection on camera or rosbag input.")
    detect_parser.add_argument("--config", default="configs/aruco_config.yaml", help="Path to YAML config.")

    tracking_parser = subparsers.add_parser("tracking", help="Track a specified ArUco marker.")
    tracking_parser.add_argument("--config", default="configs/aruco_config.yaml", help="Path to YAML config.")

    lidar_parser = subparsers.add_parser("lidar_tracking", help="Track a person-like cluster from lidar rosbag data.")
    lidar_parser.add_argument("--config", default="configs/lidar_config.yaml", help="Path to YAML config.")

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
    fusion_parser.add_argument(
        "--ie-path",
        default=None,
        help="Optional override path to IE/SPAN/GT text file for world-frame lidar pose.",
    )
    fusion_parser.add_argument(
        "--output-target-world-json",
        default="outputs/target_world_positions.jsonl",
        help="Path to per-frame world target position JSONL output.",
    )

    ground_parser = subparsers.add_parser("ground_filter_lab", help="Run standalone ground-removal lab on lidar rosbag.")
    ground_parser.add_argument("--config", default="configs/lidar_config.yaml", help="Path to lidar YAML config.")
    ground_parser.add_argument("--max-frames", type=int, default=300, help="Maximum number of frames.")
    ground_parser.add_argument("--output-jsonl", default="outputs/ground_filter_lab.jsonl", help="Stats output JSONL.")
    ground_parser.add_argument("--method", choices=["z_threshold", "adaptive_grid"], default=None, help="Override ground method.")
    ground_parser.add_argument("--z-min", type=float, default=None, help="Override z_min.")
    ground_parser.add_argument("--z-max", type=float, default=None, help="Override z_max.")
    ground_parser.add_argument("--cell-size-m", type=float, default=None, help="Override cell_size_m.")
    ground_parser.add_argument("--ground-quantile", type=float, default=None, help="Override ground_quantile.")
    ground_parser.add_argument("--clearance-m", type=float, default=None, help="Override clearance_m.")
    ground_parser.add_argument("--min-points-per-cell", type=int, default=None, help="Override min_points_per_cell.")
    ground_parser.add_argument("--fallback-clearance-m", type=float, default=None, help="Override fallback_clearance_m.")
    ground_parser.add_argument("--cluster-tolerance-m", type=float, default=None, help="Override clustering tolerance.")
    ground_parser.add_argument("--cluster-min-points", type=int, default=None, help="Override clustering min points.")
    ground_parser.add_argument("--cluster-max-points", type=int, default=None, help="Override clustering max points.")

    return parser


def dispatch(command: str) -> EntryPoint:
    try:
        return ENTRYPOINTS[command]
    except KeyError as exc:
        raise ValueError(f"Unsupported command: {command}") from exc


def main() -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args()
    entrypoint = dispatch(args.command)

    # Rebuild argv so the delegated entrypoint can keep using its own argparse.
    if args.command == "fusion_tracking":
        ie_args = [] if args.ie_path is None else ["--ie-path", args.ie_path]
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
            *ie_args,
            "--output-target-world-json",
            args.output_target_world_json,
            *remaining,
        ]
    elif args.command == "ground_filter_lab":
        sys.argv = [
            f"{parser.prog} {args.command}",
            "--config",
            args.config,
            "--max-frames",
            str(args.max_frames),
            "--output-jsonl",
            args.output_jsonl,
            *([] if args.method is None else ["--method", args.method]),
            *([] if args.z_min is None else ["--z-min", str(args.z_min)]),
            *([] if args.z_max is None else ["--z-max", str(args.z_max)]),
            *([] if args.cell_size_m is None else ["--cell-size-m", str(args.cell_size_m)]),
            *([] if args.ground_quantile is None else ["--ground-quantile", str(args.ground_quantile)]),
            *([] if args.clearance_m is None else ["--clearance-m", str(args.clearance_m)]),
            *([] if args.min_points_per_cell is None else ["--min-points-per-cell", str(args.min_points_per_cell)]),
            *([] if args.fallback_clearance_m is None else ["--fallback-clearance-m", str(args.fallback_clearance_m)]),
            *([] if args.cluster_tolerance_m is None else ["--cluster-tolerance-m", str(args.cluster_tolerance_m)]),
            *([] if args.cluster_min_points is None else ["--cluster-min-points", str(args.cluster_min_points)]),
            *([] if args.cluster_max_points is None else ["--cluster-max-points", str(args.cluster_max_points)]),
            *remaining,
        ]
    else:
        sys.argv = [f"{parser.prog} {args.command}", "--config", args.config, *remaining]
    entrypoint()


if __name__ == "__main__":
    main()
