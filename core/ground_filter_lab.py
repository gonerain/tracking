from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.lidar_tracking import (
    RosbagLidarSource,
    crop_points,
    euclidean_clusters,
    load_config,
    remove_ego_vehicle_points,
    remove_ground,
)


def parse_args() -> argparse.Namespace:
    # Keep CLI close to fusion/lidar configs so parameter A/B is fast.
    parser = argparse.ArgumentParser(description="Ground-removal playground extracted from fusion pipeline.")
    parser.add_argument("--config", default="configs/lidar_config.yaml", help="Path to lidar config YAML.")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum number of frames to process.")
    parser.add_argument(
        "--output-jsonl",
        default="outputs/ground_filter_lab.jsonl",
        help="Per-frame statistics output.",
    )

    parser.add_argument(
        "--method",
        choices=["z_threshold", "adaptive_grid", "adaptive_plane"],
        default=None,
        help="Override ground removal method.",
    )
    parser.add_argument("--z-min", type=float, default=None, help="Override ground removal z_min.")
    parser.add_argument("--z-max", type=float, default=None, help="Override ground removal z_max.")
    parser.add_argument("--cell-size-m", type=float, default=None, help="Override adaptive_grid cell_size_m.")
    parser.add_argument("--ground-quantile", type=float, default=None, help="Override adaptive_grid ground_quantile.")
    parser.add_argument("--clearance-m", type=float, default=None, help="Override adaptive_grid clearance_m.")
    parser.add_argument("--min-points-per-cell", type=int, default=None, help="Override adaptive_grid min_points_per_cell.")
    parser.add_argument("--fallback-clearance-m", type=float, default=None, help="Override adaptive_grid fallback_clearance_m.")

    parser.add_argument("--cluster-tolerance-m", type=float, default=None, help="Override clustering tolerance.")
    parser.add_argument("--cluster-min-points", type=int, default=None, help="Override clustering min points.")
    parser.add_argument("--cluster-max-points", type=int, default=None, help="Override clustering max points.")
    return parser.parse_args()


def override_ground_cfg(base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    # Runtime overrides avoid editing YAML for each experiment.
    cfg = dict(base)
    if args.method is not None:
        cfg["method"] = args.method
    if args.z_min is not None:
        cfg["z_min"] = float(args.z_min)
    if args.z_max is not None:
        cfg["z_max"] = float(args.z_max)
    if args.cell_size_m is not None:
        cfg["cell_size_m"] = float(args.cell_size_m)
    if args.ground_quantile is not None:
        cfg["ground_quantile"] = float(args.ground_quantile)
    if args.clearance_m is not None:
        cfg["clearance_m"] = float(args.clearance_m)
    if args.min_points_per_cell is not None:
        cfg["min_points_per_cell"] = int(args.min_points_per_cell)
    if args.fallback_clearance_m is not None:
        cfg["fallback_clearance_m"] = float(args.fallback_clearance_m)
    return cfg


def override_cluster_cfg(base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    # Cluster statistics are a practical proxy for ground-removal quality.
    cfg = dict(base)
    if args.cluster_tolerance_m is not None:
        cfg["tolerance_m"] = float(args.cluster_tolerance_m)
    if args.cluster_min_points is not None:
        cfg["min_cluster_points"] = int(args.cluster_min_points)
    if args.cluster_max_points is not None:
        cfg["max_cluster_points"] = int(args.cluster_max_points)
    return cfg


def run() -> None:
    args = parse_args()
    config = load_config(args.config)
    source = RosbagLidarSource(config)
    output_path = Path(str(args.output_jsonl))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ground_cfg = override_ground_cfg(config.get("ground_removal", {}), args)
    clustering_cfg = override_cluster_cfg(config.get("clustering", {}), args)
    roi_cfg = config.get("roi", {})
    ego_cfg = config.get("ego_vehicle_filter", {})

    records: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while frame_index < int(args.max_frames):
            ok, points, timestamp = source.read()
            if not ok or points is None or timestamp is None:
                break

            raw = points.copy()
            # Match the same preprocessing order used in fusion_tracking.
            roi = crop_points(raw, roi_cfg)
            no_ego = remove_ego_vehicle_points(roi, ego_cfg)
            filtered = remove_ground(no_ego, ground_cfg)
            clusters = euclidean_clusters(
                points=filtered,
                tolerance=float(clustering_cfg.get("tolerance_m", 0.45)),
                min_points=int(clustering_cfg.get("min_cluster_points", 6)),
                max_points=int(clustering_cfg.get("max_cluster_points", 5000)),
            )

            raw_z = raw[:, 2] if raw.shape[0] > 0 else np.empty((0,), dtype=np.float64)
            filtered_z = filtered[:, 2] if filtered.shape[0] > 0 else np.empty((0,), dtype=np.float64)
            record = {
                "frame_index": frame_index,
                "timestamp": float(timestamp),
                "point_count_raw": int(raw.shape[0]),
                "point_count_after_roi": int(roi.shape[0]),
                "point_count_after_ego_filter": int(no_ego.shape[0]),
                "point_count_after_ground_removal": int(filtered.shape[0]),
                # Fraction retained after ground removal; larger is not always better.
                "ground_keep_ratio": 0.0 if no_ego.shape[0] == 0 else round(float(filtered.shape[0]) / float(no_ego.shape[0]), 6),
                # Too many post-ground clusters often means leftover ground/noise.
                "cluster_count_after_ground_removal": int(len(clusters)),
                "raw_z_p10_p50_p90": None if raw_z.size == 0 else np.round(np.percentile(raw_z, [10, 50, 90]), 4).tolist(),
                "filtered_z_p10_p50_p90": None if filtered_z.size == 0 else np.round(np.percentile(filtered_z, [10, 50, 90]), 4).tolist(),
            }
            records.append(record)
            frame_index += 1
    finally:
        source.release()

    payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in records)
    if payload:
        payload += "\n"
    output_path.write_text(payload, encoding="utf-8")
    print(f"ground_filter_lab done: frames={len(records)} output={output_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
