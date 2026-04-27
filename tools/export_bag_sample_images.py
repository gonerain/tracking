#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.detect_aruco import (  # noqa: E402
    FrameSource,
    build_detector_context,
    detect_markers,
    draw_results,
    load_config,
)
from core.tracking import get_target_id_groups  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sampled images from one rosbag with optional ArUco overlay.")
    parser.add_argument("--config", default="configs/aruco_config.yaml", help="Path to ArUco YAML config.")
    parser.add_argument("--bag-path", default=None, help="Override bag path in config.")
    parser.add_argument("--topic", default=None, help="Override image topic in config.")
    parser.add_argument("--output-dir", default="outputs/bag_sample_images", help="Directory to save images and index.")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images to export.")
    parser.add_argument("--start-index", type=int, default=0, help="Frame index to start sampling from.")
    parser.add_argument("--frame-step", type=int, default=30, help="Export one image every N frames.")
    parser.add_argument("--detect-aruco", action="store_true", help="Run ArUco detection and save annotated images.")
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw image too. If --detect-aruco is off, raw image is always saved.",
    )
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality [1,100].")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def find_matching_target_groups(
    marker_ids: list[int],
    target_id_groups: list[list[int]],
) -> list[dict[str, Any]]:
    visible = set(int(marker_id) for marker_id in marker_ids)
    matches: list[dict[str, Any]] = []
    for index, group in enumerate(target_id_groups):
        group_set = set(int(target_id) for target_id in group)
        visible_ids = sorted(int(target_id) for target_id in group if target_id in visible)
        if not visible_ids:
            continue
        matches.append(
            {
                "target_group_index": int(index),
                "target_ids": [int(target_id) for target_id in group],
                "visible_ids": visible_ids,
                "missing_ids": sorted(int(target_id) for target_id in group_set - set(visible_ids)),
            }
        )
    return matches


def main() -> None:
    args = parse_args()

    if args.num_images <= 0:
        raise ValueError("--num-images must be > 0")
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be > 0")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")

    config_path = repo_path(args.config)
    config = load_config(config_path)

    config["input"] = dict(config.get("input", {}))
    config["input"]["type"] = "rosbag"
    config["input"]["rosbag"] = dict(config["input"].get("rosbag", {}))

    if args.bag_path is not None:
        config["input"]["rosbag"]["bag_path"] = str(repo_path(args.bag_path))
    if args.topic is not None:
        config["input"]["rosbag"]["topic"] = str(args.topic)

    bag_path = str(config["input"]["rosbag"].get("bag_path", ""))
    topic = str(config["input"]["rosbag"].get("topic", ""))
    if not bag_path:
        raise ValueError("Missing rosbag path in config/input. Pass --bag-path or set input.rosbag.bag_path.")
    if not topic:
        raise ValueError("Missing image topic in config/input. Pass --topic or set input.rosbag.topic.")

    detector_context = build_detector_context(config) if args.detect_aruco else None
    target_id_groups = get_target_id_groups(dict(config.get("tracking", {})))

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.jsonl"
    source = FrameSource(config)
    exported = 0
    frame_index = 0
    records: list[dict[str, Any]] = []

    try:
        while exported < args.num_images:
            ok, frame, timestamp = source.read()
            if not ok or frame is None or timestamp is None:
                break

            current_index = frame_index
            frame_index += 1

            if current_index < args.start_index:
                continue
            if (current_index - args.start_index) % args.frame_step != 0:
                continue

            ts = float(timestamp)
            ts_text = f"{ts:.3f}"
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat()

            raw_name = f"frame_{exported:03d}_idx_{current_index:06d}_ts_{ts_text}.jpg"
            raw_path = output_dir / raw_name

            marker_ids: list[int] = []
            matched_groups: list[dict[str, Any]] = []
            vis_path: str | None = None

            if args.detect_aruco and detector_context is not None:
                results = detect_markers(frame, detector_context)
                marker_ids = sorted(int(item["id"]) for item in results if "id" in item)
                matched_groups = find_matching_target_groups(marker_ids, target_id_groups)
                vis = draw_results(frame.copy(), results, detector_context)
                vis_name = f"frame_{exported:03d}_idx_{current_index:06d}_ts_{ts_text}_aruco.jpg"
                vis_abs = output_dir / vis_name
                cv2.imwrite(str(vis_abs), vis, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)])
                vis_path = vis_name

            if args.save_raw or not args.detect_aruco:
                cv2.imwrite(str(raw_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)])
                saved_raw_path: str | None = raw_name
            else:
                saved_raw_path = None

            records.append(
                {
                    "export_index": int(exported),
                    "frame_index": int(current_index),
                    "timestamp": ts,
                    "datetime": dt,
                    "bag_path": bag_path,
                    "topic": topic,
                    "raw_image": saved_raw_path,
                    "aruco_image": vis_path,
                    "detected_marker_ids": marker_ids,
                    "matched_target_groups": matched_groups,
                }
            )
            exported += 1
    finally:
        source.release()

    payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in records)
    if payload:
        payload += "\n"
    index_path.write_text(payload, encoding="utf-8")

    summary = {
        "bag_path": bag_path,
        "topic": topic,
        "output_dir": str(output_dir),
        "index_jsonl": str(index_path),
        "requested_images": int(args.num_images),
        "exported_images": int(exported),
        "start_index": int(args.start_index),
        "frame_step": int(args.frame_step),
        "detect_aruco": bool(args.detect_aruco),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
