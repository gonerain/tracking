#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fusion tracking for every bag in a folder.")
    parser.add_argument(
        "--bag-dir",
        default="data/2026-04-20_whampoa-suburban-walk-001/rosbag",
        help="Folder containing .bag files.",
    )
    parser.add_argument("--pattern", default="*.bag", help="Bag filename glob pattern.")
    parser.add_argument("--aruco-config", default="configs/aruco_config.yaml", help="Base ArUco config.")
    parser.add_argument("--lidar-config", default="configs/lidar_config.yaml", help="Base lidar config.")
    parser.add_argument(
        "--output-dir",
        default="outputs/fusion_batch",
        help="Batch output directory. Per-bag and merged results are written here.",
    )
    parser.add_argument(
        "--ie-path",
        default=None,
        help="Override IE/GT txt path. Defaults to lidar config ie_pose.path.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip bags with existing target_world_positions.jsonl.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing later bags after a failure.")
    parser.add_argument("--parallel", type=int, default=1, metavar="N", help="Number of bags to process in parallel (default: 1).")
    parser.add_argument("--export-kml", action="store_true", help="Export per-bag and merged KML after JSONL outputs are created.")
    parser.add_argument("--kml-sample-step", type=int, default=500, help="Sample step passed to KML exporter.")
    parser.add_argument(
        "--no-auto-calibrate-target-offsets",
        action="store_true",
        help="Disable one-time folder scan for ArUco group offsets.",
    )
    parser.add_argument(
        "--auto-offset-scan-frames-per-bag",
        type=int,
        default=300,
        help="Maximum camera frames to scan per bag while estimating ArUco group offsets.",
    )
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def bag_sort_key(path: Path) -> tuple[str, str]:
    return (path.stem, path.name)


def set_nested(mapping: dict[str, Any], keys: list[str], value: Any) -> None:
    current: dict[str, Any] = mapping
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def prepare_configs(
    bag_path: Path,
    bag_out_dir: Path,
    aruco_config: dict[str, Any],
    lidar_config: dict[str, Any],
) -> tuple[Path, Path]:
    aruco_runtime = json.loads(json.dumps(aruco_config))
    lidar_runtime = json.loads(json.dumps(lidar_config))

    set_nested(aruco_runtime, ["input", "rosbag", "bag_path"], str(bag_path))
    set_nested(lidar_runtime, ["input", "rosbag", "bag_path"], str(bag_path))

    set_nested(aruco_runtime, ["debug", "aruco_output_dir"], str(bag_out_dir / "aruco_debug"))
    set_nested(aruco_runtime, ["video", "output_path"], str(bag_out_dir / "tracking_output.mp4"))
    set_nested(lidar_runtime, ["debug", "fusion_output_dir"], str(bag_out_dir / "fusion_debug"))

    config_dir = bag_out_dir / "configs"
    aruco_path = config_dir / "aruco_config.yaml"
    lidar_path = config_dir / "lidar_config.yaml"
    write_yaml(aruco_path, aruco_runtime)
    write_yaml(lidar_path, lidar_runtime)
    return aruco_path, lidar_path


def auto_calibrate_target_offsets(
    bags: list[Path],
    aruco_config: dict[str, Any],
    frames_per_bag: int,
) -> dict[int, list[float]]:
    from core.detect_aruco import FrameSource, build_detector_context, detect_markers
    from core.tracking import estimate_target_offsets_from_results, get_target_id_groups, get_target_ids

    tracking_cfg = dict(aruco_config.get("tracking", {}))
    target_id_groups = get_target_id_groups(tracking_cfg)
    target_ids = get_target_ids(tracking_cfg)
    learned: dict[int, list[float]] = {}
    if frames_per_bag <= 0:
        return learned

    for bag_path in bags:
        runtime_config = json.loads(json.dumps(aruco_config))
        set_nested(runtime_config, ["input", "rosbag", "bag_path"], str(bag_path))
        context = build_detector_context(runtime_config)
        source = FrameSource(runtime_config)
        try:
            for _ in range(frames_per_bag):
                ok, frame, _timestamp = source.read()
                if not ok or frame is None:
                    break
                results = detect_markers(frame, context)
                offsets = estimate_target_offsets_from_results(results, target_id_groups, min_visible_markers=3)
                for target_id, offset in offsets.items():
                    if int(target_id) not in learned:
                        learned[int(target_id)] = [round(float(value), 6) for value in offset.tolist()]
                if all(target_id in learned for target_id in target_ids):
                    return learned
        finally:
            source.release()
    return learned


def apply_calibrated_offsets(aruco_config: dict[str, Any], offsets: dict[int, list[float]]) -> None:
    tracking_cfg = aruco_config.setdefault("tracking", {})
    tracking_cfg["target_offsets_m"] = {
        str(target_id): offset
        for target_id, offset in sorted(offsets.items())
    }
    tracking_cfg["auto_target_offsets"] = {
        "enabled": False,
        "source": "folder_pre_scan",
    }


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else f"{ROOT}{os.pathsep}{existing_pythonpath}"
    with log_path.open("w", encoding="utf-8", errors="ignore") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        log_file.flush()
        process = subprocess.run(command, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT, env=env)
    return int(process.returncode)


def merge_jsonl(inputs: list[tuple[str, Path]], output_path: Path) -> int:
    records: list[dict[str, Any]] = []
    for bag_name, path in inputs:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                item.setdefault("source_bag", bag_name)
                records.append(item)

    records.sort(key=lambda item: float(item.get("timestamp", 0.0)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(records)


def merge_fusion_json(inputs: list[tuple[str, Path]], output_path: Path) -> int:
    records: list[dict[str, Any]] = []
    for bag_name, path in inputs:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if isinstance(item, dict):
                item.setdefault("source_bag", bag_name)
                records.append(item)

    records.sort(key=lambda item: float(item.get("timestamp", 0.0)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def export_kml_func(input_jsonl: Path, output_kml: Path, gt_path: str | None, sample_step: int, log_path: Path) -> int:
    command = [
        sys.executable,
        "tools/export_target_trajectory_kml.py",
        "--input",
        str(input_jsonl),
        "--output",
        str(output_kml),
        "--sample-step",
        str(sample_step),
    ]
    if gt_path:
        command.extend(["--gt-txt", gt_path])
    return run_command(command, log_path)


def process_single_bag(
    bag_path: Path,
    bag_out_dir: Path,
    aruco_config: dict[str, Any],
    lidar_config: dict[str, Any],
    ie_path: str | None,
    export_kml: bool,
    kml_sample_step: int,
    gt_path: str | None,
    skip_existing: bool,
) -> dict[str, Any]:
    """Process a single bag file. Returns a manifest entry dict."""
    bag_name = bag_path.stem
    target_world_jsonl = bag_out_dir / "target_world_positions.jsonl"
    fusion_json = bag_out_dir / "fusion_tracking_log.json"
    status = "pending"
    returncode = None

    if skip_existing and target_world_jsonl.exists():
        status = "skipped_existing"
    else:
        aruco_runtime_path, lidar_runtime_path = prepare_configs(bag_path, bag_out_dir, aruco_config, lidar_config)
        command = [
            sys.executable,
            "core/fusion_tracking.py",
            "--aruco-config",
            str(aruco_runtime_path),
            "--lidar-config",
            str(lidar_runtime_path),
            "--output-json",
            str(fusion_json),
            "--output-target-world-json",
            str(target_world_jsonl),
        ]
        if ie_path:
            command.extend(["--ie-path", str(ie_path)])

        returncode = run_command(command, bag_out_dir / "run.log")
        status = "completed" if returncode == 0 else "failed"

    if export_kml and target_world_jsonl.exists():
        export_kml_func(
            target_world_jsonl,
            bag_out_dir / "target_trajectory.kml",
            gt_path,
            kml_sample_step,
            bag_out_dir / "kml_export.log",
        )

    return {
        "bag": str(bag_path),
        "bag_name": bag_path.name,
        "status": status,
        "returncode": returncode,
        "output_dir": str(bag_out_dir),
        "target_world_jsonl": str(target_world_jsonl),
        "fusion_json": str(fusion_json),
        "run_log": str(bag_out_dir / "run.log"),
        "kml": str(bag_out_dir / "target_trajectory.kml") if export_kml else None,
    }


def main() -> None:
    args = parse_args()
    bag_dir = repo_path(args.bag_dir)
    output_dir = repo_path(args.output_dir)
    aruco_config_path = repo_path(args.aruco_config)
    lidar_config_path = repo_path(args.lidar_config)

    # Write batch log to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_log_path = output_dir / "batch_run.log"
    file_handler = logging.FileHandler(str(batch_log_path), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("batch")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # Redirect print to logger
    _orig_print = print
    import builtins
    def _log_print(*args_p, **kwargs_p):
        msg = " ".join(str(a) for a in args_p)
        logger.info(msg)
    builtins.print = _log_print

    bags = sorted(bag_dir.glob(args.pattern), key=bag_sort_key)
    if not bags:
        raise FileNotFoundError(f"No bags found: {bag_dir}/{args.pattern}")

    aruco_config = load_yaml(aruco_config_path)
    lidar_config = load_yaml(lidar_config_path)
    ie_cfg = lidar_config.get("ie_pose", {})
    gt_path = args.ie_path
    if gt_path is None and isinstance(ie_cfg, dict):
        gt_path = str(ie_cfg.get("path", ""))

    calibrated_offsets: dict[int, list[float]] = {}
    if not args.no_auto_calibrate_target_offsets:
        print("auto-calibrating ArUco target offsets from bag folder...")
        calibrated_offsets = auto_calibrate_target_offsets(
            bags,
            aruco_config,
            frames_per_bag=int(args.auto_offset_scan_frames_per_bag),
        )
        apply_calibrated_offsets(aruco_config, calibrated_offsets)
        print(f"  calibrated_offsets={{{', '.join(f'{k}: {v}' for k, v in sorted(calibrated_offsets.items()))}}}")

    per_bag_jsonl: list[tuple[str, Path]] = []
    per_bag_fusion_json: list[tuple[str, Path]] = []
    manifest: list[dict[str, Any]] = []
    failures: list[tuple[str, int]] = []

    # Build job list
    jobs: list[dict[str, Any]] = []
    for bag_path in bags:
        bag_name = bag_path.stem
        bag_out_dir = output_dir / "bags" / bag_name
        target_world_jsonl = bag_out_dir / "target_world_positions.jsonl"
        fusion_json = bag_out_dir / "fusion_tracking_log.json"
        per_bag_jsonl.append((bag_path.name, target_world_jsonl))
        per_bag_fusion_json.append((bag_path.name, fusion_json))
        jobs.append({
            "bag_path": bag_path,
            "bag_out_dir": bag_out_dir,
            "aruco_config": aruco_config,
            "lidar_config": lidar_config,
            "ie_path": args.ie_path,
            "export_kml": args.export_kml,
            "kml_sample_step": args.kml_sample_step,
            "gt_path": gt_path,
            "skip_existing": args.skip_existing,
        })

    max_workers = max(int(args.parallel), 1)

    if max_workers == 1:
        # Sequential processing
        for index, job in enumerate(jobs, start=1):
            print(f"[{index}/{len(jobs)}] bag={job['bag_path']}")
            result = process_single_bag(**job)
            print(f"  status={result['status']}")
            manifest.append(result)
            if result["status"] == "failed":
                failures.append((result["bag_name"], result["returncode"]))
                if not args.continue_on_error:
                    break
    else:
        # Parallel processing
        print(f"Processing {len(jobs)} bags with {max_workers} parallel workers...")
        future_to_index: dict[Any, int] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for index, job in enumerate(jobs):
                future = executor.submit(process_single_bag, **job)
                future_to_index[future] = index

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "bag": str(jobs[index]["bag_path"]),
                        "bag_name": jobs[index]["bag_path"].name,
                        "status": "failed",
                        "returncode": -1,
                        "output_dir": str(jobs[index]["bag_out_dir"]),
                        "target_world_jsonl": str(jobs[index]["bag_out_dir"] / "target_world_positions.jsonl"),
                        "fusion_json": str(jobs[index]["bag_out_dir"] / "fusion_tracking_log.json"),
                        "run_log": str(jobs[index]["bag_out_dir"] / "run.log"),
                        "kml": None,
                    }
                    print(f"  [{index+1}/{len(jobs)}] EXCEPTION bag={jobs[index]['bag_path'].name}: {exc}")
                print(f"  [{index+1}/{len(jobs)}] {result['bag_name']}: {result['status']}")
                if result["status"] == "failed":
                    failures.append((result["bag_name"], result["returncode"]))

        # Collect results in original order
        results_by_bag: dict[str, dict[str, Any]] = {}
        for future in future_to_index:
            try:
                r = future.result()
                results_by_bag[r["bag"]] = r
            except Exception:
                pass
        for job in jobs:
            bag_key = str(job["bag_path"])
            if bag_key in results_by_bag:
                manifest.append(results_by_bag[bag_key])

    merged_dir = output_dir / "merged"
    merged_jsonl = merged_dir / "target_world_positions.jsonl"
    merged_count = merge_jsonl(per_bag_jsonl, merged_jsonl)
    merged_fusion_count = merge_fusion_json(per_bag_fusion_json, merged_dir / "fusion_tracking_log.json")

    print(f"merged target jsonl: {merged_jsonl} records={merged_count}")
    if merged_fusion_count:
        print(f"merged fusion json: {merged_dir / 'fusion_tracking_log.json'} records={merged_fusion_count}")

    if args.export_kml and merged_count > 0:
        kml_code = export_kml_func(
            merged_jsonl,
            merged_dir / "target_trajectory.kml",
            gt_path,
            args.kml_sample_step,
            merged_dir / "kml_export.log",
        )
        if kml_code == 0:
            print(f"merged kml: {merged_dir / 'target_trajectory.kml'}")
        else:
            print(f"merged kml export failed returncode={kml_code}; log={merged_dir / 'kml_export.log'}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "bag_dir": str(bag_dir),
                "bag_count": len(bags),
                "merged_target_world_jsonl": str(merged_jsonl),
                "merged_fusion_json": str(merged_dir / "fusion_tracking_log.json"),
                "merged_kml": str(merged_dir / "target_trajectory.kml") if args.export_kml else None,
                "calibrated_target_offsets_m": {str(k): v for k, v in sorted(calibrated_offsets.items())},
                "bags": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")

    if failures:
        print("failed bags:")
        for bag_name, returncode in failures:
            print(f"  {bag_name}: returncode={returncode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
