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
        "--batch-config",
        default=None,
        help="YAML config for multiple bag folders. When set, per-folder settings come from config jobs[].",
    )
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


def setup_batch_logger(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    import builtins

    def _log_print(*args_p, **kwargs_p):
        _ = kwargs_p
        msg = " ".join(str(a) for a in args_p)
        logger.info(msg)

    builtins.print = _log_print


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


def run_one_jobset(
    *,
    job_name: str,
    bag_dir: Path,
    output_dir: Path,
    aruco_config_path: Path,
    lidar_config_path: Path,
    pattern: str,
    ie_path_override: str | None,
    skip_existing: bool,
    continue_on_error: bool,
    parallel: int,
    export_kml: bool,
    kml_sample_step: int,
) -> dict[str, Any]:
    bags = sorted(bag_dir.glob(pattern), key=bag_sort_key)
    if not bags:
        raise FileNotFoundError(f"No bags found: {bag_dir}/{pattern}")

    aruco_config = load_yaml(aruco_config_path)
    lidar_config = load_yaml(lidar_config_path)
    ie_cfg = lidar_config.get("ie_pose", {})
    gt_path = ie_path_override
    if gt_path is None and isinstance(ie_cfg, dict):
        gt_path = str(ie_cfg.get("path", ""))

    per_bag_jsonl: list[tuple[str, Path]] = []
    per_bag_fusion_json: list[tuple[str, Path]] = []
    manifest: list[dict[str, Any]] = []
    failures: list[tuple[str, int]] = []

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
            "ie_path": ie_path_override,
            "export_kml": export_kml,
            "kml_sample_step": kml_sample_step,
            "gt_path": gt_path,
            "skip_existing": skip_existing,
        })

    max_workers = max(int(parallel), 1)

    if max_workers == 1:
        for index, job in enumerate(jobs, start=1):
            print(f"[{job_name}] [{index}/{len(jobs)}] bag={job['bag_path']}")
            result = process_single_bag(**job)
            print(f"[{job_name}]   status={result['status']}")
            manifest.append(result)
            if result["status"] == "failed":
                failures.append((result["bag_name"], result["returncode"]))
                if not continue_on_error:
                    break
    else:
        print(f"[{job_name}] Processing {len(jobs)} bags with {max_workers} parallel workers...")
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
                    print(f"[{job_name}]   [{index+1}/{len(jobs)}] EXCEPTION bag={jobs[index]['bag_path'].name}: {exc}")
                print(f"[{job_name}]   [{index+1}/{len(jobs)}] {result['bag_name']}: {result['status']}")
                if result["status"] == "failed":
                    failures.append((result["bag_name"], result["returncode"]))

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

    print(f"[{job_name}] merged target jsonl: {merged_jsonl} records={merged_count}")
    if merged_fusion_count:
        print(f"[{job_name}] merged fusion json: {merged_dir / 'fusion_tracking_log.json'} records={merged_fusion_count}")

    if export_kml and merged_count > 0:
        kml_code = export_kml_func(
            merged_jsonl,
            merged_dir / "target_trajectory.kml",
            gt_path,
            kml_sample_step,
            merged_dir / "kml_export.log",
        )
        if kml_code == 0:
            print(f"[{job_name}] merged kml: {merged_dir / 'target_trajectory.kml'}")
        else:
            print(f"[{job_name}] merged kml export failed returncode={kml_code}; log={merged_dir / 'kml_export.log'}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "job_name": job_name,
                "bag_dir": str(bag_dir),
                "bag_count": len(bags),
                "merged_target_world_jsonl": str(merged_jsonl),
                "merged_fusion_json": str(merged_dir / "fusion_tracking_log.json"),
                "merged_kml": str(merged_dir / "target_trajectory.kml") if export_kml else None,
                "calibrated_target_offsets_m": {},
                "bags": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{job_name}] manifest: {manifest_path}")

    if failures:
        print(f"[{job_name}] failed bags:")
        for bag_name, returncode in failures:
            print(f"[{job_name}]   {bag_name}: returncode={returncode}")

    return {
        "job_name": job_name,
        "bag_dir": str(bag_dir),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "failed_bags": [{"bag_name": name, "returncode": rc} for name, rc in failures],
        "bag_count": len(bags),
    }


def _value_from_job_defaults_cli(job: dict[str, Any], defaults: dict[str, Any], cli: argparse.Namespace, key: str) -> Any:
    if key in job:
        return job[key]
    if key in defaults:
        return defaults[key]
    return getattr(cli, key)


def main() -> None:
    args = parse_args()
    batch_config_path = repo_path(args.batch_config) if args.batch_config else None

    if batch_config_path:
        batch_config = load_yaml(batch_config_path)
        defaults = batch_config.get("defaults", {})
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise ValueError("batch config: defaults must be a mapping")
        raw_jobs = batch_config.get("jobs", [])
        if not isinstance(raw_jobs, list) or not raw_jobs:
            raise ValueError("batch config: jobs must be a non-empty list")

        output_root = batch_config.get("output_root", args.output_dir)
        output_root_path = repo_path(output_root)
        output_root_path.mkdir(parents=True, exist_ok=True)
        setup_batch_logger(output_root_path / "batch_run.log")

        print(f"batch config: {batch_config_path}")
        print(f"output root: {output_root_path}")
        summary: list[dict[str, Any]] = []
        failed_job_names: list[str] = []

        for index, raw_job in enumerate(raw_jobs, start=1):
            if not isinstance(raw_job, dict):
                raise ValueError(f"batch config: jobs[{index-1}] must be a mapping")
            job_name = str(raw_job.get("name", f"job_{index:02d}"))
            bag_dir = raw_job.get("bag_dir")
            if not bag_dir:
                raise ValueError(f"batch config: jobs[{index-1}].bag_dir is required")
            job_output_dir = raw_job.get("output_dir", str(output_root_path / job_name))

            print(f"[batch] start job {index}/{len(raw_jobs)} name={job_name}")
            result = run_one_jobset(
                job_name=job_name,
                bag_dir=repo_path(str(bag_dir)),
                output_dir=repo_path(str(job_output_dir)),
                aruco_config_path=repo_path(str(_value_from_job_defaults_cli(raw_job, defaults, args, "aruco_config"))),
                lidar_config_path=repo_path(str(_value_from_job_defaults_cli(raw_job, defaults, args, "lidar_config"))),
                pattern=str(_value_from_job_defaults_cli(raw_job, defaults, args, "pattern")),
                ie_path_override=_value_from_job_defaults_cli(raw_job, defaults, args, "ie_path"),
                skip_existing=bool(_value_from_job_defaults_cli(raw_job, defaults, args, "skip_existing")),
                continue_on_error=bool(_value_from_job_defaults_cli(raw_job, defaults, args, "continue_on_error")),
                parallel=int(_value_from_job_defaults_cli(raw_job, defaults, args, "parallel")),
                export_kml=bool(_value_from_job_defaults_cli(raw_job, defaults, args, "export_kml")),
                kml_sample_step=int(_value_from_job_defaults_cli(raw_job, defaults, args, "kml_sample_step")),
            )
            summary.append(result)
            if result["failed_bags"]:
                failed_job_names.append(job_name)
                if not bool(_value_from_job_defaults_cli(raw_job, defaults, args, "continue_on_error")):
                    print(f"[batch] stop on failed job: {job_name}")
                    break
            print(f"[batch] done job {index}/{len(raw_jobs)} name={job_name}")

        summary_path = output_root_path / "batch_manifest.json"
        summary_path.write_text(json.dumps({"jobs": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[batch] summary manifest: {summary_path}")
        if failed_job_names:
            raise SystemExit(1)
        return

    bag_dir = repo_path(args.bag_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_batch_logger(output_dir / "batch_run.log")
    result = run_one_jobset(
        job_name="single",
        bag_dir=bag_dir,
        output_dir=output_dir,
        aruco_config_path=repo_path(args.aruco_config),
        lidar_config_path=repo_path(args.lidar_config),
        pattern=args.pattern,
        ie_path_override=args.ie_path,
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
        parallel=args.parallel,
        export_kml=args.export_kml,
        kml_sample_step=args.kml_sample_step,
    )
    if result["failed_bags"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
