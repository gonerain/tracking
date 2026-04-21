#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export target trajectory JSONL to Google Earth KML.")
    parser.add_argument(
        "--input",
        default="outputs/target_world_positions.jsonl",
        help="Input JSONL path from fusion_tracking.",
    )
    parser.add_argument(
        "--output",
        default="outputs/target_trajectory.kml",
        help="Output KML path.",
    )
    parser.add_argument(
        "--altitude-mode",
        choices=["absolute", "clampToGround", "relativeToGround"],
        default="clampToGround",
        help="KML altitude mode.",
    )
    parser.add_argument(
        "--lidar-config",
        default="configs/lidar_config.yaml",
        help="Lidar config path used when recomputing target positions from target_lidar_m.",
    )
    parser.add_argument(
        "--recompute-target-from-lidar",
        action="store_true",
        help="Recompute target LLA from target_lidar_m and ie_pose instead of trusting stored target_world_lla.",
    )
    parser.add_argument(
        "--target-source",
        choices=["auto", "single", "multi_tracks", "aruco_groups"],
        default="auto",
        help="Target trajectory source in JSONL.",
    )
    return parser.parse_args()


def load_target_lla_points(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows: list[tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = item.get("timestamp")
        lla = item.get("target_world_lla")
        if ts is None or not isinstance(lla, list) or len(lla) < 3:
            continue
        try:
            lat = float(lla[0])
            lon = float(lla[1])
            alt = float(lla[2])
            t = float(ts)
        except (TypeError, ValueError):
            continue
        rows.append((t, lon, lat, alt))

    rows.sort(key=lambda row: row[0])
    return rows


def load_span_lla_points(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows: list[tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = item.get("timestamp")
        pose = item.get("ie_pose")
        if ts is None or not isinstance(pose, dict):
            continue
        try:
            lat = float(pose["latitude_deg"])
            lon = float(pose["longitude_deg"])
            alt = float(pose["height_m"])
            t = float(ts)
        except (KeyError, TypeError, ValueError):
            continue
        rows.append((t, lon, lat, alt))

    rows.sort(key=lambda row: row[0])
    return rows


def load_synced_target_span_points(
    path: Path,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    target_rows: list[tuple[float, float, float, float]] = []
    span_rows: list[tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = item.get("timestamp")
        target_lla = item.get("target_world_lla")
        pose = item.get("ie_pose")
        if ts is None or not isinstance(target_lla, list) or len(target_lla) < 3 or not isinstance(pose, dict):
            continue
        try:
            t = float(ts)
            t_lat = float(target_lla[0])
            t_lon = float(target_lla[1])
            t_alt = float(target_lla[2])
            s_lat = float(pose["latitude_deg"])
            s_lon = float(pose["longitude_deg"])
            s_alt = float(pose["height_m"])
        except (KeyError, TypeError, ValueError):
            continue
        target_rows.append((t, t_lon, t_lat, t_alt))
        span_rows.append((t, s_lon, s_lat, s_alt))
    return target_rows, span_rows


def load_synced_target_span_points_recomputed(
    path: Path,
    lidar_config_path: Path,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    from core import fusion_tracking as ft

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if not lidar_config_path.exists():
        raise FileNotFoundError(f"Lidar config not found: {lidar_config_path}")

    lidar_config = yaml.safe_load(lidar_config_path.read_text(encoding="utf-8"))
    ft.configure_span_lidar_extrinsics(lidar_config)

    target_rows: list[tuple[float, float, float, float]] = []
    span_rows: list[tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = item.get("timestamp")
        pose = item.get("ie_pose")
        target_lidar = item.get("target_lidar_m")
        if ts is None or not isinstance(pose, dict):
            continue
        try:
            t = float(ts)
            s_lat = float(pose["latitude_deg"])
            s_lon = float(pose["longitude_deg"])
            s_alt = float(pose["height_m"])
            roll_deg = float(pose["roll_deg"])
            pitch_deg = float(pose["pitch_deg"])
            heading_deg = float(pose["heading_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        span_rows.append((t, s_lon, s_lat, s_alt))
        if not isinstance(target_lidar, list) or len(target_lidar) < 3:
            continue
        try:
            target_lidar_vec = np.asarray(target_lidar[:3], dtype=np.float64)
        except (TypeError, ValueError):
            continue

        target_body = ft.lidar_to_ie_body(target_lidar_vec)
        rot_enu_from_body = ft.rotation_matrix_from_ie(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            heading_deg=heading_deg,
        )
        span_ecef = ft._geodetic_to_ecef(s_lat, s_lon, s_alt)
        enu_to_ecef = ft._enu_to_ecef_matrix(s_lat, s_lon)
        target_ecef = span_ecef + enu_to_ecef @ (rot_enu_from_body @ target_body)
        t_lat, t_lon, t_alt = ft._ecef_to_geodetic(target_ecef)
        target_rows.append((t, t_lon, t_lat, t_alt))
    return target_rows, span_rows


def write_kml(
    target_points: list[tuple[float, float, float, float]],
    span_points: list[tuple[float, float, float, float]],
    out_path: Path,
    altitude_mode: str,
) -> None:
    if len(target_points) < 2:
        raise ValueError(f"Not enough valid target points for trajectory: {len(target_points)}")
    if len(span_points) < 2:
        raise ValueError(f"Not enough valid SPAN points for trajectory: {len(span_points)}")

    target_coords_line = "\n".join(f"{lon:.9f},{lat:.9f},{alt:.3f}" for _, lon, lat, alt in target_points)
    span_coords_line = "\n".join(f"{lon:.9f},{lat:.9f},{alt:.3f}" for _, lon, lat, alt in span_points)
    first = target_points[0]
    last = target_points[-1]
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Target and SPAN Trajectories</name>
    <Style id="trajLine">
      <LineStyle>
        <color>ff3c78d8</color>
        <width>3</width>
      </LineStyle>
    </Style>
    <Style id="spanLine">
      <LineStyle>
        <color>ff0000ff</color>
        <width>5</width>
      </LineStyle>
    </Style>
    <Style id="startPoint">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>1.0</scale>
      </IconStyle>
    </Style>
    <Style id="endPoint">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.0</scale>
      </IconStyle>
    </Style>
    <Placemark>
      <name>Target Track</name>
      <styleUrl>#trajLine</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>
{target_coords_line}
        </coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>SPAN Track</name>
      <styleUrl>#spanLine</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>
{span_coords_line}
        </coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>Start</name>
      <styleUrl>#startPoint</styleUrl>
      <Point>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>{first[1]:.9f},{first[2]:.9f},{first[3]:.3f}</coordinates>
      </Point>
    </Placemark>
    <Placemark>
      <name>End</name>
      <styleUrl>#endPoint</styleUrl>
      <Point>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>{last[1]:.9f},{last[2]:.9f},{last[3]:.3f}</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(kml, encoding="utf-8")


def _track_color_hex(track_index: int) -> str:
    # KML color is aabbggrr
    palette = [
        "ff3c78d8",  # orange-ish
        "ff00a5ff",  # cyan
        "ff32cd32",  # lime
        "ffebce87",  # sky blue
        "ff7b68ee",  # medium slate
        "ff1493ff",  # deep pink
    ]
    return palette[track_index % len(palette)]


def load_multi_target_tracks(
    path: Path,
    source: str = "auto",
) -> tuple[dict[str, list[tuple[float, float, float, float]]], list[tuple[float, float, float, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    track_rows: dict[str, list[tuple[float, float, float, float]]] = {}
    span_rows: list[tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts = item.get("timestamp")
        pose = item.get("ie_pose")
        if ts is None or not isinstance(pose, dict):
            continue
        try:
            t = float(ts)
            s_lat = float(pose["latitude_deg"])
            s_lon = float(pose["longitude_deg"])
            s_alt = float(pose["height_m"])
        except (KeyError, TypeError, ValueError):
            continue
        span_rows.append((t, s_lon, s_lat, s_alt))

        use_multi = source in {"auto", "multi_tracks"}
        use_groups = source in {"auto", "aruco_groups"}
        use_single = source in {"auto", "single"}
        added = False

        if use_multi:
            for target in item.get("multi_targets", []):
                lla = target.get("target_world_lla")
                tid = target.get("track_id")
                if tid is None or not isinstance(lla, list) or len(lla) < 3:
                    continue
                try:
                    lat = float(lla[0])
                    lon = float(lla[1])
                    alt = float(lla[2])
                except (TypeError, ValueError):
                    continue
                key = f"track_{int(tid)}"
                track_rows.setdefault(key, []).append((t, lon, lat, alt))
                added = True

        if use_groups and (source == "aruco_groups" or not added):
            for target in item.get("aruco_group_targets", []):
                lla = target.get("target_world_lla")
                gid = target.get("target_group_index")
                if gid is None or not isinstance(lla, list) or len(lla) < 3:
                    continue
                try:
                    lat = float(lla[0])
                    lon = float(lla[1])
                    alt = float(lla[2])
                except (TypeError, ValueError):
                    continue
                key = f"group_{int(gid)}"
                track_rows.setdefault(key, []).append((t, lon, lat, alt))
                added = True

        if use_single and not added:
            lla = item.get("target_world_lla")
            if isinstance(lla, list) and len(lla) >= 3:
                try:
                    lat = float(lla[0])
                    lon = float(lla[1])
                    alt = float(lla[2])
                except (TypeError, ValueError):
                    lat = lon = alt = None
                if lat is not None:
                    track_rows.setdefault("single", []).append((t, lon, lat, alt))

    for key in list(track_rows.keys()):
        track_rows[key].sort(key=lambda row: row[0])
        if len(track_rows[key]) < 2:
            track_rows.pop(key, None)
    span_rows.sort(key=lambda row: row[0])
    return track_rows, span_rows


def write_multi_kml(
    target_tracks: dict[str, list[tuple[float, float, float, float]]],
    span_points: list[tuple[float, float, float, float]],
    out_path: Path,
    altitude_mode: str,
) -> None:
    if not target_tracks:
        raise ValueError("No valid multi-target tracks to export.")
    if len(span_points) < 2:
        raise ValueError(f"Not enough valid SPAN points for trajectory: {len(span_points)}")

    sorted_items = sorted(target_tracks.items(), key=lambda item: item[0])
    first_track = sorted_items[0][1]
    first = first_track[0]
    last = first_track[-1]

    placemarks: list[str] = []
    for index, (name, points) in enumerate(sorted_items):
        coords = "\n".join(f"{lon:.9f},{lat:.9f},{alt:.3f}" for _, lon, lat, alt in points)
        color = _track_color_hex(index)
        placemarks.append(
            f"""
    <Style id="traj_{name}">
      <LineStyle>
        <color>{color}</color>
        <width>3</width>
      </LineStyle>
    </Style>
    <Placemark>
      <name>Target {name}</name>
      <styleUrl>#traj_{name}</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>
{coords}
        </coordinates>
      </LineString>
    </Placemark>
"""
        )

    span_coords_line = "\n".join(f"{lon:.9f},{lat:.9f},{alt:.3f}" for _, lon, lat, alt in span_points)
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Multi-Target and SPAN Trajectories</name>
{''.join(placemarks)}
    <Style id="spanLine">
      <LineStyle>
        <color>ff0000ff</color>
        <width>5</width>
      </LineStyle>
    </Style>
    <Style id="startPoint">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>1.0</scale>
      </IconStyle>
    </Style>
    <Style id="endPoint">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.0</scale>
      </IconStyle>
    </Style>
    <Placemark>
      <name>SPAN Track</name>
      <styleUrl>#spanLine</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>
{span_coords_line}
        </coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>Start</name>
      <styleUrl>#startPoint</styleUrl>
      <Point>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>{first[1]:.9f},{first[2]:.9f},{first[3]:.3f}</coordinates>
      </Point>
    </Placemark>
    <Placemark>
      <name>End</name>
      <styleUrl>#endPoint</styleUrl>
      <Point>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>{last[1]:.9f},{last[2]:.9f},{last[3]:.3f}</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(kml, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(str(args.input))
    output_path = Path(str(args.output))
    if args.recompute_target_from_lidar:
        target_points, span_points = load_synced_target_span_points_recomputed(
            input_path,
            Path(str(args.lidar_config)),
        )
        write_kml(target_points, span_points, output_path, args.altitude_mode)
        duration_s = target_points[-1][0] - target_points[0][0]
        print(
            f"saved={output_path} mode=recompute-single target_points={len(target_points)} span_points={len(span_points)} duration_s={duration_s:.3f}"
        )
        return

    target_tracks, span_points = load_multi_target_tracks(input_path, source=args.target_source)
    if target_tracks and len(span_points) >= 2:
        write_multi_kml(target_tracks, span_points, output_path, args.altitude_mode)
        first_track = next(iter(target_tracks.values()))
        duration_s = first_track[-1][0] - first_track[0][0]
        print(
            f"saved={output_path} mode=multi tracks={len(target_tracks)} span_points={len(span_points)} duration_s={duration_s:.3f}"
        )
        return

    # Fallback to legacy single-target export.
    target_points, span_points = load_synced_target_span_points(input_path)
    if len(target_points) < 2 or len(span_points) < 2:
        target_points = load_target_lla_points(input_path)
        span_points = load_span_lla_points(input_path)
    write_kml(target_points, span_points, output_path, args.altitude_mode)
    duration_s = target_points[-1][0] - target_points[0][0]
    print(f"saved={output_path} mode=single target_points={len(target_points)} span_points={len(span_points)} duration_s={duration_s:.3f}")


if __name__ == "__main__":
    main()
