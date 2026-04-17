#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main() -> None:
    args = parse_args()
    input_path = Path(str(args.input))
    output_path = Path(str(args.output))
    target_points, span_points = load_synced_target_span_points(input_path)
    if len(target_points) < 2 or len(span_points) < 2:
        # Fallback to independent trajectories only when synced records are too few.
        target_points = load_target_lla_points(input_path)
        span_points = load_span_lla_points(input_path)
    write_kml(target_points, span_points, output_path, args.altitude_mode)
    duration_s = target_points[-1][0] - target_points[0][0]
    print(
        f"saved={output_path} target_points={len(target_points)} span_points={len(span_points)} duration_s={duration_s:.3f}"
    )


if __name__ == "__main__":
    main()
