#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

Point = Tuple[float, float, float, float, Optional[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full GT and target trajectories to Google Earth KML.")
    parser.add_argument("--input", default="outputs/target_world_positions.jsonl", help="Input JSONL path.")
    parser.add_argument(
        "--gt-txt",
        default="data/0421-PM-2026/CollectionSystem/IE/0421-PM-2026.txt",
        help="NovAtel/Inertial Explorer GT.txt path. Use full GT curve for IE trajectory.",
    )
    parser.add_argument("--output", default="outputs/target_trajectory.kml", help="Output KML path.")
    parser.add_argument(
        "--altitude-mode",
        choices=["absolute", "clampToGround", "relativeToGround"],
        default="clampToGround",
        help="KML altitude mode.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=500,
        help="Add one detail point placemark every N points. Set <=0 to disable.",
    )
    return parser.parse_args()


def dms_to_deg(deg_token: str, minute_token: str, second_token: str) -> float:
    deg = float(deg_token)
    minute = float(minute_token)
    second = float(second_token)
    sign = -1.0 if deg < 0 else 1.0
    return sign * (abs(deg) + minute / 60.0 + second / 3600.0)


def load_gt_points(path: Path) -> list[Point]:
    if not path.exists():
        raise FileNotFoundError(f"GT file not found: {path}")

    points: list[Point] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or not re.match(r"^[0-9]", line):
            continue
        parts = line.split()
        if len(parts) < 18:
            continue
        try:
            utc_s = float(parts[0])
            lat = dms_to_deg(parts[3], parts[4], parts[5])
            lon = dms_to_deg(parts[6], parts[7], parts[8])
            alt = float(parts[9])
            heading = float(parts[-2])
        except (TypeError, ValueError):
            continue
        points.append((utc_s, lon, lat, alt, heading))
    points.sort(key=lambda p: p[0])
    return points


def load_target_tracks(path: Path) -> dict[str, list[Point]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    target_tracks: dict[str, list[Point]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts = item.get("timestamp")
        if ts is None:
            continue
        try:
            t = float(ts)
        except (TypeError, ValueError):
            continue

        group_targets = item.get("aruco_group_targets")
        if isinstance(group_targets, list) and group_targets:
            for target in group_targets:
                if not isinstance(target, dict):
                    continue
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
                target_tracks.setdefault(f"target_{int(gid)}", []).append((t, lon, lat, alt, None))
            continue

    cleaned: dict[str, list[Point]] = {}
    for name, points in target_tracks.items():
        points.sort(key=lambda p: p[0])
        if len(points) >= 2:
            cleaned[name] = points
    return cleaned


def color_for_track(index: int) -> str:
    # KML color format: aabbggrr.
    palette = ["ff3c78d8", "ff00a5ff", "ff32cd32", "ffebce87", "ff7b68ee", "ff1493ff"]
    return palette[index % len(palette)]


def coords(points: list[Point]) -> str:
    return "\n".join(f"{lon:.9f},{lat:.9f},{alt:.3f}" for _, lon, lat, alt, _ in points)


def utc_s(ts: float) -> str:
    return f"{float(ts):.0f}s"


def desc_for_track(points: list[Point]) -> str:
    first = points[0]
    last = points[-1]
    duration_s = last[0] - first[0]
    heading_part = ""
    if first[4] is not None or last[4] is not None:
        heading_part = f"<br/>heading_start_deg={first[4]:.6f}<br/>heading_end_deg={last[4]:.6f}"
    return (
        f"points={len(points)}<br/>"
        f"utc_start={utc_s(first[0])}<br/>"
        f"utc_end={utc_s(last[0])}<br/>"
        f"duration_s={duration_s:.0f}"
        f"{heading_part}"
    )


def point_placemark(name: str, point: Point, altitude_mode: str, style_url: str = "") -> str:
    t, lon, lat, alt, heading = point
    style = f"<styleUrl>{style_url}</styleUrl>" if style_url else ""
    heading_part = "" if heading is None else f"<br/>heading_deg={heading:.6f}"
    return f"""
      <Placemark>
        <name>{name}</name>
        {style}
        <description><![CDATA[utc={utc_s(t)}<br/>lon={lon:.9f}<br/>lat={lat:.9f}<br/>alt={alt:.3f}{heading_part}]]></description>
        <Point>
          <altitudeMode>{altitude_mode}</altitudeMode>
          <coordinates>{lon:.9f},{lat:.9f},{alt:.3f}</coordinates>
        </Point>
      </Placemark>
"""


def sampled_points_folder(folder_name: str, prefix: str, points: list[Point], sample_step: int, altitude_mode: str) -> str:
    if sample_step <= 0:
        return ""
    marks: list[str] = []
    for index in range(0, len(points), sample_step):
        marks.append(point_placemark(f"{prefix}_{index}", points[index], altitude_mode))
    if (len(points) - 1) % sample_step != 0:
        marks.append(point_placemark(f"{prefix}_{len(points) - 1}", points[-1], altitude_mode))
    return f"""
    <Folder>
      <name>{folder_name}</name>
{''.join(marks)}
    </Folder>
"""


def write_kml(
    gt_points: list[Point],
    target_tracks: dict[str, list[Point]],
    out_path: Path,
    altitude_mode: str,
    sample_step: int,
) -> None:
    if len(gt_points) < 2:
        raise ValueError(f"Not enough GT points: {len(gt_points)}")
    if not target_tracks:
        raise ValueError("No valid target tracks found.")

    target_parts: list[str] = []
    target_sample_folders: list[str] = []
    for index, (name, points) in enumerate(sorted(target_tracks.items(), key=lambda item: item[0])):
        color = color_for_track(index)
        target_parts.append(
            f"""
    <Style id="{name}_line">
      <LineStyle>
        <color>{color}</color>
        <width>4</width>
      </LineStyle>
    </Style>
    <Folder>
      <name>{name}</name>
      <description><![CDATA[{desc_for_track(points)}]]></description>
      <Placemark>
        <name>{name}_track</name>
        <styleUrl>#{name}_line</styleUrl>
        <description><![CDATA[{desc_for_track(points)}]]></description>
        <LineString>
          <tessellate>1</tessellate>
          <altitudeMode>{altitude_mode}</altitudeMode>
          <coordinates>
{coords(points)}
          </coordinates>
        </LineString>
      </Placemark>
{point_placemark(f"{name}_start", points[0], altitude_mode, "#startPoint")}
{point_placemark(f"{name}_end", points[-1], altitude_mode, "#endPoint")}
    </Folder>
"""
        )
        target_sample_folders.append(sampled_points_folder(f"{name}_sampled_points", name, points, sample_step, altitude_mode))

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Full GT and Target Trajectories</name>
    <description><![CDATA[
      GT uses full GT.txt curve. Time values are UTC seconds from GT/JSONL timestamps.
      <br/>gt_points={len(gt_points)}
      <br/>target_tracks={len(target_tracks)}
    ]]></description>
    <Style id="gtLine">
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
    <Folder>
      <name>GT_Full_Curve</name>
      <description><![CDATA[{desc_for_track(gt_points)}]]></description>
      <Placemark>
        <name>gt_full_track</name>
        <styleUrl>#gtLine</styleUrl>
        <description><![CDATA[{desc_for_track(gt_points)}]]></description>
        <LineString>
          <tessellate>1</tessellate>
          <altitudeMode>{altitude_mode}</altitudeMode>
          <coordinates>
{coords(gt_points)}
          </coordinates>
        </LineString>
      </Placemark>
{point_placemark("gt_start", gt_points[0], altitude_mode, "#startPoint")}
{point_placemark("gt_end", gt_points[-1], altitude_mode, "#endPoint")}
    </Folder>
    <Folder>
      <name>Targets</name>
{''.join(target_parts)}
    </Folder>
{sampled_points_folder("GT_Sampled_Points", "gt", gt_points, sample_step, altitude_mode)}
{''.join(target_sample_folders)}
  </Document>
</kml>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(kml, encoding="utf-8")


def main() -> None:
    args = parse_args()
    gt_points = load_gt_points(Path(args.gt_txt))
    target_tracks = load_target_tracks(Path(args.input))
    write_kml(gt_points, target_tracks, Path(args.output), args.altitude_mode, args.sample_step)
    print(
        f"saved={args.output} gt_points={len(gt_points)} target_tracks={len(target_tracks)} "
        f"sample_step={args.sample_step}"
    )


if __name__ == "__main__":
    main()
