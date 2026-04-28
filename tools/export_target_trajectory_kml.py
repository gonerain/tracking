#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

Point = Tuple[float, float, float, float, Optional[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full GT and target trajectories to Google Earth KML.")
    parser.add_argument("--input", default="outputs/target_world_positions.csv", help="Input CSV path.")
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

    import csv
    target_tracks: dict[str, list[Point]] = {}
    with path.open(encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row.get("utc_seconds", ""))
            except (TypeError, ValueError):
                continue
            for gid in range(2):
                prefix = f"target_{gid}"
                lat_s = row.get(f"{prefix}_lat", "")
                lon_s = row.get(f"{prefix}_lon", "")
                alt_s = row.get(f"{prefix}_height", "")
                if not lat_s or not lon_s or not alt_s:
                    continue
                try:
                    lat = float(lat_s)
                    lon = float(lon_s)
                    alt = float(alt_s)
                except (TypeError, ValueError):
                    continue
                target_tracks.setdefault(prefix, []).append((t, lon, lat, alt, None))

    cleaned: dict[str, list[Point]] = {}
    for name, points in target_tracks.items():
        points.sort(key=lambda p: p[0])
        if len(points) >= 2:
            cleaned[name] = points
    return cleaned


def color_for_track(index: int) -> str:
    # KML color format: aabbggrr.
    # target_0 = green, target_1 = yellow, then cycle
    palette = ["ff00ff00", "ff00ffff", "ffff0000", "ffff00ff", "ffffff00"]
    return palette[index % len(palette)]


def coords(points: list[Point]) -> str:
    return "\n".join(f"{lon:.9f},{lat:.9f},{alt:.3f}" for _, lon, lat, alt, _ in points)


def utc_s(ts: float) -> str:
    return f"{float(ts):.0f}s"


def unix_to_iso8601(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def gx_track(points: list[Point], altitude_mode: str, schema_id: str) -> str:
    """Render a gx:Track with per-point timestamps in ExtendedData.

    Each point has a <when> entry and the same timestamp is stored in
    ExtendedData so Google Earth shows it in the info balloon on click.
    """
    whens = "\n".join(f"        <when>{unix_to_iso8601(t)}</when>" for t, *_ in points)
    coords_gx = "\n".join(
        f"        <gx:coord>{lon:.9f} {lat:.9f} {alt:.3f}</gx:coord>"
        for _, lon, lat, alt, _ in points
    )
    ts_values = "\n".join(
        f"          <gx:value>{t:.3f}</gx:value>" for t, *_ in points
    )
    return (
        f"      <gx:Track>\n"
        f"        <altitudeMode>{altitude_mode}</altitudeMode>\n"
        f"{whens}\n"
        f"{coords_gx}\n"
        f"        <ExtendedData>\n"
        f"          <SchemaData schemaUrl=\"#{schema_id}\">\n"
        f"            <gx:SimpleArrayData name=\"utc_seconds\">\n"
        f"{ts_values}\n"
        f"            </gx:SimpleArrayData>\n"
        f"          </SchemaData>\n"
        f"        </ExtendedData>\n"
        f"      </gx:Track>"
    )


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


def point_placemark(name: str, point: Point, altitude_mode: str, style_url: str = "", show_timestamp: bool = False) -> str:
    t, lon, lat, alt, heading = point
    style = f"<styleUrl>{style_url}</styleUrl>" if style_url else ""
    heading_part = "" if heading is None else f"<br/>heading_deg={heading:.6f}"
    iso = unix_to_iso8601(t)
    timestamp_elem = f"<TimeStamp><when>{iso}</when></TimeStamp>" if show_timestamp else ""
    label = iso if show_timestamp else name
    return f"""
      <Placemark>
        <name>{label}</name>
        {style}
        {timestamp_elem}
        <description><![CDATA[utc={utc_s(t)}<br/>datetime={iso}<br/>lon={lon:.9f}<br/>lat={lat:.9f}<br/>alt={alt:.3f}{heading_part}]]></description>
        <Point>
          <altitudeMode>{altitude_mode}</altitudeMode>
          <coordinates>{lon:.9f},{lat:.9f},{alt:.3f}</coordinates>
        </Point>
      </Placemark>
"""


def write_kml(
    gt_points: list[Point],
    target_tracks: dict[str, list[Point]],
    out_path: Path,
    altitude_mode: str,
) -> None:
    if not target_tracks:
        raise ValueError("No valid target tracks found.")

    # Clip GT to the time range of the target tracks (with 5s padding)
    all_target_ts = [p[0] for pts in target_tracks.values() for p in pts]
    if all_target_ts:
        t_min = min(all_target_ts) - 5.0
        t_max = max(all_target_ts) + 5.0
        gt_points = [p for p in gt_points if t_min <= p[0] <= t_max]

    # Downsample if still too large for Google Earth to render
    if len(gt_points) > 10000:
        step = max(1, len(gt_points) // 10000)
        gt_points = gt_points[::step]

    if len(gt_points) < 2:
        raise ValueError(f"Not enough GT points: {len(gt_points)}")

    target_parts: list[str] = []
    for index, (name, points) in enumerate(sorted(target_tracks.items(), key=lambda item: item[0])):
        color = color_for_track(index)
        schema_id = f"{name}_schema"
        target_parts.append(
            f"""
    <Schema name="{schema_id}" id="{schema_id}">
      <gx:SimpleArrayField name="utc_seconds" type="xsd:string">
        <displayName>UTC Seconds</displayName>
      </gx:SimpleArrayField>
    </Schema>
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
{gx_track(points, altitude_mode, schema_id)}
      </Placemark>
{point_placemark(f"{name}_start", points[0], altitude_mode, "#startPoint", show_timestamp=True)}
{point_placemark(f"{name}_end", points[-1], altitude_mode, "#endPoint", show_timestamp=True)}
    </Folder>
"""
        )

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
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
        <width>4</width>
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
      <name>Targets</name>
{''.join(target_parts)}
    </Folder>
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
  </Document>
</kml>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(kml, encoding="utf-8")


def main() -> None:
    args = parse_args()
    gt_points = load_gt_points(Path(args.gt_txt))
    target_tracks = load_target_tracks(Path(args.input))
    write_kml(gt_points, target_tracks, Path(args.output), args.altitude_mode)
    print(
        f"saved={args.output} gt_points={len(gt_points)} target_tracks={len(target_tracks)}"
    )


if __name__ == "__main__":
    main()
