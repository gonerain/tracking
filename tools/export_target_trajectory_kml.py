#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export IE GT and target trajectory to Google Earth KML.")
    parser.add_argument("--input", default="outputs/target_world_positions.jsonl", help="Input JSONL path.")
    parser.add_argument("--output", default="outputs/target_trajectory.kml", help="Output KML path.")
    parser.add_argument(
        "--altitude-mode",
        choices=["absolute", "clampToGround", "relativeToGround"],
        default="clampToGround",
        help="KML altitude mode.",
    )
    return parser.parse_args()


def load_ie_and_targets(
    path: Path,
) -> tuple[list[tuple[float, float, float, float]], dict[str, list[tuple[float, float, float, float]]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ie_points: list[tuple[float, float, float, float]] = []
    target_tracks: dict[str, list[tuple[float, float, float, float]]] = {}

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
            ie_lat = float(pose["latitude_deg"])
            ie_lon = float(pose["longitude_deg"])
            ie_alt = float(pose["height_m"])
        except (KeyError, TypeError, ValueError):
            continue

        ie_points.append((t, ie_lon, ie_lat, ie_alt))

        # Preferred: one trajectory per configured ArUco group (one board == one target).
        group_targets = item.get("aruco_group_targets")
        if isinstance(group_targets, list) and group_targets:
            for gt in group_targets:
                lla = gt.get("target_world_lla") if isinstance(gt, dict) else None
                gid = gt.get("target_group_index") if isinstance(gt, dict) else None
                if gid is None or not isinstance(lla, list) or len(lla) < 3:
                    continue
                try:
                    lat = float(lla[0])
                    lon = float(lla[1])
                    alt = float(lla[2])
                except (TypeError, ValueError):
                    continue
                target_tracks.setdefault(f"target_{int(gid)}", []).append((t, lon, lat, alt))
            continue

        # Legacy single-target fallback (only when group targets are unavailable).
        lla = item.get("target_world_lla")
        if isinstance(lla, list) and len(lla) >= 3:
            try:
                lat = float(lla[0])
                lon = float(lla[1])
                alt = float(lla[2])
            except (TypeError, ValueError):
                continue
            target_tracks.setdefault("target_0", []).append((t, lon, lat, alt))

    ie_points.sort(key=lambda p: p[0])
    cleaned_tracks: dict[str, list[tuple[float, float, float, float]]] = {}
    for name, pts in target_tracks.items():
        pts.sort(key=lambda p: p[0])
        if len(pts) >= 2:
            cleaned_tracks[name] = pts
    return ie_points, cleaned_tracks


def color_for_track(index: int) -> str:
    # KML color format: aabbggrr
    palette = [
        "ff3c78d8",
        "ff00a5ff",
        "ff32cd32",
        "ffebce87",
        "ff7b68ee",
        "ff1493ff",
    ]
    return palette[index % len(palette)]


def write_kml(
    ie_points: list[tuple[float, float, float, float]],
    target_tracks: dict[str, list[tuple[float, float, float, float]]],
    out_path: Path,
    altitude_mode: str,
) -> None:
    if len(ie_points) < 2:
        raise ValueError(f"Not enough IE points: {len(ie_points)}")
    if not target_tracks:
        raise ValueError("No valid target tracks found.")

    ie_coords = "\n".join(f"{lon:.9f},{lat:.9f},{alt:.3f}" for _, lon, lat, alt in ie_points)

    target_styles_and_marks: list[str] = []
    sorted_tracks = sorted(target_tracks.items(), key=lambda kv: kv[0])
    for i, (name, pts) in enumerate(sorted_tracks):
        coords = "\n".join(f"{lon:.9f},{lat:.9f},{alt:.3f}" for _, lon, lat, alt in pts)
        color = color_for_track(i)
        target_styles_and_marks.append(
            f"""
    <Style id=\"{name}_style\">
      <LineStyle>
        <color>{color}</color>
        <width>3</width>
      </LineStyle>
    </Style>
    <Placemark>
      <name>{name}</name>
      <styleUrl>#{name}_style</styleUrl>
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

    first_target = sorted_tracks[0][1][0]
    last_target = sorted_tracks[0][1][-1]

    kml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<kml xmlns=\"http://www.opengis.net/kml/2.2\">
  <Document>
    <name>Target and IE Trajectories</name>
{''.join(target_styles_and_marks)}
    <Style id=\"ie_style\">
      <LineStyle>
        <color>ff0000ff</color>
        <width>5</width>
      </LineStyle>
    </Style>
    <Style id=\"startPoint\">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>1.0</scale>
      </IconStyle>
    </Style>
    <Style id=\"endPoint\">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.0</scale>
      </IconStyle>
    </Style>
    <Placemark>
      <name>ie_gt</name>
      <styleUrl>#ie_style</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>
{ie_coords}
        </coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>target_start</name>
      <styleUrl>#startPoint</styleUrl>
      <Point>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>{first_target[1]:.9f},{first_target[2]:.9f},{first_target[3]:.3f}</coordinates>
      </Point>
    </Placemark>
    <Placemark>
      <name>target_end</name>
      <styleUrl>#endPoint</styleUrl>
      <Point>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>{last_target[1]:.9f},{last_target[2]:.9f},{last_target[3]:.3f}</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(kml, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    ie_points, target_tracks = load_ie_and_targets(input_path)
    write_kml(ie_points, target_tracks, output_path, args.altitude_mode)

    print(f"saved={output_path} ie_points={len(ie_points)} target_tracks={len(target_tracks)}")


if __name__ == "__main__":
    main()
