#!/usr/bin/env python3
import json
import sys
import numpy as np

records = []
path = sys.argv[1] if len(sys.argv) > 1 else "outputs/merged/target_world_positions.jsonl"
with open(path) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

groups = {}
for i, rec in enumerate(records):
    for gt in rec.get("aruco_group_targets", []):
        gidx = gt.get("target_group_index", -1)
        enu = gt.get("target_world_enu_m")
        lidar_m = gt.get("target_lidar_m")
        if enu is None:
            continue
        groups.setdefault(gidx, []).append({
            "frame": i, "ts": rec["timestamp"],
            "enu": np.array(enu), "lidar": lidar_m,
            "interp": gt.get("interpolated", False),
        })

for gidx in sorted(groups.keys()):
    pts = groups[gidx]
    print(f"=== Group {gidx}: {len(pts)} points ===")
    cnt = 0
    for j in range(1, len(pts)):
        dt = pts[j]["ts"] - pts[j - 1]["ts"]
        if dt <= 0:
            continue
        dist = np.linalg.norm(pts[j]["enu"][:2] - pts[j - 1]["enu"][:2])
        speed = dist / dt
        if speed > 5.0:
            cnt += 1
            if cnt <= 8:
                p = pts[j]
                pp = pts[j - 1]
                print(f"  frame={p['frame']} speed={speed:.1f}m/s dist={dist:.3f}m dt={dt:.3f}s")
                print(f"    prev lidar={pp['lidar']}  interp={pp['interp']}")
                print(f"    curr lidar={p['lidar']}  interp={p['interp']}")
                if j + 1 < len(pts):
                    dt2 = pts[j + 1]["ts"] - p["ts"]
                    if dt2 > 0:
                        d2 = np.linalg.norm(pts[j + 1]["enu"][:2] - p["enu"][:2])
                        print(f"    next speed={d2/dt2:.1f}m/s (bounce-back?)")
    print(f"  Total >5m/s spikes: {cnt}")
