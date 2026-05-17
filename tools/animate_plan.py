#!/usr/bin/env python3
"""Generate a GIF/MP4 animation of a saved plan.

Usage:
    python3 tools/animate_plan.py --plan budget_600s
    python3 tools/animate_plan.py --plan budget_600s --fps 15
    python3 tools/animate_plan.py --plan budget_600s --duration-scale 0.5   # 2x speed
    python3 tools/animate_plan.py --plan budget_600s --mp4                  # requires ffmpeg
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.animate import animate_plan_gif
from src.osm_parser import parse_osm
from src.plan_writer import Plan
from src.topology import build_lanelets, load_topology_ground_truth, load_topology_overrides
from src.waypoint_anchor import extract_start_area, extract_waypoint_areas


def _rehydrate_plan(saved: dict, osm_path: str) -> Plan:
    """Build a Plan-like object from the saved JSON, only filling the fields the
    animator needs."""
    plan = type("PlanRehydrated", (), {})()
    plan.lanelet_sequence = list(saved.get("lanelet_sequence", []))
    plan.waypoint_sequence = list(saved.get("waypoint_sequence", []))
    plan.expected_total_time_s = float(saved.get("expected_total_time_s", 0.0))
    plan.expected_distance_m = float(saved.get("expected_distance_m", 0.0))
    plan.expected_score = int(saved.get("expected_score", 0))
    plan.expected_waypoints_visited = int(saved.get("expected_waypoints_visited", 0))
    plan.start_pose = (
        float(saved.get("start_pose", {}).get("x", 0.0)),
        float(saved.get("start_pose", {}).get("y", 0.0)),
        float(saved.get("start_pose", {}).get("yaw_rad", 0.0)),
    )
    plan.budget_s = float(saved.get("params", {}).get("time_budget_s", 600.0))
    plan.osm_path = osm_path
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Animate a saved BFMC plan.")
    parser.add_argument("--plan", required=True, help="Plan name (folder under data/outputs/plans/).")
    parser.add_argument("--osm", default=os.path.join(_PROJECT_ROOT, "data", "lanelet2_map_FINAL_RandomStartingArea.osm"))
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--duration-scale", type=float, default=0.25,
        help="Animation duration = plan_time * scale. 0.25 = 4x speed (10 min plan -> 2.5 min animation).",
    )
    parser.add_argument("--mp4", action="store_true", help="Output as MP4 (requires ffmpeg).")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    plan_dir = os.path.join(_PROJECT_ROOT, "data", "outputs", "plans", args.plan)
    plan_json = os.path.join(plan_dir, "plan.json")
    if not os.path.exists(plan_json):
        print(f"ERROR: plan not found: {plan_json}", file=sys.stderr)
        return 1
    with open(plan_json) as fh:
        saved = json.load(fh)

    doc = parse_osm(args.osm)
    gt = load_topology_ground_truth(os.path.join(_PROJECT_ROOT, "data", "topology_current.json"))
    overrides = load_topology_overrides(os.path.join(_PROJECT_ROOT, "data", "topology_overrides.json"))
    lanelets = build_lanelets(doc, step_m=0.05, overrides=overrides, ground_truth=gt)
    waypoints = extract_waypoint_areas(doc)
    start_area = extract_start_area(doc)
    plan = _rehydrate_plan(saved, args.osm)

    ext = ".mp4" if args.mp4 else ".gif"
    out_path = args.output or os.path.join(plan_dir, f"animation{ext}")

    print(f"Rendering animation -> {out_path}")
    animate_plan_gif(
        plan, lanelets, waypoints, start_area, out_path,
        fps=args.fps, duration_scale=args.duration_scale, use_mp4=args.mp4,
    )
    print(f"Done. ({os.path.getsize(out_path) / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
