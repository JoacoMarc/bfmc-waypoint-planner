#!/usr/bin/env python3
"""Sensitivity and reachability analysis for a saved plan.

Usage:
    python3 tools/analyze_plan.py --plan budget_600s --sensitivity
    python3 tools/analyze_plan.py --plan budget_600s --reachability --budget-range 300,1500
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

import numpy as np

from src.cost_matrix import CostParams, build_cost_matrix, load_lanelet_penalties
from src.osm_parser import parse_osm
from src.plan_writer import SCORE_RANDOM_START, SCORE_WAYPOINT
from src.reachability import (
    reachability_curve, write_reachability_markdown, write_reachability_svg,
)
from src.sensitivity import (
    SensitivityAxis, run_sensitivity_sweep,
    write_sensitivity_markdown, write_sensitivity_svg,
)
from src.topology import build_lanelets, load_topology_ground_truth, load_topology_overrides
from src.traffic_light import load_traffic_lights
from src.waypoint_anchor import (
    compute_waypoint_anchors, extract_start_area, extract_waypoint_areas,
    find_start_exit_lanelets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a saved BFMC plan.")
    parser.add_argument("--plan", required=True, help="Plan name (folder under data/outputs/plans/).")
    parser.add_argument("--osm", default=os.path.join(_PROJECT_ROOT, "data", "lanelet2_map_FINAL_RandomStartingArea.osm"))
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--reachability", action="store_true")
    parser.add_argument("--budget-range", type=str, default="300,1500",
                        help="Comma-separated: start,end[,step]. Default 300,1500,150.")
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--sa-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plan_dir = os.path.join(_PROJECT_ROOT, "data", "outputs", "plans", args.plan)
    plan_json = os.path.join(plan_dir, "plan.json")
    if not os.path.exists(plan_json):
        print(f"ERROR: plan not found: {plan_json}", file=sys.stderr)
        return 1
    with open(plan_json) as fh:
        saved = json.load(fh)

    # Rebuild the same context the plan was generated with.
    doc = parse_osm(args.osm)
    gt = load_topology_ground_truth(os.path.join(_PROJECT_ROOT, "data", "topology_current.json"))
    overrides = load_topology_overrides(os.path.join(_PROJECT_ROOT, "data", "topology_overrides.json"))
    lanelets = build_lanelets(doc, step_m=0.05, overrides=overrides, ground_truth=gt)
    waypoints = extract_waypoint_areas(doc)
    start_area = extract_start_area(doc)
    anchors = compute_waypoint_anchors(lanelets, waypoints)

    saved_params = saved.get("params", {})
    lanelet_pen = load_lanelet_penalties(os.path.join(_PROJECT_ROOT, "data", "lanelet_penalties.json"))
    tl = load_traffic_lights(os.path.join(_PROJECT_ROOT, "data", "traffic_lights.json"))
    base_params = CostParams(
        speed_urban=float(saved_params.get("speed_urban", 0.2)),
        speed_highway=float(saved_params.get("speed_highway", 0.4)),
        efficiency_factor=float(saved_params.get("efficiency_factor", 0.7)),
        penalty_intersection_s=float(saved_params.get("penalty_intersection_s", 1.5)),
        penalty_stopline_s=float(saved_params.get("penalty_stopline_s", 4.0)),
        penalty_crosswalk_s=float(saved_params.get("penalty_crosswalk_s", 1.0)),
        penalty_roundabout_s=float(saved_params.get("penalty_roundabout_s", 1.0)),
        lanelet_penalties_s=lanelet_pen,
        traffic_lights=tl if tl.total_lanelets() else None,
    )
    budget_s = float(saved_params.get("time_budget_s", 600.0))
    start_pose_d = saved.get("start_pose", {})
    start_pose = (
        float(start_pose_d.get("x", 0.0)),
        float(start_pose_d.get("y", 0.0)),
        float(start_pose_d.get("yaw_rad", 0.0)),
    )
    start_bonus = int(saved_params.get("score_random_start", SCORE_RANDOM_START))
    if saved.get("expected_score", 0) - SCORE_WAYPOINT * saved.get("expected_waypoints_visited", 0) == 0:
        start_bonus = 0

    if start_area is None:
        print("ERROR: no start area found", file=sys.stderr)
        return 1
    sel = find_start_exit_lanelets(lanelets, start_area)
    cache_dir = os.path.join(_PROJECT_ROOT, "data", "cache")

    if args.sensitivity:
        print("[sensitivity] sweeping parameters...")
        axes = [
            SensitivityAxis("speed_urban", [0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26]),
            SensitivityAxis("speed_highway", [0.30, 0.35, 0.40, 0.45, 0.50]),
            SensitivityAxis("efficiency_factor", [0.55, 0.65, 0.70, 0.75, 0.85]),
            SensitivityAxis("penalty_stopline_s", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            SensitivityAxis("penalty_intersection_s", [0.5, 1.0, 1.5, 2.0, 3.0]),
        ]
        results = run_sensitivity_sweep(
            osm_path=args.osm, lanelets=lanelets, anchors=anchors,
            start_exit_lanelets=sel, base_params=base_params, axes=axes,
            budget_s=budget_s, restarts=args.restarts, sa_iterations=args.sa_iterations,
            seed=args.seed, start_bonus=start_bonus, cache_dir=cache_dir,
            waypoints=waypoints, start_pose=start_pose, verbose=True,
        )
        md_path = os.path.join(plan_dir, "sensitivity.md")
        svg_path = os.path.join(plan_dir, "sensitivity.svg")
        write_sensitivity_markdown(results, md_path)
        write_sensitivity_svg(results, svg_path)
        print(f"  -> {md_path}")
        print(f"  -> {svg_path}")

    if args.reachability:
        print("[reachability] computing curve...")
        parts = [float(x) for x in args.budget_range.split(",")]
        if len(parts) == 2:
            parts.append(150.0)
        start, end, step = parts[0], parts[1], parts[2]
        budgets = list(np.arange(start, end + step / 2, step))
        # Use a fresh cost matrix built with the plan's params.
        cm = build_cost_matrix(lanelets, anchors, sel, base_params, args.osm,
                               cache_dir=cache_dir, verbose=False)
        N = len(anchors)
        scores = np.zeros(N + 1, dtype=float)
        scores[0] = float(start_bonus)
        for i in range(1, N + 1):
            scores[i] = float(SCORE_WAYPOINT)
        wp_id_per_node = [""] + [a.wp_id for a in anchors]
        rows = reachability_curve(
            cost_matrix=cm, scores=scores, wp_id_per_node=wp_id_per_node,
            budgets=budgets, restarts=args.restarts, sa_iterations=args.sa_iterations,
            seed=args.seed,
        )
        md_path = os.path.join(plan_dir, "reachability.md")
        svg_path = os.path.join(plan_dir, "reachability.svg")
        write_reachability_markdown(rows, md_path)
        write_reachability_svg(rows, svg_path)
        print(f"  -> {md_path}")
        print(f"  -> {svg_path}")

    if not (args.sensitivity or args.reachability):
        print("Nothing to do. Use --sensitivity and/or --reachability.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
