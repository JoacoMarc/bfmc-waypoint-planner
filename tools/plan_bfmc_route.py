#!/usr/bin/env python3
"""CLI orchestrator for the BFMC waypoint route planner.

Usage:
    python tools/plan_bfmc_route.py --osm data/lanelet2_map_FINAL_RandomStartingArea.osm
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json as _json
import math as _math
import os
import sys
import time

# Make src importable when running from project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from src.constraints import Constraints, load_constraints, validate_plan as validate_constraints
from src.cost_matrix import CostParams, build_cost_matrix, load_lanelet_penalties
from src.multi_start import run_multi_start
from src.traffic_light import load_traffic_lights
from src.orienteering_solver import solve_orienteering
from src.osm_parser import parse_osm
from src.plan_writer import SCORE_RANDOM_START, SCORE_WAYPOINT, build_plan, write_plan_json
from src.topology import build_lanelets, load_topology_ground_truth, load_topology_overrides
from src.visualizer_svg import render_plan_svg
from src.waypoint_anchor import (
    compute_waypoint_anchors,
    extract_start_area,
    extract_waypoint_areas,
    find_start_exit_lanelets,
    pick_start_pose,
)


def _parse_pose(value: str) -> tuple[float, float, float] | None:
    if not value or value.upper() == "AUTO":
        return None
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--start-pose must be 'AUTO' or 'x,y,yaw_rad'"
        )
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute BFMC route plan (orienteering over waypoint dots).",
    )
    parser.add_argument("--osm", required=True, help="Path to the Lanelet2 OSM file.")
    parser.add_argument("--speed-urban", type=float, default=0.2, help="Urban speed (m/s).")
    parser.add_argument("--speed-highway", type=float, default=0.4, help="Highway speed (m/s).")
    parser.add_argument("--time-budget", type=float, default=600.0, help="Time budget (s).")
    parser.add_argument("--efficiency", type=float, default=0.7, help="Efficiency factor 0..1.")
    parser.add_argument("--penalty-stopline", type=float, default=4.0, help="Stopline penalty (s).")
    parser.add_argument("--penalty-intersection", type=float, default=1.5, help="Intersection penalty (s).")
    parser.add_argument("--penalty-crosswalk", type=float, default=1.0, help="Crosswalk penalty (s).")
    parser.add_argument("--penalty-roundabout", type=float, default=1.0, help="Roundabout penalty (s).")
    parser.add_argument("--start-pose", type=str, default="AUTO", help="AUTO or 'x,y,yaw_rad'.")
    parser.add_argument(
        "--start-mode",
        choices=["random", "default"],
        default="random",
        help="'random': start anywhere inside the random start polygon (+15 pts bonus). "
             "'default': start at the beginning of --default-start-lanelet (no bonus).",
    )
    parser.add_argument(
        "--default-start-lanelet",
        type=str,
        default="995",
        help="Lanelet ID used as the starting lanelet when --start-mode=default.",
    )
    parser.add_argument("--solver", choices=["greedy", "sa", "hybrid"], default="hybrid")
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--sa-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Plan name (used as subfolder under --plans-dir). If omitted, a timestamp is used.",
    )
    parser.add_argument(
        "--plans-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "outputs", "plans"),
        help="Parent directory where each plan is stored as its own subfolder.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Explicit JSON output path (overrides --name/--plans-dir).",
    )
    parser.add_argument(
        "--output-svg",
        type=str,
        default=None,
        help="Explicit SVG output path (overrides --name/--plans-dir).",
    )
    parser.add_argument(
        "--max-waypoints",
        action="store_true",
        help="Ignore the time budget and connect as many waypoints as possible. The plan may exceed --time-budget; the solver maximizes the number of waypoints visited and only uses --time-budget for reporting.",
    )
    parser.add_argument(
        "--list-plans",
        action="store_true",
        help="List previously saved plans and exit.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "cache"),
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable cost matrix cache.")
    parser.add_argument(
        "--topology-overrides",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "topology_overrides.json"),
        help="JSON file with manual successor add/remove rules (ignored if --topology-ground-truth is set).",
    )
    parser.add_argument(
        "--topology-ground-truth",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "topology_current.json"),
        help="JSON file with the full successor map. Takes absolute precedence over OSM inference.",
    )
    parser.add_argument(
        "--lanelet-penalties",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "lanelet_penalties.json"),
        help="JSON file with per-lanelet extra time penalties (e.g. traffic lights, parking).",
    )
    parser.add_argument(
        "--traffic-lights",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "traffic_lights.json"),
        help="JSON file with traffic-light cycle model. Overrides lanelet_penalties for listed lanelets.",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "constraints.json"),
        help="JSON with order constraints (must_visit_first / forbidden / before_after).",
    )
    parser.add_argument(
        "--multi-start",
        type=int,
        default=1,
        help="Number of poses to sample inside the random start polygon (random mode only). "
             "Best plan kept. 1 = original heuristic.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # --list-plans short-circuits the pipeline.
    if args.list_plans:
        _print_plans_list(args.plans_dir)
        return 0

    osm_path = args.osm
    if not os.path.isabs(osm_path):
        osm_path = os.path.join(_PROJECT_ROOT, osm_path)
    if not os.path.exists(osm_path):
        print(f"ERROR: OSM file not found: {osm_path}", file=sys.stderr)
        return 1

    plan_name, out_json_path, out_svg_path = _resolve_output_paths(args)

    print(f"[1/7] Parsing OSM: {osm_path}")
    t0 = time.time()
    doc = parse_osm(osm_path)
    print(
        f"      nodes={len(doc.nodes)}  ways={len(doc.ways)}  "
        f"lanelets={len(doc.lanelets)}  multipolygons={len(doc.multipolygons)}  "
        f"start_area_way={doc.start_area_way_id}  ({time.time() - t0:.2f}s)"
    )

    print("[2/7] Building lanelet topology")
    t0 = time.time()
    ground_truth = load_topology_ground_truth(args.topology_ground_truth)
    if ground_truth is not None:
        total_edges = sum(len(v) for v in ground_truth.values())
        print(f"      using ground-truth topology from {args.topology_ground_truth} ({total_edges} edges, {len(ground_truth)} lanelets)")
        overrides = {}
    else:
        overrides = load_topology_overrides(args.topology_overrides)
        if overrides:
            n_add = sum(len(v) for v in overrides.get("add_successors", {}).values())
            n_rem = sum(len(v) for v in overrides.get("remove_successors", {}).values())
            print(f"      overrides: +{n_add} successors / -{n_rem} successors from {args.topology_overrides}")
    lanelets = build_lanelets(doc, step_m=0.05, overrides=overrides, ground_truth=ground_truth)
    print(f"      built {len(lanelets)} lanelets  ({time.time() - t0:.2f}s)")

    print("[3/7] Extracting waypoints and start area")
    t0 = time.time()
    waypoints = extract_waypoint_areas(doc)
    if args.start_mode == "default":
        # In default-start mode the random-start polygon is ignored; the car
        # starts at the beginning of a fixed lanelet.
        start_area = None
        start_lid = args.default_start_lanelet
        if start_lid not in lanelets:
            print(f"ERROR: --default-start-lanelet {start_lid} not found in the OSM.", file=sys.stderr)
            return 1
        print(f"      waypoints={len(waypoints)}  start_mode=default  start_lanelet={start_lid}  ({time.time() - t0:.2f}s)")
    else:
        start_area = extract_start_area(doc)
        if start_area is None:
            print("ERROR: no start area (way with area=yes) found in OSM.", file=sys.stderr)
            return 1
        print(f"      waypoints={len(waypoints)}  start_mode=random  start_area={start_area.wp_id}  ({time.time() - t0:.2f}s)")

    print("[4/7] Anchoring waypoints to lanelets")
    t0 = time.time()
    anchors = compute_waypoint_anchors(lanelets, waypoints)
    wp_with_anchor = {a.wp_id for a in anchors}
    skipped_no_anchor = [w.wp_id for w in waypoints if w.wp_id not in wp_with_anchor]
    print(
        f"      anchors={len(anchors)}  unique_wp_with_anchor={len(wp_with_anchor)}/{len(waypoints)}"
        f"  ({time.time() - t0:.2f}s)"
    )
    if skipped_no_anchor:
        print(f"      WARN: waypoints with NO anchor (no lanelet traversal): {skipped_no_anchor}")

    print("[5/7] Computing cost matrix (Dijkstra)")
    t0 = time.time()
    lanelet_penalties = load_lanelet_penalties(args.lanelet_penalties)
    if lanelet_penalties:
        applicable = {k: v for k, v in lanelet_penalties.items() if k in lanelets}
        print(f"      lanelet penalties: {len(applicable)} lanelets penalized from {args.lanelet_penalties}")
    traffic_lights = load_traffic_lights(args.traffic_lights)
    if traffic_lights.total_lanelets():
        wait = traffic_lights.expected_wait_for(next(iter(traffic_lights.by_lanelet)))
        print(
            f"      traffic-light model: {traffic_lights.total_lanelets()} lanelets, "
            f"E[wait]≈{wait:.2f}s each (replaces literal penalty for those)"
        )
    constraints_obj = load_constraints(args.constraints)
    if not constraints_obj.is_empty:
        print(f"      constraints: {constraints_obj.to_summary_dict()}")
    params = CostParams(
        speed_urban=args.speed_urban,
        speed_highway=args.speed_highway,
        efficiency_factor=args.efficiency,
        penalty_intersection_s=args.penalty_intersection,
        penalty_stopline_s=args.penalty_stopline,
        penalty_crosswalk_s=args.penalty_crosswalk,
        penalty_roundabout_s=args.penalty_roundabout,
        lanelet_penalties_s=lanelet_penalties,
        traffic_lights=traffic_lights if traffic_lights.total_lanelets() else None,
    )
    if args.start_mode == "default":
        # Single "start exit" lanelet: the chosen one, with exit_arc=0 (car starts
        # at the very beginning of the lanelet and traverses it fully).
        start_lanelet = lanelets[args.default_start_lanelet]
        start_xy_first = (float(start_lanelet.centerline[0, 0]), float(start_lanelet.centerline[0, 1]))
        start_exit_lanelets = [(args.default_start_lanelet, start_xy_first, 0.0)]
        print(f"      default-start lanelet: {args.default_start_lanelet} (start xy={start_xy_first})")
    else:
        start_exit_lanelets = find_start_exit_lanelets(lanelets, start_area)
        if not start_exit_lanelets:
            print("ERROR: no lanelet exits the start area.", file=sys.stderr)
            return 1
        dead_ends = [lid for lid, _, _ in start_exit_lanelets if not lanelets[lid].successor_ids]
        usable = [lid for lid, _, _ in start_exit_lanelets if lanelets[lid].successor_ids]
        print(f"      start_exit_lanelets={len(start_exit_lanelets)}: usable={usable}, dead_end={dead_ends}")
        if dead_ends:
            print(
                f"      WARN: {len(dead_ends)} start-exit lanelets are dead-ends in the OSM "
                f"(no successor). They will be ignored by the solver."
            )
    start_bonus = SCORE_RANDOM_START if args.start_mode == "random" else 0
    solver_budget = _math.inf if args.max_waypoints else args.time_budget
    if args.max_waypoints:
        print(f"      mode: MAX WAYPOINTS — budget constraint disabled (time_budget={args.time_budget:.0f}s used only for reporting)")

    # Multi-start applies only to random mode.
    use_multi_start = (
        args.start_mode == "random"
        and args.multi_start > 1
        and start_area is not None
    )
    multi_start_attempts: list[dict] | None = None
    multi_start_candidate = None
    if use_multi_start:
        print(f"[5b] Multi-start search: sampling {args.multi_start} poses in random polygon")
        from src.multi_start import run_multi_start as _run_multi_start  # local re-import
        ms = _run_multi_start(
            osm_path=osm_path,
            lanelets=lanelets,
            anchors=anchors,
            start_area=start_area,
            start_exit_lanelets=start_exit_lanelets,
            params=params,
            budget_s=solver_budget,
            restarts=args.restarts,
            sa_iterations=args.sa_iterations,
            seed=args.seed,
            n_samples=args.multi_start,
            cache_dir=None if args.no_cache else args.cache_dir,
            constraints=constraints_obj,
            verbose=args.verbose,
        )
        cost_matrix = ms.best_cost_matrix
        solution = type("Sol", (), {})()
        solution.sequence = list(ms.best_plan.waypoint_sequence)  # not actually used here
        multi_start_attempts = ms.attempts
        multi_start_candidate = ms.best_candidate
        # Rebuild plan/cost_matrix and re-solve to keep a uniform downstream path.
        start_exit_lanelets = [(ms.best_candidate.lanelet_id, ms.best_candidate.xy, ms.best_candidate.exit_arc_m)]
        cost_matrix = build_cost_matrix(
            lanelets, anchors, start_exit_lanelets, params, osm_path,
            cache_dir=None if args.no_cache else args.cache_dir, verbose=False,
        )
        print(f"      best candidate: lanelet {ms.best_candidate.lanelet_id} at {ms.best_candidate.xy}")
    else:
        cost_matrix = build_cost_matrix(
            lanelets, anchors, start_exit_lanelets, params, osm_path,
            cache_dir=None if args.no_cache else args.cache_dir, verbose=args.verbose,
        )
    print(f"      cost matrix shape={cost_matrix.D.shape}  ({time.time() - t0:.2f}s)")

    print("[6/7] Solving Orienteering Problem")
    t0 = time.time()
    N = len(anchors)
    scores = np.zeros(N + 1, dtype=float)
    scores[0] = float(start_bonus)
    for i in range(1, N + 1):
        scores[i] = float(SCORE_WAYPOINT)
    wp_id_per_node: list[str] = [""] + [a.wp_id for a in anchors]
    solution = solve_orienteering(
        cost_matrix.D,
        scores,
        wp_id_per_node,
        budget_s=solver_budget,
        restarts=args.restarts,
        sa_iterations=args.sa_iterations,
        seed=args.seed,
        verbose=args.verbose,
        constraints=constraints_obj if not constraints_obj.is_empty else None,
    )
    print(
        f"      sequence_len={len(solution.sequence)}  "
        f"score={int(solution.score)}  cost={solution.cost_s:.2f}s  ({time.time() - t0:.2f}s)"
    )

    if not constraints_obj.is_empty:
        ok, errors = validate_constraints(solution.sequence, wp_id_per_node, constraints_obj)
        if not ok:
            print(f"      WARN: constraint violations in best plan:")
            for err in errors:
                print(f"        - {err}")

    print("[7/7] Writing JSON + SVG")
    t0 = time.time()
    manual_pose = _parse_pose(args.start_pose)
    if manual_pose is not None:
        start_pose = manual_pose
    elif args.start_mode == "default":
        cl = lanelets[args.default_start_lanelet].centerline
        sx, sy = float(cl[0, 0]), float(cl[0, 1])
        syaw = _math.atan2(float(cl[1, 1] - cl[0, 1]), float(cl[1, 0] - cl[0, 0]))
        start_pose = (sx, sy, syaw)
    elif multi_start_candidate is not None:
        start_pose = (multi_start_candidate.xy[0], multi_start_candidate.xy[1], multi_start_candidate.yaw_rad)
    else:
        sx, sy, syaw, _ = pick_start_pose(lanelets, start_area)
        start_pose = (sx, sy, syaw)

    plan = build_plan(
        osm_path=osm_path,
        params=params,
        budget_s=args.time_budget,
        solution=solution,
        cost_matrix=cost_matrix,
        anchors=anchors,
        waypoints=waypoints,
        start_pose=start_pose,
        start_bonus=start_bonus,
    )
    write_plan_json(plan, out_json_path)
    render_plan_svg(
        plan, lanelets, waypoints, start_area, out_svg_path,
        lanelet_penalties=lanelet_penalties,
    )
    _write_plan_metadata(plan_name, out_json_path, args)
    print(f"      JSON: {out_json_path}")
    print(f"      SVG:  {out_svg_path}")
    print(f"      ({time.time() - t0:.2f}s)")

    print()
    print("=== SUMMARY ===")
    print(f"Plan name:                {plan_name}")
    print(f"Start mode:               {args.start_mode}" + (
        f"  (lanelet {args.default_start_lanelet})" if args.start_mode == "default" else ""
    ))
    print(f"Expected score:           {int(plan.expected_score)} pts")
    if start_bonus:
        print(f"   - Random start bonus:  {start_bonus} pts")
    print(f"   - Waypoints visited:   {plan.expected_waypoints_visited}/{len(waypoints)} ({plan.expected_waypoints_visited * SCORE_WAYPOINT} pts)")
    over_budget = plan.expected_total_time_s > args.time_budget + 1e-3
    budget_str = f"{plan.expected_total_time_s:.2f}s / {args.time_budget:.0f}s"
    if over_budget:
        budget_str += f"  (OVER budget by {plan.expected_total_time_s - args.time_budget:.1f}s)"
    print(f"Expected time:            {budget_str}")
    print(f"Expected distance:        {plan.expected_distance_m:.2f}m")
    print(f"First waypoint:           {plan.first_waypoint_id}")
    print(f"Start pose:               ({start_pose[0]:.3f}, {start_pose[1]:.3f}, {start_pose[2]:.3f} rad)")
    print(f"Skipped waypoints:        {len(plan.skipped_waypoints)}")
    if plan.skipped_waypoints:
        print(f"   {plan.skipped_waypoints}")
    return 0


def _resolve_output_paths(args) -> tuple[str, str, str]:
    """Determine the plan name and the JSON/SVG output paths.

    Precedence:
      1. --output-json / --output-svg (if both are explicit, both used as-is).
      2. --name as the plan folder under --plans-dir.
      3. Auto-generated timestamp name (with `_max` suffix in max-waypoints mode).
    """
    if args.name:
        plan_name = args.name
    else:
        plan_name = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.max_waypoints:
            plan_name += "_max"
    plan_dir = os.path.join(args.plans_dir, plan_name)
    os.makedirs(plan_dir, exist_ok=True)
    out_json = args.output_json or os.path.join(plan_dir, "plan.json")
    out_svg = args.output_svg or os.path.join(plan_dir, "plan.svg")
    return plan_name, out_json, out_svg


def _write_plan_metadata(plan_name: str, json_path: str, args) -> None:
    """Write a small metadata.json alongside the plan so it can be listed/sorted later."""
    plan_dir = os.path.dirname(json_path)
    try:
        with open(json_path) as fh:
            plan = _json.load(fh)
    except Exception:
        plan = {}
    meta = {
        "name": plan_name,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "mode": "max_waypoints" if args.max_waypoints else "budget",
        "start_mode": args.start_mode,
        "default_start_lanelet": args.default_start_lanelet if args.start_mode == "default" else None,
        "time_budget_s": args.time_budget,
        "expected_score": plan.get("expected_score"),
        "expected_total_time_s": plan.get("expected_total_time_s"),
        "expected_waypoints_visited": plan.get("expected_waypoints_visited"),
        "expected_distance_m": plan.get("expected_distance_m"),
        "first_waypoint_id": plan.get("first_waypoint_id"),
    }
    with open(os.path.join(plan_dir, "metadata.json"), "w") as fh:
        _json.dump(meta, fh, indent=2)


def _print_plans_list(plans_dir: str) -> None:
    if not os.path.isdir(plans_dir):
        print(f"No plans directory at {plans_dir}")
        return
    rows: list[dict] = []
    for entry in sorted(os.listdir(plans_dir)):
        plan_dir = os.path.join(plans_dir, entry)
        if not os.path.isdir(plan_dir):
            continue
        meta_path = os.path.join(plan_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as fh:
                    rows.append(_json.load(fh))
                continue
            except Exception:
                pass
        # Fallback to bare directory listing.
        rows.append({"name": entry, "created_at": "?", "mode": "?", "expected_score": "?",
                     "expected_total_time_s": "?", "expected_waypoints_visited": "?"})
    if not rows:
        print(f"(no plans found in {plans_dir})")
        return
    print(f"Plans in {plans_dir}:")
    header = f"{'NAME':<28} {'CREATED':<20} {'MODE':<14} {'SCORE':>6} {'TIME':>9} {'WPS':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        score = r.get("expected_score")
        t = r.get("expected_total_time_s")
        wps = r.get("expected_waypoints_visited")
        print(
            f"{str(r.get('name', '?'))[:28]:<28} "
            f"{str(r.get('created_at', '?'))[:19]:<20} "
            f"{str(r.get('mode', '?')):<14} "
            f"{(int(score) if isinstance(score, (int, float)) else '?'):>6} "
            f"{(f'{t:.1f}s' if isinstance(t, (int, float)) else '?'):>9} "
            f"{(str(wps) if wps is not None else '?'):>6}"
        )


if __name__ == "__main__":
    sys.exit(main())
