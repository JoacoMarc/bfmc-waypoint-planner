"""Serialize the OP solution into a structured JSON plan.

The plan contains:
  - params: solver parameters used.
  - first_waypoint_id: wp_id of the first waypoint visited.
  - start_pose: (x, y, yaw_rad) within the start area.
  - waypoint_sequence: ordered list with order, wp_id, anchor_lanelet_id, entry/exit_xy, score, cumulative_time_s.
  - lanelet_sequence: ordered list of lanelet IDs that conform the full route.
  - skipped_waypoints: wp_ids that were not visited.
  - expected_total_time_s, expected_score, expected_distance_m, expected_waypoints_visited.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass

from .cost_matrix import CostMatrix, CostParams
from .orienteering_solver import OPSolution
from .waypoint_anchor import WaypointAnchor, WaypointArea


SCORE_RANDOM_START = 15
SCORE_WAYPOINT = 3


@dataclass
class Plan:
    osm_path: str
    params: CostParams
    budget_s: float
    start_pose: tuple[float, float, float]
    first_waypoint_id: str | None
    waypoint_sequence: list[dict]
    lanelet_sequence: list[str]
    skipped_waypoints: list[str]
    expected_total_time_s: float
    expected_score: float
    expected_distance_m: float
    expected_waypoints_visited: int


def build_plan(
    osm_path: str,
    params: CostParams,
    budget_s: float,
    solution: OPSolution,
    cost_matrix: CostMatrix,
    anchors: list[WaypointAnchor],
    waypoints: list[WaypointArea],
    start_pose: tuple[float, float, float],
    start_bonus: int = SCORE_RANDOM_START,
) -> Plan:
    seq = solution.sequence
    waypoint_sequence: list[dict] = []
    lanelet_sequence: list[str] = []
    cumulative_time = 0.0
    cumulative_distance = 0.0
    first_wp_id: str | None = None
    visited_wp_ids: set[str] = set()
    score = 0.0

    for order_idx in range(1, len(seq)):
        from_node = seq[order_idx - 1]
        to_node = seq[order_idx]
        anchor = cost_matrix.index_to_anchor[to_node]
        if anchor is None:
            continue
        t = float(cost_matrix.D[from_node, to_node])
        d = float(cost_matrix.distance_m[from_node, to_node])
        cumulative_time += t
        cumulative_distance += d
        wp_id = anchor.wp_id
        if first_wp_id is None:
            first_wp_id = wp_id
        if wp_id not in visited_wp_ids:
            visited_wp_ids.add(wp_id)
            score += SCORE_WAYPOINT
        waypoint_sequence.append(
            {
                "order": order_idx,
                "wp_id": wp_id,
                "anchor_lanelet_id": anchor.lanelet_id,
                "entry_xy": [round(anchor.entry_xy[0], 4), round(anchor.entry_xy[1], 4)],
                "exit_xy": [round(anchor.exit_xy[0], 4), round(anchor.exit_xy[1], 4)],
                "score": SCORE_WAYPOINT,
                "edge_time_s": round(t, 3),
                "edge_distance_m": round(d, 3),
                "cumulative_time_s": round(cumulative_time, 3),
            }
        )
        edge_lanelets = list(cost_matrix.path_lanelets.get((from_node, to_node), []))
        # Avoid duplicating the joining lanelet
        for lid in edge_lanelets:
            if not lanelet_sequence or lanelet_sequence[-1] != lid:
                lanelet_sequence.append(lid)

    # Add start bonus (15 pts for random start mode, 0 for default start)
    score += float(start_bonus)

    all_wp_ids = [w.wp_id for w in waypoints]
    skipped = [wid for wid in all_wp_ids if wid not in visited_wp_ids]

    return Plan(
        osm_path=osm_path,
        params=params,
        budget_s=budget_s,
        start_pose=start_pose,
        first_waypoint_id=first_wp_id,
        waypoint_sequence=waypoint_sequence,
        lanelet_sequence=lanelet_sequence,
        skipped_waypoints=skipped,
        expected_total_time_s=cumulative_time,
        expected_score=score,
        expected_distance_m=cumulative_distance,
        expected_waypoints_visited=len(visited_wp_ids),
    )


def write_plan_json(plan: Plan, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "version": 1,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "osm_path": plan.osm_path,
        "params": {
            **plan.params.to_dict(),
            "time_budget_s": plan.budget_s,
            "score_random_start": SCORE_RANDOM_START,
            "score_waypoint": SCORE_WAYPOINT,
        },
        "first_waypoint_id": plan.first_waypoint_id,
        "start_pose": {
            "x": round(plan.start_pose[0], 4),
            "y": round(plan.start_pose[1], 4),
            "yaw_rad": round(plan.start_pose[2], 4),
        },
        "waypoint_sequence": plan.waypoint_sequence,
        "lanelet_sequence": plan.lanelet_sequence,
        "skipped_waypoints": plan.skipped_waypoints,
        "expected_total_time_s": round(plan.expected_total_time_s, 3),
        "expected_score": int(plan.expected_score),
        "expected_distance_m": round(plan.expected_distance_m, 3),
        "expected_waypoints_visited": plan.expected_waypoints_visited,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
