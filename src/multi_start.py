"""Multi-start optimization for the random-start mode.

The default `pick_start_pose()` chooses ONE pose inside the random start polygon.
This module samples N poses, computes a full plan for each, and returns the best.
A pose is paired with the start-exit lanelet whose centerline is closest AND
heading-compatible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from .cost_matrix import CostMatrix, CostParams, build_cost_matrix
from .geometry import point_in_polygon
from .orienteering_solver import solve_orienteering
from .plan_writer import SCORE_RANDOM_START, SCORE_WAYPOINT, Plan, build_plan
from .topology import Lanelet
from .waypoint_anchor import WaypointAnchor, WaypointArea


@dataclass(frozen=True)
class StartCandidate:
    xy: tuple[float, float]
    yaw_rad: float
    lanelet_id: str
    exit_arc_m: float


def sample_poses_in_polygon(
    polygon: np.ndarray, n: int, rng: random.Random
) -> list[tuple[float, float]]:
    """Rejection sampling inside the polygon's bbox."""
    x_min = float(polygon[:, 0].min())
    x_max = float(polygon[:, 0].max())
    y_min = float(polygon[:, 1].min())
    y_max = float(polygon[:, 1].max())
    out: list[tuple[float, float]] = []
    tries = 0
    max_tries = n * 50
    while len(out) < n and tries < max_tries:
        tries += 1
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        if point_in_polygon((x, y), polygon):
            out.append((x, y))
    return out


def pose_to_candidate(
    xy: tuple[float, float],
    lanelets: dict[str, Lanelet],
    start_exit_lanelets: list[tuple[str, tuple[float, float], float]],
    max_lateral_m: float = 0.6,
) -> StartCandidate | None:
    """Pair (x, y) with the closest start-exit lanelet whose centerline passes
    near the point. Returns the projection arc length on that lanelet.
    """
    best: tuple[float, str, float, tuple[float, float], float] | None = None
    for lid, _xy, _arc in start_exit_lanelets:
        l = lanelets.get(lid)
        if l is None or l.centerline.shape[0] < 2:
            continue
        cl = l.centerline
        # find closest centerline sample to (x,y)
        d2 = (cl[:, 0] - xy[0]) ** 2 + (cl[:, 1] - xy[1]) ** 2
        i = int(np.argmin(d2))
        lateral = float(np.sqrt(d2[i]))
        if lateral > max_lateral_m:
            continue
        # arc length from start to sample i
        diffs = np.linalg.norm(np.diff(cl[: i + 1], axis=0), axis=1) if i > 0 else np.zeros(0)
        arc = float(diffs.sum())
        # heading at this sample
        j = min(cl.shape[0] - 1, i + 1)
        if j == i:
            j = max(0, i - 1)
            dx = cl[i, 0] - cl[j, 0]
            dy = cl[i, 1] - cl[j, 1]
        else:
            dx = cl[j, 0] - cl[i, 0]
            dy = cl[j, 1] - cl[i, 1]
        yaw = math.atan2(float(dy), float(dx))
        chosen_xy = (float(cl[i, 0]), float(cl[i, 1]))
        score = lateral  # smaller is better
        if best is None or score < best[0]:
            best = (score, lid, arc, chosen_xy, yaw)
    if best is None:
        return None
    _score, lid, arc, chosen_xy, yaw = best
    return StartCandidate(xy=chosen_xy, yaw_rad=yaw, lanelet_id=lid, exit_arc_m=arc)


@dataclass
class MultiStartResult:
    best_plan: Plan
    best_cost_matrix: CostMatrix
    best_candidate: StartCandidate
    attempts: list[dict]


def run_multi_start(
    *,
    osm_path: str,
    lanelets: dict[str, Lanelet],
    anchors: list[WaypointAnchor],
    start_area: WaypointArea,
    start_exit_lanelets: list[tuple[str, tuple[float, float], float]],
    params: CostParams,
    budget_s: float,
    restarts: int,
    sa_iterations: int,
    seed: int,
    n_samples: int,
    cache_dir: str | None,
    constraints=None,
    verbose: bool = False,
) -> MultiStartResult:
    """Sample N poses, solve OP for each, keep best by score (tie-break by cost)."""
    rng = random.Random(seed)
    raw_xy = sample_poses_in_polygon(start_area.polygon_xy, n_samples, rng)
    candidates: list[StartCandidate] = []
    for xy in raw_xy:
        c = pose_to_candidate(xy, lanelets, start_exit_lanelets)
        if c is not None:
            candidates.append(c)
    if not candidates:
        raise RuntimeError("Multi-start: no valid candidates inside the random start polygon.")
    if verbose:
        print(f"  multi-start: {len(candidates)} valid candidates from {n_samples} samples")

    N = len(anchors)
    scores = np.zeros(N + 1, dtype=float)
    scores[0] = float(SCORE_RANDOM_START)
    for i in range(1, N + 1):
        scores[i] = float(SCORE_WAYPOINT)
    wp_id_per_node: list[str] = [""] + [a.wp_id for a in anchors]

    best: tuple[float, float, Plan, CostMatrix, StartCandidate] | None = None
    attempts: list[dict] = []
    for idx, cand in enumerate(candidates):
        sel = [(cand.lanelet_id, cand.xy, cand.exit_arc_m)]
        cm = build_cost_matrix(
            lanelets, anchors, sel, params, osm_path,
            cache_dir=cache_dir, verbose=False,
        )
        sol = solve_orienteering(
            cm.D, scores, wp_id_per_node, budget_s=budget_s,
            restarts=restarts, sa_iterations=sa_iterations,
            seed=seed + idx * 17, verbose=False, constraints=constraints,
        )
        plan = build_plan(
            osm_path=osm_path, params=params, budget_s=budget_s,
            solution=sol, cost_matrix=cm, anchors=anchors, waypoints=[],
            start_pose=(cand.xy[0], cand.xy[1], cand.yaw_rad),
            start_bonus=SCORE_RANDOM_START,
        )
        attempts.append({
            "idx": idx,
            "xy": [round(cand.xy[0], 3), round(cand.xy[1], 3)],
            "yaw_rad": round(cand.yaw_rad, 3),
            "lanelet": cand.lanelet_id,
            "score": int(plan.expected_score),
            "time_s": round(plan.expected_total_time_s, 1),
            "waypoints": plan.expected_waypoints_visited,
        })
        key = (-plan.expected_score, plan.expected_total_time_s)
        if best is None or key < (-best[0], best[1]):
            best = (plan.expected_score, plan.expected_total_time_s, plan, cm, cand)
        if verbose:
            print(f"  [pose {idx + 1}/{len(candidates)}] lid={cand.lanelet_id} "
                  f"score={int(plan.expected_score)} t={plan.expected_total_time_s:.0f}s")

    assert best is not None
    return MultiStartResult(
        best_plan=best[2],
        best_cost_matrix=best[3],
        best_candidate=best[4],
        attempts=attempts,
    )
