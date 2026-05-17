"""Build the cost matrix between waypoint anchors using Dijkstra on the lanelet graph.

Cost is in seconds: length / v_effective + penalty_s(attr).
The matrix is asymmetric (one-way lanelets).
Node 0 is the virtual start node; nodes 1..N are anchors.
The dijkstra runs on lanelet endpoints; intra-lanelet costs are added explicitly
based on (entry_arc, exit_arc) of each anchor.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .topology import (
    ATTR_CROSSWALK,
    ATTR_INTERSECTION,
    ATTR_ROUNDABOUT,
    ATTR_STOPLINE,
    Lanelet,
)
from .traffic_light import TrafficLightModel
from .waypoint_anchor import WaypointAnchor


INF = float("inf")


def load_lanelet_penalties(path: str | None) -> dict[str, float]:
    """Load per-lanelet extra time penalties from JSON. Returns {} if missing.

    Keys starting with `_` are treated as comments and ignored.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        data = json.load(fh)
    out: dict[str, float] = {}
    for k, v in data.items():
        if str(k).startswith("_"):
            continue
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class CostParams:
    speed_urban: float = 0.2
    speed_highway: float = 0.4
    efficiency_factor: float = 0.7
    penalty_intersection_s: float = 1.5
    penalty_stopline_s: float = 4.0
    penalty_crosswalk_s: float = 1.0
    penalty_roundabout_s: float = 1.0
    # Per-lanelet extra penalties (parking, traffic lights, etc.). Lanelets that
    # appear in `traffic_lights` are overridden with their expected wait time.
    lanelet_penalties_s: dict[str, float] = field(default_factory=dict)
    traffic_lights: TrafficLightModel | None = None

    def to_dict(self) -> dict:
        tl_summary = self.traffic_lights.to_summary_dict() if self.traffic_lights else None
        return {
            "speed_urban": self.speed_urban,
            "speed_highway": self.speed_highway,
            "efficiency_factor": self.efficiency_factor,
            "penalty_intersection_s": self.penalty_intersection_s,
            "penalty_stopline_s": self.penalty_stopline_s,
            "penalty_crosswalk_s": self.penalty_crosswalk_s,
            "penalty_roundabout_s": self.penalty_roundabout_s,
            "lanelet_penalties_s": dict(sorted(self.lanelet_penalties_s.items())),
            "traffic_lights": tl_summary,
        }


def lanelet_speed_mps(lanelet: Lanelet, params: CostParams) -> float:
    v = params.speed_highway if lanelet.is_highway else params.speed_urban
    return v * params.efficiency_factor


def lanelet_penalty_s(lanelet: Lanelet, params: CostParams) -> float:
    """Combined penalty.

    Components:
      - attr_pen: from `attribute_kind` (stopline, intersection, crosswalk, roundabout).
      - lanelet_pen: per-lanelet literal from `lanelet_penalties_s`. If the lanelet
        also has a traffic-light cycle, that literal is REPLACED by E[wait]
        (so the same lanelet doesn't get double-charged).
    """
    attr_pen = 0.0
    if lanelet.attribute_kind == ATTR_STOPLINE:
        attr_pen = params.penalty_stopline_s
    elif lanelet.attribute_kind == ATTR_INTERSECTION:
        attr_pen = params.penalty_intersection_s
    elif lanelet.attribute_kind == ATTR_CROSSWALK:
        attr_pen = params.penalty_crosswalk_s
    elif lanelet.attribute_kind == ATTR_ROUNDABOUT:
        attr_pen = params.penalty_roundabout_s

    if params.traffic_lights is not None and params.traffic_lights.has(lanelet.lanelet_id):
        # Replace the literal per-lanelet penalty with the expected wait time.
        lanelet_pen = params.traffic_lights.expected_wait_for(lanelet.lanelet_id)
    else:
        lanelet_pen = float(params.lanelet_penalties_s.get(lanelet.lanelet_id, 0.0))
    return attr_pen + lanelet_pen


def lanelet_total_cost_s(lanelet: Lanelet, params: CostParams) -> float:
    """Time (s) to traverse the entire lanelet, including attribute penalty.
    Clamped to >= 0 so negative bonuses can't break Dijkstra invariants."""
    v = max(lanelet_speed_mps(lanelet, params), 1e-6)
    return max(0.0, float(lanelet.length_m / v) + lanelet_penalty_s(lanelet, params))


def lanelet_partial_cost_s(
    lanelet: Lanelet, params: CostParams, length_m: float
) -> float:
    """Time (s) to traverse a partial section of length_m of the lanelet. Penalty is
    charged proportionally to the fraction of the lanelet covered. Clamped to >= 0."""
    v = max(lanelet_speed_mps(lanelet, params), 1e-6)
    if lanelet.length_m <= 1e-9:
        return max(0.0, float(length_m / v))
    frac = max(0.0, min(1.0, length_m / lanelet.length_m))
    return max(0.0, float(length_m / v) + lanelet_penalty_s(lanelet, params) * frac)


@dataclass
class CostMatrix:
    D: np.ndarray  # (N+1, N+1), seconds
    distance_m: np.ndarray  # (N+1, N+1), physical distance
    path_lanelets: dict[tuple[int, int], list[str]]  # (i, j) -> lanelet ids traversed (excluding self crossings)
    index_to_anchor: list[WaypointAnchor | None]  # length N+1, index 0 is start (None)
    start_exit_lanelet_ids: list[str] = field(default_factory=list)


def _hash_params(
    osm_path: str,
    params: CostParams,
    num_anchors: int,
    start_exit_lanelets: list[tuple[str, tuple[float, float], float]] | None = None,
) -> str:
    try:
        mtime = os.path.getmtime(osm_path)
    except OSError:
        mtime = 0.0
    start_key = []
    if start_exit_lanelets:
        for lid, _xy, arc in start_exit_lanelets:
            start_key.append((str(lid), round(float(arc), 4)))
        start_key.sort()
    payload = json.dumps(
        {
            "osm_path": os.path.abspath(osm_path),
            "mtime": mtime,
            "params": params.to_dict(),
            "num_anchors": num_anchors,
            "start_exit": start_key,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_cache(cache_dir: str, key: str) -> dict | None:
    path = os.path.join(cache_dir, f"cost_matrix_{key}.npz")
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        return {
            "D": data["D"],
            "distance_m": data["distance_m"],
            "path_lanelets": data["path_lanelets"].item(),
        }
    except Exception:
        return None


def _save_cache(cache_dir: str, key: str, D, distance_m, path_lanelets) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"cost_matrix_{key}.npz")
    np.savez(path, D=D, distance_m=distance_m, path_lanelets=np.array(path_lanelets, dtype=object))


def build_cost_matrix(
    lanelets: dict[str, Lanelet],
    anchors: list[WaypointAnchor],
    start_exit_lanelets: list[tuple[str, tuple[float, float], float]],
    params: CostParams,
    osm_path: str,
    cache_dir: str | None = None,
    verbose: bool = False,
) -> CostMatrix:
    """Build the (N+1) x (N+1) cost matrix.

    Node 0 is the virtual start. Nodes 1..N correspond to anchors[0..N-1].
    For each anchor i, the "exit point" is at exit_arc_m of anchor.lanelet_id.
    For each anchor j, the "entry point" is at entry_arc_m of anchor.lanelet_id.
    D[i, j] = time to go from anchor i's exit to anchor j's entry.
    D[0, j] = time from start to anchor j's entry (via any start-exit-lanelet).
    """
    N = len(anchors)
    cache_key = (
        _hash_params(osm_path, params, N, start_exit_lanelets) if cache_dir else None
    )

    if cache_dir and cache_key:
        cached = _load_cache(cache_dir, cache_key)
        if cached is not None:
            if verbose:
                print(f"[cost_matrix] Loaded cache: {cache_key}")
            return CostMatrix(
                D=cached["D"],
                distance_m=cached["distance_m"],
                path_lanelets=cached["path_lanelets"],
                index_to_anchor=[None] + list(anchors),
                start_exit_lanelet_ids=[lid for lid, _, _ in start_exit_lanelets],
            )

    # Precompute total cost per lanelet
    total_cost = {lid: lanelet_total_cost_s(l, params) for lid, l in lanelets.items()}
    speed_mps = {lid: lanelet_speed_mps(l, params) for lid, l in lanelets.items()}
    penalty_s = {lid: lanelet_penalty_s(l, params) for lid, l in lanelets.items()}

    # Dijkstra: source = (lanelet_id, "at_start_of_lanelet").
    # We want shortest path from (src_lid at exit_arc) to (dst_lid at entry_arc) without
    # re-entering src_lid. To avoid recomputing for each source, we Dijkstra from each
    # source lanelet's "end" to all other lanelets' "start", and use partial costs at
    # the endpoints.

    # adjacency: lanelet_id -> list of successor ids
    adj: dict[str, list[str]] = {lid: list(l.successor_ids) for lid, l in lanelets.items()}

    def dijkstra_from_lanelet_start(src: str) -> tuple[dict[str, float], dict[str, str]]:
        """Shortest time from "start of src" to "start of every other lanelet".
        Cost from start-of-src to start-of-x = (full cost of src) + (full cost of each intermediate).
        Wait — we want time from end-of-src to start-of-dst. That's just the time of
        whatever intermediate lanelets between them (NOT including src), so we should
        start Dijkstra at the successors of src.
        Implement: distances[lid] = time from end-of-src to start-of-lid. Initialized as inf except
        distances[succ] = total_cost[succ] - ... no, we want "start of succ" = 0 (the auto is now
        at the start of succ after exiting src). Then to reach start of x deeper, add total_cost
        of the lanelets you traverse before x. So distances[succ] = 0, and the edge cost from
        a lanelet u to a successor v is total_cost[u] (you fully traverse u to get to v's start).
        """
        dist: dict[str, float] = {}
        prev: dict[str, str] = {}
        # Initial: from end of src, we are at start of each successor at cost 0.
        heap: list[tuple[float, str]] = []
        for succ in adj.get(src, []):
            if succ == src:
                continue
            d_new = 0.0
            if succ not in dist or d_new < dist[succ]:
                dist[succ] = d_new
                heapq.heappush(heap, (d_new, succ))
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, INF):
                continue
            for v in adj.get(u, []):
                if v == src:
                    continue
                nd = d + total_cost[u]
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, prev

    # Index mapping
    # node 0 = virtual start
    # node 1..N = anchors[0..N-1]
    D = np.full((N + 1, N + 1), INF, dtype=float)
    distance_m = np.full((N + 1, N + 1), INF, dtype=float)
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(distance_m, 0.0)
    path_lanelets: dict[tuple[int, int], list[str]] = {}

    # Compute D[i, j] for i in 1..N: from anchor i's exit to anchor j's entry
    # Precompute Dijkstra from each unique source lanelet (= anchor.lanelet_id for i)
    unique_src_lids = sorted({a.lanelet_id for a in anchors})
    dij_cache: dict[str, tuple[dict[str, float], dict[str, str]]] = {}
    for src in unique_src_lids:
        dij_cache[src] = dijkstra_from_lanelet_start(src)

    for i, a in enumerate(anchors, start=1):
        # Cost from anchor i's exit to start of its own lanelet (for same-lanelet hops)
        src_lid = a.lanelet_id
        src_lanelet = lanelets[src_lid]
        # time from exit_arc to end of src lanelet:
        remaining_src_m = max(0.0, src_lanelet.length_m - a.exit_arc_m)
        time_remaining_src = lanelet_partial_cost_s(src_lanelet, params, remaining_src_m)
        dist_src, prev_src = dij_cache[src_lid]

        for j, b in enumerate(anchors, start=1):
            if i == j:
                continue
            dst_lid = b.lanelet_id
            dst_lanelet = lanelets[dst_lid]
            # Same lanelet special case: i and j on the same lanelet
            if src_lid == dst_lid:
                if b.entry_arc_m >= a.exit_arc_m:
                    # Travel forward in the same lanelet
                    segment_m = b.entry_arc_m - a.exit_arc_m
                    t = lanelet_partial_cost_s(src_lanelet, params, segment_m)
                    D[i, j] = t
                    distance_m[i, j] = segment_m
                    path_lanelets[(i, j)] = [src_lid]
                else:
                    # Must loop around — Dijkstra
                    if dst_lid in dist_src:
                        time_to_dst_start = dist_src[dst_lid]
                        time_in_dst = lanelet_partial_cost_s(
                            dst_lanelet, params, b.entry_arc_m
                        )
                        total_t = time_remaining_src + time_to_dst_start + time_in_dst
                        D[i, j] = total_t
                        # Build path
                        path = _reconstruct_path(prev_src, dst_lid)
                        path.insert(0, src_lid)
                        path_lanelets[(i, j)] = path
                        distance_m[i, j] = _path_distance(path, lanelets, src_partial=remaining_src_m, dst_partial=b.entry_arc_m)
                continue

            # Different lanelets
            if dst_lid in dist_src:
                time_to_dst_start = dist_src[dst_lid]
                time_in_dst = lanelet_partial_cost_s(dst_lanelet, params, b.entry_arc_m)
                total_t = time_remaining_src + time_to_dst_start + time_in_dst
                D[i, j] = total_t
                path = _reconstruct_path(prev_src, dst_lid)
                # Path starts at src_lid (we cross from a.exit through end of src into adjacency)
                # path here is the sequence of "intermediate" lanelets reached AFTER src_lid.
                # However due to how dist_src was computed (initial successors get dist 0), the
                # path doesn't include src_lid itself; prepend it.
                path.insert(0, src_lid)
                path_lanelets[(i, j)] = path
                distance_m[i, j] = _path_distance(
                    path, lanelets,
                    src_partial=remaining_src_m,
                    dst_partial=b.entry_arc_m,
                )

    # D[0, j]: from virtual start to anchor j's entry.
    # Try each start_exit_lanelet as the first lanelet.
    for j, b in enumerate(anchors, start=1):
        dst_lid = b.lanelet_id
        dst_lanelet = lanelets[dst_lid]
        best_t = INF
        best_d = INF
        best_path: list[str] = []
        for s_lid, _s_xy, s_exit_arc in start_exit_lanelets:
            s_lanelet = lanelets[s_lid]
            # Time from start pose (at s_exit_arc on s_lid) to end of s_lid
            remaining_s_m = max(0.0, s_lanelet.length_m - s_exit_arc)
            time_remaining_s = lanelet_partial_cost_s(s_lanelet, params, remaining_s_m)
            if s_lid == dst_lid:
                if b.entry_arc_m >= s_exit_arc:
                    segment_m = b.entry_arc_m - s_exit_arc
                    t = lanelet_partial_cost_s(s_lanelet, params, segment_m)
                    d = segment_m
                    path = [s_lid]
                    if t < best_t:
                        best_t = t
                        best_d = d
                        best_path = path
                    continue
            dist_s, prev_s = dij_cache.get(s_lid) or dijkstra_from_lanelet_start(s_lid)
            if dst_lid not in dist_s:
                continue
            time_to_dst_start = dist_s[dst_lid]
            time_in_dst = lanelet_partial_cost_s(dst_lanelet, params, b.entry_arc_m)
            total_t = time_remaining_s + time_to_dst_start + time_in_dst
            if total_t < best_t:
                best_t = total_t
                path = _reconstruct_path(prev_s, dst_lid)
                path.insert(0, s_lid)
                best_path = path
                best_d = _path_distance(
                    path, lanelets, src_partial=remaining_s_m, dst_partial=b.entry_arc_m
                )
        if best_t < INF:
            D[0, j] = best_t
            distance_m[0, j] = best_d
            path_lanelets[(0, j)] = best_path

    if cache_dir and cache_key:
        _save_cache(cache_dir, cache_key, D, distance_m, path_lanelets)
        if verbose:
            print(f"[cost_matrix] Saved cache: {cache_key}")

    return CostMatrix(
        D=D,
        distance_m=distance_m,
        path_lanelets=path_lanelets,
        index_to_anchor=[None] + list(anchors),
        start_exit_lanelet_ids=[lid for lid, _, _ in start_exit_lanelets],
    )


def _reconstruct_path(prev: dict[str, str], target: str) -> list[str]:
    if target not in prev:
        # target itself is the first successor reached directly from src; no intermediates
        return [target]
    path: list[str] = []
    cur: str | None = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


def _path_distance(
    path: list[str], lanelets: dict[str, Lanelet],
    src_partial: float, dst_partial: float,
) -> float:
    if not path:
        return 0.0
    if len(path) == 1:
        return float(min(src_partial, lanelets[path[0]].length_m))
    total = float(src_partial)
    for lid in path[1:-1]:
        total += float(lanelets[lid].length_m)
    total += float(dst_partial)
    return total
