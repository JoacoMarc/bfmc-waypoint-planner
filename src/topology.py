"""Build lanelets with centerlines, attributes and successor/predecessor topology.

Hybrid matching strategy:
  Stage A (centerline-endpoint tier system):
    Each lanelet has a start/end position (centerline endpoints) and a tangent
    at each endpoint. A candidate B is a successor of A when B.start is near
    A.end AND headings are compatible AND B.start is "in front of" A.end.

    Tiers:
      tight  (gap ≤ 0.05 m): exact endpoint contact, heading ≤ 95°.
      lane   (≤ 0.35 m):     within a lane width, heading ≤ 95°.
      intersection (≤ 1.6 m): long gap, heading ≤ 95°, near-colinear forward
                              displacement (dot ≥ 0.7). Captures the BFMC pattern
                              of "approach lanelet → turn lanelet" pairs.

    The first non-empty tier wins; tighter tiers take precedence.

  Stage B (boundary-node sharing fallback):
    When a lanelet has no successors from stage A and it shares ≥1 boundary
    endpoint node ID with another lanelet whose start is reasonably close, we
    register the connection. This catches BFMC's "junction crossing" pattern
    where the OSM links lanelets via shared boundary nodes across an intersection.

  Post-filter:
    Reject any successor where A.end and B.start coincide (< 0.10 m) AND headings
    differ by > 90°. This is the false-positive pattern at intersection nodes
    where two turn lanelets meet from opposite directions (e.g. 888 → 865).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .geometry import densify_polyline, resample_polyline
from .osm_parser import OsmDocument, OsmLanelet, OsmNode, OsmWay


_GAP_TIGHT_M = 0.05
_GAP_LANE_M = 0.35
_GAP_INTERSECTION_M = 1.6

_HEADING_MAX_RAD = math.radians(95.0)
_FORWARD_DOT_MIN_TIGHT = 0.0
_FORWARD_DOT_MIN_INTERSECTION = 0.7

# Stage B (boundary-node sharing fallback) parameters
_BOUNDARY_FALLBACK_GAP_M = 2.5
_BOUNDARY_FALLBACK_HEADING_RAD = math.radians(120.0)

# Post-filter: reject coincident-endpoint, opposite-heading false positives.
_REJECT_OPPOSITE_GAP_M = 0.10
_OPPOSITE_HEADING_RAD = math.radians(90.0)


ATTR_NORMAL = "normal"
ATTR_INTERSECTION = "intersection"
ATTR_STOPLINE = "stopline"
ATTR_CROSSWALK = "crosswalk"
ATTR_ROUNDABOUT = "roundabout"


@dataclass(frozen=True)
class Lanelet:
    lanelet_id: str
    centerline: np.ndarray
    length_m: float
    speed_limit: float
    is_highway: bool
    turn_direction: str | None
    attribute_kind: str
    one_way: bool
    successor_ids: tuple[str, ...]
    predecessor_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Endpoint:
    xy: np.ndarray
    tangent: np.ndarray


def load_topology_overrides(path: str | None) -> dict:
    """Load topology overrides from a JSON file. Returns empty dict if path is None or missing."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def load_topology_ground_truth(path: str | None) -> dict[str, list[str]] | None:
    """Load a full topology map ({lanelet_id: [succ_ids]}) from JSON.

    When provided, this REPLACES the OSM-inferred successor/predecessor topology.
    Predecessors are derived automatically from the successor map.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    # Reject if it looks like an overrides file (has add_successors etc.)
    if "add_successors" in data or "remove_successors" in data:
        return None
    return {str(k): [str(s) for s in v] for k, v in data.items()}


def build_lanelets(
    doc: OsmDocument,
    step_m: float = 0.05,
    overrides: dict | None = None,
    ground_truth: dict[str, list[str]] | None = None,
) -> dict[str, Lanelet]:
    overrides = overrides or {}
    specs: list[dict] = []
    for rel in doc.lanelets.values():
        left_way = doc.ways.get(rel.left_way_id)
        right_way = doc.ways.get(rel.right_way_id)
        if left_way is None or right_way is None:
            continue
        left = _way_to_polyline(left_way, doc.nodes)
        right = _way_to_polyline(right_way, doc.nodes)
        if left.shape[0] < 2 or right.shape[0] < 2:
            continue
        right_aligned = _align_direction(left, right)
        centerline = _centerline_from_bounds(left, right_aligned, step_m=step_m)
        if centerline.shape[0] < 2:
            continue
        specs.append({"rel": rel, "centerline": centerline, "left_way": left_way, "right_way": right_way})

    lanelets_partial: dict[str, dict] = {}
    starts: dict[str, _Endpoint] = {}
    ends: dict[str, _Endpoint] = {}
    boundary_node_sets: dict[str, set[str]] = {}
    boundary_endpoint_nodes: dict[str, set[str]] = {}
    for spec in specs:
        rel: OsmLanelet = spec["rel"]
        centerline = spec["centerline"]
        length = float(np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))
        tags = rel.tags
        try:
            speed_limit = float(tags.get("speed_limit", "0.2"))
        except (TypeError, ValueError):
            speed_limit = 0.2
        is_highway = speed_limit > 1.0
        turn = tags.get("turn_direction", "").strip().lower() or None
        attribute = _attribute_from_tags(tags)
        one_way = tags.get("one_way", "yes").lower() == "yes"

        lanelets_partial[rel.relation_id] = {
            "centerline": centerline,
            "length_m": length,
            "speed_limit": speed_limit,
            "is_highway": is_highway,
            "turn_direction": turn,
            "attribute_kind": attribute,
            "one_way": one_way,
        }
        starts[rel.relation_id] = _endpoint_at_start(centerline)
        ends[rel.relation_id] = _endpoint_at_end(centerline)
        left_way: OsmWay = spec["left_way"]
        right_way: OsmWay = spec["right_way"]
        boundary_node_sets[rel.relation_id] = set(left_way.node_ids) | set(right_way.node_ids)
        boundary_endpoint_nodes[rel.relation_id] = {
            left_way.node_ids[0], left_way.node_ids[-1],
            right_way.node_ids[0], right_way.node_ids[-1],
        }

    # Stage A: centerline-endpoint tier matching
    successors: dict[str, list[str]] = {lid: [] for lid in lanelets_partial}
    predecessors: dict[str, list[str]] = {lid: [] for lid in lanelets_partial}
    for a_id, a_end in ends.items():
        succ = _candidates_after(a_end, starts, exclude_id=a_id)
        successors[a_id] = list(succ)
    for b_id, b_start in starts.items():
        for a_id, a_end in ends.items():
            if a_id == b_id:
                continue
            if _is_connection(a_end, b_start):
                predecessors[b_id].append(a_id)
        # Dedup
        seen: set[str] = set()
        unique: list[str] = []
        for x in predecessors[b_id]:
            if x in seen:
                continue
            seen.add(x)
            unique.append(x)
        predecessors[b_id] = unique

    # Stage B: boundary-node sharing fallback for lanelets that still have no successor
    for a_id in lanelets_partial:
        if successors[a_id]:
            continue
        a_end = ends[a_id]
        a_nodes = boundary_endpoint_nodes[a_id]
        candidates: list[tuple[float, str]] = []
        for b_id in lanelets_partial:
            if b_id == a_id:
                continue
            # Require they share at least one boundary endpoint node ID
            if not (a_nodes & boundary_endpoint_nodes[b_id]):
                continue
            b_start = starts[b_id]
            gap = float(np.linalg.norm(b_start.xy - a_end.xy))
            if gap > _BOUNDARY_FALLBACK_GAP_M:
                continue
            # Forward dot must be at least slightly positive (B is in front of A)
            disp = b_start.xy - a_end.xy
            disp_norm = float(np.linalg.norm(disp))
            if disp_norm > 1e-6:
                forward = float(np.dot(disp / disp_norm, a_end.tangent))
                if forward < 0.0:
                    continue
            candidates.append((gap, b_id))
        candidates.sort(key=lambda x: x[0])
        for _, b_id in candidates:
            if b_id not in successors[a_id]:
                successors[a_id].append(b_id)
                if a_id not in predecessors[b_id]:
                    predecessors[b_id].append(a_id)

    # Post-filter: reject coincident-endpoint opposite-direction false positives.
    def is_opposite_endpoint(a_id: str, b_id: str) -> bool:
        a_end_xy = ends[a_id].xy
        b_start_xy = starts[b_id].xy
        gap = float(np.linalg.norm(b_start_xy - a_end_xy))
        if gap >= _REJECT_OPPOSITE_GAP_M:
            return False
        a_t = ends[a_id].tangent
        b_t = starts[b_id].tangent
        dot = float(np.clip(np.dot(a_t, b_t), -1.0, 1.0))
        angle = math.acos(dot)
        return angle > _OPPOSITE_HEADING_RAD

    # Ground-truth topology takes absolute precedence: replace inferred successors/predecessors.
    if ground_truth is not None:
        successors = {lid: [] for lid in lanelets_partial}
        predecessors = {lid: [] for lid in lanelets_partial}
        for lid, succ_list in ground_truth.items():
            if lid not in lanelets_partial:
                continue
            for s in succ_list:
                if s not in lanelets_partial or s == lid:
                    continue
                if s not in successors[lid]:
                    successors[lid].append(s)
                if lid not in predecessors[s]:
                    predecessors[s].append(lid)
    else:
        # Apply manual overrides: add/remove specific successor edges.
        add_succ = overrides.get("add_successors", {})
        remove_succ = overrides.get("remove_successors", {})
        for lid, to_add in add_succ.items():
            if lid not in lanelets_partial:
                continue
            existing = successors.setdefault(lid, [])
            for s in to_add:
                if s in lanelets_partial and s not in existing:
                    existing.append(s)
                    if lid not in predecessors.setdefault(s, []):
                        predecessors[s].append(lid)
        for lid, to_remove in remove_succ.items():
            if lid not in successors:
                continue
            successors[lid] = [s for s in successors[lid] if s not in to_remove]
            for s in to_remove:
                if s in predecessors and lid in predecessors[s]:
                    predecessors[s].remove(lid)

    lanelets: dict[str, Lanelet] = {}
    for lid, base in lanelets_partial.items():
        if ground_truth is not None:
            # User-provided ground truth: trust it as-is, skip the opposite-endpoint filter.
            succ = list(successors[lid])
            pred = list(predecessors[lid])
        else:
            manual_added = set(overrides.get("add_successors", {}).get(lid, []))
            succ = [
                s for s in successors[lid]
                if s in manual_added or not is_opposite_endpoint(lid, s)
            ]
            pred = []
            for p in predecessors[lid]:
                p_added = set(overrides.get("add_successors", {}).get(p, []))
                if lid in p_added or not is_opposite_endpoint(p, lid):
                    pred.append(p)
        lanelets[lid] = Lanelet(
            lanelet_id=lid,
            centerline=base["centerline"],
            length_m=base["length_m"],
            speed_limit=base["speed_limit"],
            is_highway=base["is_highway"],
            turn_direction=base["turn_direction"],
            attribute_kind=base["attribute_kind"],
            one_way=base["one_way"],
            successor_ids=tuple(succ),
            predecessor_ids=tuple(pred),
        )
    return lanelets


def _endpoint_at_start(centerline: np.ndarray) -> _Endpoint:
    p = centerline[0]
    n = min(centerline.shape[0] - 1, 5)
    t = centerline[n] - centerline[0]
    norm = float(np.linalg.norm(t))
    t = (t / norm) if norm > 1e-9 else np.array([1.0, 0.0])
    return _Endpoint(xy=np.asarray(p, dtype=float), tangent=t.astype(float))


def _endpoint_at_end(centerline: np.ndarray) -> _Endpoint:
    p = centerline[-1]
    n = min(centerline.shape[0] - 1, 5)
    t = centerline[-1] - centerline[-1 - n]
    norm = float(np.linalg.norm(t))
    t = (t / norm) if norm > 1e-9 else np.array([1.0, 0.0])
    return _Endpoint(xy=np.asarray(p, dtype=float), tangent=t.astype(float))


def _candidates_after(
    a_end: _Endpoint, starts: dict[str, _Endpoint], exclude_id: str
) -> list[str]:
    tight: list[tuple[float, str]] = []
    lane: list[tuple[float, str]] = []
    intersection: list[tuple[float, str]] = []
    for b_id, b_start in starts.items():
        if b_id == exclude_id:
            continue
        gap = float(np.linalg.norm(b_start.xy - a_end.xy))
        if gap > _GAP_INTERSECTION_M:
            continue
        if not _heading_within(a_end.tangent, b_start.tangent, _HEADING_MAX_RAD):
            continue
        disp = b_start.xy - a_end.xy
        disp_norm = float(np.linalg.norm(disp))
        forward = 1.0
        if disp_norm > 1e-6:
            forward = float(np.dot(disp / disp_norm, a_end.tangent))
        if gap <= _GAP_TIGHT_M:
            if forward >= _FORWARD_DOT_MIN_TIGHT:
                tight.append((gap, b_id))
        elif gap <= _GAP_LANE_M:
            if forward >= _FORWARD_DOT_MIN_TIGHT:
                lane.append((gap, b_id))
        else:
            if forward >= _FORWARD_DOT_MIN_INTERSECTION:
                intersection.append((gap, b_id))

    if tight:
        return _dedup_by_gap(tight + lane)
    if lane:
        return _dedup_by_gap(lane)
    return _dedup_by_gap(intersection)


def _is_connection(a_end: _Endpoint, b_start: _Endpoint) -> bool:
    gap = float(np.linalg.norm(b_start.xy - a_end.xy))
    if gap > _GAP_INTERSECTION_M:
        return False
    if not _heading_within(a_end.tangent, b_start.tangent, _HEADING_MAX_RAD):
        return False
    disp = b_start.xy - a_end.xy
    disp_norm = float(np.linalg.norm(disp))
    forward = 1.0
    if disp_norm > 1e-6:
        forward = float(np.dot(disp / disp_norm, a_end.tangent))
    if gap <= _GAP_LANE_M:
        return forward >= _FORWARD_DOT_MIN_TIGHT
    return forward >= _FORWARD_DOT_MIN_INTERSECTION


def _heading_within(t_a: np.ndarray, t_b: np.ndarray, max_rad: float) -> bool:
    dot = float(np.clip(np.dot(t_a, t_b), -1.0, 1.0))
    return math.acos(dot) <= max_rad


def _dedup_by_gap(items: list[tuple[float, str]]) -> list[str]:
    items = sorted(items, key=lambda x: x[0])
    seen: set[str] = set()
    out: list[str] = []
    for _gap, lid in items:
        if lid in seen:
            continue
        seen.add(lid)
        out.append(lid)
    return out


def _way_to_polyline(way: OsmWay, nodes: dict[str, OsmNode]) -> np.ndarray:
    pts = []
    for nid in way.node_ids:
        n = nodes.get(nid)
        if n is not None:
            pts.append((n.x, n.y))
    if not pts:
        return np.empty((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)


def _align_direction(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if right.shape[0] < 2:
        return right
    forward_score = float(np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1]))
    reverse_score = float(np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(left[-1] - right[0]))
    if reverse_score < forward_score:
        return right[::-1]
    return right


def _centerline_from_bounds(
    left: np.ndarray, right: np.ndarray, step_m: float
) -> np.ndarray:
    n = max(2, left.shape[0], right.shape[0])
    left_rs = resample_polyline(left, n=n)
    right_rs = resample_polyline(right, n=n)
    center = 0.5 * (left_rs + right_rs)
    return densify_polyline(center, step_m=max(0.05, float(step_m)))


def _attribute_from_tags(tags: dict[str, str]) -> str:
    subtype = tags.get("subtype", "").lower()
    location = tags.get("location", "").lower()
    turn = tags.get("turn_direction", "").lower()
    if subtype == "crosswalk":
        return ATTR_CROSSWALK
    if subtype in {"stop_line", "stopline"}:
        return ATTR_STOPLINE
    if subtype == "roundabout" or location == "roundabout":
        return ATTR_ROUNDABOUT
    if location == "intersection" or turn in {"left", "right", "straight"}:
        return ATTR_INTERSECTION
    return ATTR_NORMAL
