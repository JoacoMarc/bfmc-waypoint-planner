"""Anchor each waypoint multipolygon onto the lanelets that traverse it.

A waypoint anchor is a pair (waypoint_id, lanelet_id) plus the (entry_arc, exit_arc)
where the lanelet's centerline crosses the waypoint polygon. Multiple anchors per
waypoint are possible (parallel lanes, or a polygon covered by adjacent lanelets);
the OP solver deduplicates by wp_id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    centerline_crosses_polygon,
    point_in_polygon,
    polygon_bbox,
    polygon_centroid,
)
from .osm_parser import OsmDocument, way_polyline
from .topology import Lanelet


@dataclass(frozen=True)
class WaypointArea:
    wp_id: str
    polygon_xy: np.ndarray  # (N, 2)
    centroid: tuple[float, float]
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class WaypointAnchor:
    wp_id: str
    lanelet_id: str
    entry_arc_m: float
    exit_arc_m: float
    crossing_length_m: float
    entry_xy: tuple[float, float]
    exit_xy: tuple[float, float]


def extract_waypoint_areas(doc: OsmDocument) -> list[WaypointArea]:
    """Build WaypointArea list from multipolygon relations."""
    areas: list[WaypointArea] = []
    for mp in doc.multipolygons.values():
        way = doc.ways.get(mp.outer_way_id)
        if way is None:
            continue
        pts = way_polyline(way, doc.nodes)
        if len(pts) < 3:
            continue
        polygon = np.asarray(pts, dtype=float)
        centroid = polygon_centroid(polygon)
        bbox = polygon_bbox(polygon)
        areas.append(
            WaypointArea(
                wp_id=mp.relation_id, polygon_xy=polygon, centroid=centroid, bbox=bbox
            )
        )
    return areas


def extract_start_area(doc: OsmDocument) -> WaypointArea | None:
    """Build a WaypointArea for the random start polygon (way with area=yes)."""
    if doc.start_area_way_id is None:
        return None
    way = doc.ways.get(doc.start_area_way_id)
    if way is None:
        return None
    pts = way_polyline(way, doc.nodes)
    if len(pts) < 3:
        return None
    polygon = np.asarray(pts, dtype=float)
    return WaypointArea(
        wp_id=f"start:{doc.start_area_way_id}",
        polygon_xy=polygon,
        centroid=polygon_centroid(polygon),
        bbox=polygon_bbox(polygon),
    )


def compute_waypoint_anchors(
    lanelets: dict[str, Lanelet], waypoints: list[WaypointArea]
) -> list[WaypointAnchor]:
    """For each waypoint, find all lanelets whose centerline traverses it."""
    anchors: list[WaypointAnchor] = []
    for wp in waypoints:
        xmin, ymin, xmax, ymax = wp.bbox
        for lid, lanelet in lanelets.items():
            cl = lanelet.centerline
            # Quick reject by bbox
            if (
                cl[:, 0].max() < xmin
                or cl[:, 0].min() > xmax
                or cl[:, 1].max() < ymin
                or cl[:, 1].min() > ymax
            ):
                continue
            result = centerline_crosses_polygon(cl, wp.polygon_xy)
            if result is None:
                continue
            entry_arc, exit_arc, entry_xy, exit_xy = result
            anchors.append(
                WaypointAnchor(
                    wp_id=wp.wp_id,
                    lanelet_id=lid,
                    entry_arc_m=entry_arc,
                    exit_arc_m=exit_arc,
                    crossing_length_m=max(0.0, exit_arc - entry_arc),
                    entry_xy=entry_xy,
                    exit_xy=exit_xy,
                )
            )
    return anchors


def find_start_exit_lanelets(
    lanelets: dict[str, Lanelet], start_area: WaypointArea
) -> list[tuple[str, tuple[float, float], float]]:
    """Find lanelets that EXIT the start area: their centerline starts inside and ends outside.

    These are the valid "first lanelets" the car can follow to leave the start area.
    Returns list of (lanelet_id, exit_xy_on_boundary, exit_arc_m).
    """
    out: list[tuple[str, tuple[float, float], float]] = []
    xmin, ymin, xmax, ymax = start_area.bbox
    for lid, lanelet in lanelets.items():
        cl = lanelet.centerline
        if (
            cl[:, 0].max() < xmin
            or cl[:, 0].min() > xmax
            or cl[:, 1].max() < ymin
            or cl[:, 1].min() > ymax
        ):
            continue
        first_inside = point_in_polygon(
            (float(cl[0, 0]), float(cl[0, 1])), start_area.polygon_xy
        )
        last_inside = point_in_polygon(
            (float(cl[-1, 0]), float(cl[-1, 1])), start_area.polygon_xy
        )
        # Lanelet must START inside and END outside the polygon.
        if not (first_inside and not last_inside):
            continue
        result = centerline_crosses_polygon(cl, start_area.polygon_xy)
        if result is None:
            continue
        _entry_arc, exit_arc, _entry_xy, exit_xy = result
        out.append((lid, exit_xy, exit_arc))
    return out


def pick_start_pose(
    lanelets: dict[str, Lanelet], start_area: WaypointArea
) -> tuple[float, float, float, str]:
    """Pick a start pose (x, y, yaw_rad) inside the start area on an outgoing lanelet.

    Strategy: among lanelets that exit the start area, pick the one whose centerline
    is densest inside the polygon (most samples). The start pose is the centerline
    sample closest to the polygon centroid; the yaw is the centerline tangent at
    that sample. Returns (x, y, yaw, lanelet_id).
    """
    candidates = find_start_exit_lanelets(lanelets, start_area)
    if not candidates:
        # Fallback: centroid + yaw 0
        return (start_area.centroid[0], start_area.centroid[1], 0.0, "")
    # Pick the lanelet whose entry point is closest to the centroid
    cx, cy = start_area.centroid
    best_lid = ""
    best_dist = float("inf")
    best_xy = (cx, cy)
    best_idx = 0
    best_cl: np.ndarray | None = None
    for lid in [c[0] for c in candidates]:
        lanelet = lanelets[lid]
        cl = lanelet.centerline
        # Find centerline sample closest to centroid
        d = np.linalg.norm(cl - np.asarray([cx, cy], dtype=float), axis=1)
        i = int(np.argmin(d))
        if float(d[i]) < best_dist:
            best_dist = float(d[i])
            best_lid = lid
            best_xy = (float(cl[i, 0]), float(cl[i, 1]))
            best_idx = i
            best_cl = cl
    if best_cl is None or best_cl.shape[0] < 2:
        return (best_xy[0], best_xy[1], 0.0, best_lid)
    j = min(best_cl.shape[0] - 1, best_idx + 1)
    if j == best_idx:
        j = max(0, best_idx - 1)
        dx = best_cl[best_idx, 0] - best_cl[j, 0]
        dy = best_cl[best_idx, 1] - best_cl[j, 1]
    else:
        dx = best_cl[j, 0] - best_cl[best_idx, 0]
        dy = best_cl[j, 1] - best_cl[best_idx, 1]
    import math as _m
    yaw = _m.atan2(dy, dx)
    return (best_xy[0], best_xy[1], yaw, best_lid)
