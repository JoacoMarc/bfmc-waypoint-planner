"""Geometric utilities for the BFMC waypoint planner.

Pure functions over numpy arrays. No external dependencies beyond numpy.
"""

from __future__ import annotations

import math

import numpy as np


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test. polygon shape (N, 2), counterclockwise or clockwise."""
    px, py = float(point[0]), float(point[1])
    n = polygon.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-18) + xi):
            inside = not inside
        j = i
    return inside


def polygon_centroid(polygon: np.ndarray) -> tuple[float, float]:
    """Centroid via shoelace. polygon shape (N, 2). For degenerate polygons returns mean of points."""
    if polygon.shape[0] < 3:
        return (float(polygon[:, 0].mean()), float(polygon[:, 1].mean()))
    x = polygon[:, 0]
    y = polygon[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = 0.5 * float(np.sum(cross))
    if abs(area) < 1e-12:
        return (float(x.mean()), float(y.mean()))
    cx = float(np.sum((x + x_next) * cross) / (6.0 * area))
    cy = float(np.sum((y + y_next) * cross) / (6.0 * area))
    return (cx, cy)


def polygon_bbox(polygon: np.ndarray) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax)."""
    return (
        float(polygon[:, 0].min()),
        float(polygon[:, 1].min()),
        float(polygon[:, 0].max()),
        float(polygon[:, 1].max()),
    )


def polyline_length(polyline: np.ndarray) -> float:
    """Sum of segment lengths."""
    if polyline.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))


def polyline_arclengths(polyline: np.ndarray) -> np.ndarray:
    """Cumulative arclength at each vertex. Shape (N,)."""
    if polyline.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    if polyline.shape[0] == 1:
        return np.zeros((1,), dtype=float)
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def project_arclength(
    point: tuple[float, float], centerline: np.ndarray
) -> tuple[float, float]:
    """Project a point onto a polyline and return (arclength_m, lateral_offset_m).

    Iterates segments; finds the closest projection.
    """
    if centerline.shape[0] < 2:
        if centerline.shape[0] == 1:
            return (
                0.0,
                float(math.hypot(point[0] - centerline[0, 0], point[1] - centerline[0, 1])),
            )
        return (0.0, 0.0)
    p = np.asarray(point, dtype=float)
    arclen = polyline_arclengths(centerline)
    best_arc = 0.0
    best_lat = float("inf")
    for i in range(centerline.shape[0] - 1):
        a = centerline[i]
        b = centerline[i + 1]
        ab = b - a
        ab_norm_sq = float(np.dot(ab, ab))
        if ab_norm_sq < 1e-18:
            t = 0.0
        else:
            t = float(np.dot(p - a, ab) / ab_norm_sq)
            t = max(0.0, min(1.0, t))
        proj = a + t * ab
        lat = float(np.linalg.norm(p - proj))
        if lat < best_lat:
            best_lat = lat
            best_arc = float(arclen[i] + t * math.sqrt(ab_norm_sq))
    return (best_arc, best_lat)


def centerline_crosses_polygon(
    centerline: np.ndarray, polygon: np.ndarray
) -> tuple[float, float, tuple[float, float], tuple[float, float]] | None:
    """Detect if a centerline enters and exits a polygon.

    Returns (entry_arc_m, exit_arc_m, entry_xy, exit_xy) or None.
    Strategy: mark each sample as inside/outside, find the first inside index and the last.
    Then refine entry/exit by linear interpolation between the last outside and first inside (and viceversa).
    """
    if centerline.shape[0] < 2:
        return None
    n = centerline.shape[0]
    inside_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        inside_mask[i] = point_in_polygon((centerline[i, 0], centerline[i, 1]), polygon)

    if not inside_mask.any():
        return None

    first_in = int(np.argmax(inside_mask))
    last_in = n - 1 - int(np.argmax(inside_mask[::-1]))

    arclen = polyline_arclengths(centerline)

    # Refine entry: between sample first_in-1 (outside) and first_in (inside)
    if first_in > 0 and not inside_mask[first_in - 1]:
        entry_xy, entry_arc = _interp_boundary(
            centerline[first_in - 1],
            centerline[first_in],
            float(arclen[first_in - 1]),
            float(arclen[first_in]),
            polygon,
            outside_to_inside=True,
        )
    else:
        entry_xy = (float(centerline[first_in, 0]), float(centerline[first_in, 1]))
        entry_arc = float(arclen[first_in])

    # Refine exit: between sample last_in (inside) and last_in+1 (outside)
    if last_in < n - 1 and not inside_mask[last_in + 1]:
        exit_xy, exit_arc = _interp_boundary(
            centerline[last_in],
            centerline[last_in + 1],
            float(arclen[last_in]),
            float(arclen[last_in + 1]),
            polygon,
            outside_to_inside=False,
        )
    else:
        exit_xy = (float(centerline[last_in, 0]), float(centerline[last_in, 1]))
        exit_arc = float(arclen[last_in])

    if exit_arc < entry_arc:
        exit_arc = entry_arc
    return (entry_arc, exit_arc, entry_xy, exit_xy)


def _interp_boundary(
    p_out: np.ndarray,
    p_in: np.ndarray,
    arc_out: float,
    arc_in: float,
    polygon: np.ndarray,
    outside_to_inside: bool,
    n_bisect: int = 12,
) -> tuple[tuple[float, float], float]:
    """Find via bisection the crossing point between p_out (outside) and p_in (inside)."""
    a, b = (p_out.copy(), p_in.copy()) if outside_to_inside else (p_in.copy(), p_out.copy())
    arc_a, arc_b = (arc_out, arc_in) if outside_to_inside else (arc_in, arc_out)
    for _ in range(n_bisect):
        m = 0.5 * (a + b)
        arc_m = 0.5 * (arc_a + arc_b)
        m_inside = point_in_polygon((float(m[0]), float(m[1])), polygon)
        if outside_to_inside:
            if m_inside:
                b = m
                arc_b = arc_m
            else:
                a = m
                arc_a = arc_m
        else:
            if m_inside:
                a = m
                arc_a = arc_m
            else:
                b = m
                arc_b = arc_m
    boundary = 0.5 * (a + b)
    arc = 0.5 * (arc_a + arc_b)
    return (float(boundary[0]), float(boundary[1])), float(arc)


def resample_polyline(polyline: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to exactly n points evenly spaced by arclength."""
    if polyline.shape[0] < 2:
        if polyline.shape[0] == 1:
            return np.tile(polyline[0], (n, 1))
        return np.zeros((n, 2), dtype=float)
    seg_lens = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    out = np.zeros((n, 2), dtype=float)
    for idx, target in enumerate(np.linspace(0.0, total, num=n)):
        seg_idx = int(np.searchsorted(cum, target, side="right") - 1)
        seg_idx = max(0, min(seg_idx, polyline.shape[0] - 2))
        seg_len = float(seg_lens[seg_idx])
        if seg_len <= 1e-12:
            out[idx] = polyline[seg_idx]
            continue
        t = (target - float(cum[seg_idx])) / seg_len
        out[idx] = polyline[seg_idx] + t * (polyline[seg_idx + 1] - polyline[seg_idx])
    return out


def densify_polyline(polyline: np.ndarray, step_m: float) -> np.ndarray:
    """Densify a polyline so consecutive samples are <= step_m apart."""
    if polyline.shape[0] < 2:
        return polyline
    out: list[np.ndarray] = [polyline[0]]
    for i in range(polyline.shape[0] - 1):
        p0 = polyline[i]
        p1 = polyline[i + 1]
        dist = float(np.linalg.norm(p1 - p0))
        if dist <= 1e-12:
            continue
        steps = max(1, int(math.ceil(dist / max(step_m, 1e-3))))
        for s in range(1, steps + 1):
            t = float(s) / float(steps)
            out.append(p0 + t * (p1 - p0))
    return np.asarray(out, dtype=float)
