"""Tests for src/geometry.py."""

import numpy as np

from src.geometry import (
    centerline_crosses_polygon,
    point_in_polygon,
    polygon_centroid,
    project_arclength,
    resample_polyline,
)


def test_point_in_polygon_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert point_in_polygon((0.5, 0.5), sq)
    assert not point_in_polygon((1.5, 0.5), sq)
    assert not point_in_polygon((-0.5, 0.5), sq)
    assert not point_in_polygon((0.5, -0.5), sq)


def test_point_in_polygon_triangle():
    tri = np.array([[0, 0], [2, 0], [1, 2]], dtype=float)
    assert point_in_polygon((1.0, 0.5), tri)
    assert not point_in_polygon((1.0, 2.5), tri)


def test_centroid_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cx, cy = polygon_centroid(sq)
    assert abs(cx - 0.5) < 1e-6
    assert abs(cy - 0.5) < 1e-6


def test_project_arclength_simple():
    line = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    arc, lat = project_arclength((1.5, 0.5), line)
    assert abs(arc - 1.5) < 1e-6
    assert abs(lat - 0.5) < 1e-6


def test_centerline_crosses_polygon():
    line = np.array([[i * 0.1, 0.5] for i in range(21)], dtype=float)  # 0..2 in x, y=0.5
    sq = np.array([[0.7, 0], [1.3, 0], [1.3, 1], [0.7, 1]], dtype=float)
    result = centerline_crosses_polygon(line, sq)
    assert result is not None
    entry_arc, exit_arc, entry_xy, exit_xy = result
    # Entry should be near x=0.7, exit near x=1.3
    assert 0.6 < entry_xy[0] < 0.8
    assert 1.2 < exit_xy[0] < 1.4
    assert entry_arc < exit_arc


def test_centerline_does_not_cross():
    line = np.array([[i * 0.1, 2.0] for i in range(21)], dtype=float)
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert centerline_crosses_polygon(line, sq) is None


def test_resample_polyline_n_points():
    line = np.array([[0, 0], [3, 0]], dtype=float)
    out = resample_polyline(line, n=4)
    assert out.shape == (4, 2)
    assert abs(out[0, 0] - 0.0) < 1e-9
    assert abs(out[-1, 0] - 3.0) < 1e-9
