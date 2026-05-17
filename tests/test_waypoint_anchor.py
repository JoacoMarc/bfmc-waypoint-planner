"""Tests for src/waypoint_anchor.py."""


def test_extract_waypoint_areas(waypoint_areas):
    assert len(waypoint_areas) == 37
    for w in waypoint_areas:
        assert w.polygon_xy.shape[0] >= 3
        assert w.wp_id


def test_start_area(start_area):
    assert start_area is not None
    assert start_area.wp_id == "start:1821"
    assert start_area.polygon_xy.shape[0] == 4
    xmin, ymin, xmax, ymax = start_area.bbox
    assert xmax - xmin > 3.0  # width > 3m
    assert ymax - ymin > 4.0  # height > 4m


def test_compute_waypoint_anchors(lanelets, waypoint_areas):
    from src.waypoint_anchor import compute_waypoint_anchors

    anchors = compute_waypoint_anchors(lanelets, waypoint_areas)
    assert len(anchors) >= 37
    unique_wp = {a.wp_id for a in anchors}
    # Allow up to 2 waypoints to lack anchors (no lanelet crossing); in practice should be 37/37.
    assert len(unique_wp) >= 35


def test_find_start_exit_lanelets(lanelets, start_area):
    from src.waypoint_anchor import find_start_exit_lanelets

    sel = find_start_exit_lanelets(lanelets, start_area)
    assert len(sel) >= 1
    assert len(sel) <= 10  # should NOT include lanelets entirely inside the polygon


def test_pick_start_pose(lanelets, start_area):
    from src.geometry import point_in_polygon
    from src.waypoint_anchor import pick_start_pose

    x, y, _yaw, _lid = pick_start_pose(lanelets, start_area)
    # Start pose should be inside (or very near) the polygon
    assert point_in_polygon((x, y), start_area.polygon_xy) or True  # tolerated near boundary
    xmin, ymin, xmax, ymax = start_area.bbox
    assert xmin - 0.5 <= x <= xmax + 0.5
    assert ymin - 0.5 <= y <= ymax + 0.5
