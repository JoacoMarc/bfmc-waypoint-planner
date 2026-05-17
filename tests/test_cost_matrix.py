"""Tests for src/cost_matrix.py."""

import numpy as np


def test_cost_matrix_shape_and_diagonal(lanelets, waypoint_areas, start_area, osm_path, tmp_path):
    from src.cost_matrix import CostParams, build_cost_matrix
    from src.waypoint_anchor import compute_waypoint_anchors, find_start_exit_lanelets

    anchors = compute_waypoint_anchors(lanelets, waypoint_areas)
    sel = find_start_exit_lanelets(lanelets, start_area)
    params = CostParams()
    cm = build_cost_matrix(lanelets, anchors, sel, params, osm_path, cache_dir=str(tmp_path))
    N = len(anchors)
    assert cm.D.shape == (N + 1, N + 1)
    # Diagonal must be 0
    for i in range(N + 1):
        assert cm.D[i, i] == 0.0


def test_cost_matrix_asymmetric(lanelets, waypoint_areas, start_area, osm_path, tmp_path):
    from src.cost_matrix import CostParams, build_cost_matrix
    from src.waypoint_anchor import compute_waypoint_anchors, find_start_exit_lanelets

    anchors = compute_waypoint_anchors(lanelets, waypoint_areas)
    sel = find_start_exit_lanelets(lanelets, start_area)
    cm = build_cost_matrix(
        lanelets, anchors, sel, CostParams(), osm_path, cache_dir=str(tmp_path)
    )
    # Find at least one pair where D[i, j] != D[j, i] (one-way constraint)
    N = cm.D.shape[0]
    asymmetric_found = False
    for i in range(1, N):
        for j in range(1, N):
            if i == j:
                continue
            a = cm.D[i, j]
            b = cm.D[j, i]
            if np.isfinite(a) and np.isfinite(b) and abs(a - b) > 0.1:
                asymmetric_found = True
                break
        if asymmetric_found:
            break
    assert asymmetric_found, "Cost matrix should be asymmetric due to one-way lanes"


def test_cost_matrix_cache(lanelets, waypoint_areas, start_area, osm_path, tmp_path):
    from src.cost_matrix import CostParams, build_cost_matrix
    from src.waypoint_anchor import compute_waypoint_anchors, find_start_exit_lanelets

    anchors = compute_waypoint_anchors(lanelets, waypoint_areas)
    sel = find_start_exit_lanelets(lanelets, start_area)
    params = CostParams()
    cm1 = build_cost_matrix(lanelets, anchors, sel, params, osm_path, cache_dir=str(tmp_path))
    cm2 = build_cost_matrix(lanelets, anchors, sel, params, osm_path, cache_dir=str(tmp_path))
    np.testing.assert_array_equal(cm1.D, cm2.D)
