"""Tests for src/orienteering_solver.py."""

import numpy as np

from src.orienteering_solver import solve_orienteering


def test_synthetic_5_nodes():
    """A 5-node graph where the optimal path is known. Node 0 is start (score 0).
    Nodes 1..4 each have score 3. Travel time between any two nodes is 100s except
    along a short loop: 0->1->2->3->4 each cost 100s. Budget 400 lets us visit all 4."""
    D = np.full((5, 5), 1000.0)
    np.fill_diagonal(D, 0.0)
    # Make a clear path 0->1->2->3->4 with low costs
    for i in range(4):
        D[i, i + 1] = 100.0
        D[i + 1, i] = 100.0
    scores = np.array([0.0, 3.0, 3.0, 3.0, 3.0])
    wp_id_per_node = ["", "a", "b", "c", "d"]
    sol = solve_orienteering(D, scores, wp_id_per_node, budget_s=400.0, restarts=10, sa_iterations=2000, seed=1)
    assert sol.score == 12.0  # all 4 waypoints
    assert len(sol.sequence) == 5
    assert sol.cost_s <= 400.0


def test_tight_budget():
    """With budget 250s, only 2 of the 4 waypoints should be visited (each costs 100s)."""
    D = np.full((5, 5), 1000.0)
    np.fill_diagonal(D, 0.0)
    for i in range(4):
        D[i, i + 1] = 100.0
        D[i + 1, i] = 100.0
    scores = np.array([0.0, 3.0, 3.0, 3.0, 3.0])
    wp_id_per_node = ["", "a", "b", "c", "d"]
    sol = solve_orienteering(D, scores, wp_id_per_node, budget_s=250.0, restarts=10, sa_iterations=2000, seed=1)
    # 2 waypoints = 6 points expected
    assert sol.score == 6.0
    assert sol.cost_s <= 250.0


def test_deduplication_by_wp_id():
    """If two nodes share the same wp_id, only one should be visited."""
    D = np.array([
        [0.0,   10.0, 10.0, 100.0],
        [10.0,  0.0,  5.0,  20.0],
        [10.0,  5.0,  0.0,  20.0],
        [100.0, 20.0, 20.0, 0.0],
    ])
    scores = np.array([0.0, 3.0, 3.0, 3.0])
    wp_id_per_node = ["", "shared", "shared", "other"]
    sol = solve_orienteering(D, scores, wp_id_per_node, budget_s=80.0, restarts=10, sa_iterations=1000, seed=1)
    # The solver should pick either node 1 or 2 (not both) + node 3
    visited_wps = set(sol.visited_wp_ids)
    assert "other" in visited_wps
    assert "shared" in visited_wps
    # The score should not exceed 6 (3 + 3), since "shared" counted once
    assert sol.score == 6.0
