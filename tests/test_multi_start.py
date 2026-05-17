"""Tests for src/multi_start.py."""

import random
import numpy as np

from src.geometry import point_in_polygon
from src.multi_start import sample_poses_in_polygon


def test_sample_poses_inside_polygon():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    rng = random.Random(0)
    poses = sample_poses_in_polygon(sq, n=20, rng=rng)
    assert len(poses) == 20
    for p in poses:
        assert point_in_polygon(p, sq)


def test_sample_poses_concave():
    # L-shaped polygon
    L = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    rng = random.Random(42)
    poses = sample_poses_in_polygon(L, n=10, rng=rng)
    for p in poses:
        assert point_in_polygon(p, L)
