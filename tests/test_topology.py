"""Tests for src/topology.py."""


def test_all_lanelets_built(lanelets):
    assert len(lanelets) == 175


def test_centerlines_nonempty(lanelets):
    for lid, l in lanelets.items():
        assert l.centerline.shape[0] >= 2, f"Lanelet {lid} has empty centerline"
        assert l.length_m > 0.0


def test_topology_connectivity(lanelets):
    """At least 95% of lanelets should have either successors or predecessors (no isolated nodes)."""
    isolated = [
        lid for lid, l in lanelets.items() if not l.successor_ids and not l.predecessor_ids
    ]
    assert len(isolated) <= 8, f"Too many isolated lanelets: {len(isolated)} ({isolated[:10]})"


def test_no_coincident_endpoint_opposite_pairs(lanelets):
    """No A -> B where A.end and B.start coincide (gap < 0.1 m) AND headings differ > 90°.

    This is the false-positive pattern at intersection junction nodes (e.g. 888 -> 865)
    where two turn lanelets share an endpoint but flow in opposite directions.
    Larger gaps with opposing headings are legitimate "junction crossing" successors.
    """
    import math
    import numpy as np

    bad = []
    for lid, l in lanelets.items():
        if l.centerline.shape[0] < 2:
            continue
        a_end = l.centerline[-1]
        a_tan = l.centerline[-1] - l.centerline[-2]
        a_tan /= max(np.linalg.norm(a_tan), 1e-9)
        for sid in l.successor_ids:
            sl = lanelets.get(sid)
            if sl is None or sl.centerline.shape[0] < 2:
                continue
            b_start = sl.centerline[0]
            gap = float(np.linalg.norm(a_end - b_start))
            if gap >= 0.10:
                continue
            b_tan = sl.centerline[1] - sl.centerline[0]
            b_tan /= max(np.linalg.norm(b_tan), 1e-9)
            dot = float(np.clip(np.dot(a_tan, b_tan), -1.0, 1.0))
            if math.acos(dot) > math.radians(90.0):
                bad.append((lid, sid))
    assert len(bad) == 0, f"Coincident-endpoint opposite-heading pairs (false positives): {bad}"


def test_user_confirmed_connections(lanelets):
    """Verify successor relations confirmed by the user."""
    # 946 -> 888, 891, 903
    assert "888" in lanelets["946"].successor_ids
    assert "891" in lanelets["946"].successor_ids
    assert "903" in lanelets["946"].successor_ids
    # 891 -> 828
    assert "828" in lanelets["891"].successor_ids
    # 903 -> 908
    assert "908" in lanelets["903"].successor_ids
    # 888 -> 868 (connector lanelet; 872/877 are reached via 868 per user)
    assert "868" in lanelets["888"].successor_ids
    # 888 must NOT go directly to 872 (must go via 868)
    assert "872" not in lanelets["888"].successor_ids
    # 868 -> 872, 877
    assert "872" in lanelets["868"].successor_ids
    assert "877" in lanelets["868"].successor_ids
    # 457 -> 460 (junction-crossing across opposite-direction lanes)
    assert "460" in lanelets["457"].successor_ids
    # 762 -> 765
    assert "765" in lanelets["762"].successor_ids
    # 872 -> 665 (junction-crossing)
    assert "665" in lanelets["872"].successor_ids
    # 888 must NOT connect to 865 (coincident endpoint, opposite direction)
    assert "865" not in lanelets["888"].successor_ids


def test_highway_count(lanelets):
    """The OSM has 16 highway lanelets (speed_limit=10)."""
    hw_count = sum(1 for l in lanelets.values() if l.is_highway)
    assert hw_count == 16


def test_no_self_loops(lanelets):
    for lid, l in lanelets.items():
        assert lid not in l.successor_ids
        assert lid not in l.predecessor_ids
