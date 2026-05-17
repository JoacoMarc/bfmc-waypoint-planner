"""Tests for src/osm_parser.py."""


def test_counts(osm_doc):
    assert len(osm_doc.lanelets) == 175
    assert len(osm_doc.multipolygons) == 37
    assert osm_doc.start_area_way_id == "1821"
    assert len(osm_doc.nodes) > 0
    assert len(osm_doc.ways) > 0


def test_flip_y_makes_start_area_top(osm_doc):
    """After Y-flip, the start area corners should be in the positive-Y region."""
    start_corners = ["1817", "1818", "1819", "1820"]
    ys = [osm_doc.nodes[nid].y for nid in start_corners if nid in osm_doc.nodes]
    assert all(y > -0.1 for y in ys), f"Start area Y should be >0 after flip, got {ys}"


def test_lanelets_have_left_right(osm_doc):
    for lid, l in osm_doc.lanelets.items():
        assert l.left_way_id, f"Lanelet {lid} missing left"
        assert l.right_way_id, f"Lanelet {lid} missing right"


def test_multipolygons_have_outer(osm_doc):
    for mid, mp in osm_doc.multipolygons.items():
        assert mp.outer_way_id, f"Multipolygon {mid} missing outer"


def test_speed_limit_tag_present(osm_doc):
    """All lanelet relations should have a speed_limit tag (0.2 or 10)."""
    seen = set()
    for l in osm_doc.lanelets.values():
        sl = l.tags.get("speed_limit", "")
        seen.add(sl)
    assert "0.2" in seen
    assert "10" in seen
