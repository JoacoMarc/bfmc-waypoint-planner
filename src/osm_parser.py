"""Standalone OSM Lanelet2 parser for BFMC waypoint planner.

Replicates the relevant subset of urt-brain-bosch/src/routing/lanelet/from_osm.py
without depending on it. Reads:
  - <node> with tag local_x / local_y (no lat/lon needed)
  - <way> with refs to nodes and tags
  - <relation type=lanelet> with members role=left|right (way refs)
  - <relation type=multipolygon> with member role=outer (way ref) -> waypoint area
  - <way ... > with tag area=yes -> random start area

Applies the same Y-flip as the brain so coordinates align with the project frame.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OsmNode:
    node_id: str
    x: float
    y: float


@dataclass(frozen=True)
class OsmWay:
    way_id: str
    node_ids: tuple[str, ...]
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OsmLanelet:
    relation_id: str
    left_way_id: str
    right_way_id: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OsmMultipolygon:
    relation_id: str
    outer_way_id: str


@dataclass(frozen=True)
class OsmDocument:
    nodes: dict[str, OsmNode]
    ways: dict[str, OsmWay]
    lanelets: dict[str, OsmLanelet]
    multipolygons: dict[str, OsmMultipolygon]
    start_area_way_id: str | None


def parse_osm(path: str) -> OsmDocument:
    root = ET.parse(path).getroot()

    raw_nodes: dict[str, tuple[float, float]] = {}
    for node_el in root.findall("node"):
        node_id = (node_el.get("id") or "").strip()
        if not node_id:
            continue
        tags = _load_tags(node_el)
        local_x = _first_float(tags, "local_x", "urt:local_x")
        local_y = _first_float(tags, "local_y", "urt:local_y")
        if local_x is None or local_y is None:
            lat = _safe_float(node_el.get("lat"))
            lon = _safe_float(node_el.get("lon"))
            raw_nodes[node_id] = (float(lon), float(lat))
        else:
            raw_nodes[node_id] = (float(local_x), float(local_y))

    nodes = _flip_nodes_y(raw_nodes)

    ways: dict[str, OsmWay] = {}
    start_area_way_id: str | None = None
    for way_el in root.findall("way"):
        way_id = (way_el.get("id") or "").strip()
        if not way_id:
            continue
        node_ids = tuple(
            (nd.get("ref") or "").strip()
            for nd in way_el.findall("nd")
            if (nd.get("ref") or "").strip()
        )
        tags = _load_tags(way_el)
        ways[way_id] = OsmWay(way_id=way_id, node_ids=node_ids, tags=tags)
        if tags.get("area", "").lower() == "yes":
            start_area_way_id = way_id

    lanelets: dict[str, OsmLanelet] = {}
    multipolygons: dict[str, OsmMultipolygon] = {}
    for rel_el in root.findall("relation"):
        rel_id = (rel_el.get("id") or "").strip()
        if not rel_id:
            continue
        tags = _load_tags(rel_el)
        rel_type = tags.get("type", "").lower()
        members = [
            (
                (m.get("type") or "").strip(),
                (m.get("ref") or "").strip(),
                (m.get("role") or "").strip(),
            )
            for m in rel_el.findall("member")
        ]
        if rel_type == "lanelet":
            left = _find_member(members, member_type="way", role="left")
            right = _find_member(members, member_type="way", role="right")
            if left and right:
                lanelets[rel_id] = OsmLanelet(
                    relation_id=rel_id, left_way_id=left, right_way_id=right, tags=tags
                )
        elif rel_type == "multipolygon":
            outer = _find_member(members, member_type="way", role="outer")
            if outer:
                multipolygons[rel_id] = OsmMultipolygon(
                    relation_id=rel_id, outer_way_id=outer
                )

    return OsmDocument(
        nodes=nodes,
        ways=ways,
        lanelets=lanelets,
        multipolygons=multipolygons,
        start_area_way_id=start_area_way_id,
    )


def _flip_nodes_y(raw_nodes: dict[str, tuple[float, float]]) -> dict[str, OsmNode]:
    """Mirror the imported OSM vertically into the project frame.

    Identical formula to urt-brain-bosch/src/routing/lanelet/from_osm.py::_flip_nodes_y:
        y_mid_sum = y_min + y_max
        y_flipped = y_mid_sum - y_raw
    """
    if not raw_nodes:
        return {}
    ys = [y for _, y in raw_nodes.values()]
    y_min = min(ys)
    y_max = max(ys)
    y_mid_sum = float(y_min + y_max)
    return {
        node_id: OsmNode(node_id=node_id, x=float(x), y=float(y_mid_sum - y))
        for node_id, (x, y) in raw_nodes.items()
    }


def _load_tags(element: ET.Element) -> dict[str, str]:
    return {
        (tag.get("k") or ""): (tag.get("v") or "")
        for tag in element.findall("tag")
        if tag.get("k") is not None
    }


def _safe_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _first_float(tags: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key not in tags:
            continue
        try:
            return float(tags[key])
        except (TypeError, ValueError):
            continue
    return None


def _find_member(
    members: list[tuple[str, str, str]], *, member_type: str, role: str
) -> str | None:
    for m_type, ref, m_role in members:
        if m_type == member_type and m_role == role and ref:
            return ref
    return None


def way_polyline(way: OsmWay, nodes: dict[str, OsmNode]) -> list[tuple[float, float]]:
    """Return list of (x, y) for the nodes in a way, in order."""
    out: list[tuple[float, float]] = []
    for nid in way.node_ids:
        n = nodes.get(nid)
        if n is not None:
            out.append((n.x, n.y))
    return out
