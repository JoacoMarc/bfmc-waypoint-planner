"""Shared pytest fixtures."""

import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root() -> str:
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def osm_path(project_root) -> str:
    return os.path.join(project_root, "data", "lanelet2_map_FINAL_RandomStartingArea.osm")


@pytest.fixture(scope="session")
def osm_doc(osm_path):
    from src.osm_parser import parse_osm

    return parse_osm(osm_path)


@pytest.fixture(scope="session")
def lanelets(osm_doc, project_root):
    from src.topology import build_lanelets, load_topology_ground_truth, load_topology_overrides

    gt = load_topology_ground_truth(
        os.path.join(project_root, "data", "topology_current.json")
    )
    overrides = load_topology_overrides(
        os.path.join(project_root, "data", "topology_overrides.json")
    )
    return build_lanelets(osm_doc, step_m=0.05, overrides=overrides, ground_truth=gt)


@pytest.fixture(scope="session")
def waypoint_areas(osm_doc):
    from src.waypoint_anchor import extract_waypoint_areas

    return extract_waypoint_areas(osm_doc)


@pytest.fixture(scope="session")
def start_area(osm_doc):
    from src.waypoint_anchor import extract_start_area

    return extract_start_area(osm_doc)
