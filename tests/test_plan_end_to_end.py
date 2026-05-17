"""End-to-end test of the CLI pipeline against the real OSM."""

import json
import os
import subprocess
import sys


def test_cli_runs(project_root, tmp_path):
    osm = os.path.join(project_root, "data", "lanelet2_map_FINAL_RandomStartingArea.osm")
    out_json = str(tmp_path / "plan.json")
    out_svg = str(tmp_path / "plan.svg")
    cmd = [
        sys.executable,
        os.path.join(project_root, "tools", "plan_bfmc_route.py"),
        "--osm", osm,
        "--output-json", out_json,
        "--output-svg", out_svg,
        "--restarts", "5",
        "--sa-iterations", "1000",
        "--cache-dir", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"CLI failed: stdout={result.stdout} stderr={result.stderr}"
    assert os.path.exists(out_json)
    assert os.path.exists(out_svg)
    # Validate JSON structure
    with open(out_json) as fh:
        plan = json.load(fh)
    assert plan["version"] == 1
    assert "waypoint_sequence" in plan
    assert "lanelet_sequence" in plan
    assert plan["expected_waypoints_visited"] > 0
    assert plan["expected_score"] >= 15
    assert plan["expected_total_time_s"] <= 600.0 + 1.0


def test_lanelet_sequence_connectivity(project_root):
    """The lanelet sequence from a planning run must contain valid successor links."""
    from src.osm_parser import parse_osm
    from src.topology import build_lanelets, load_topology_ground_truth, load_topology_overrides

    osm = os.path.join(project_root, "data", "lanelet2_map_FINAL_RandomStartingArea.osm")
    out_json = os.path.join(project_root, "data", "outputs", "bfmc_plan.json")
    if not os.path.exists(out_json):
        # Skip if no recent plan; the CLI test above writes a fresh one in tmp.
        return
    doc = parse_osm(osm)
    gt = load_topology_ground_truth(os.path.join(project_root, "data", "topology_current.json"))
    ov = load_topology_overrides(os.path.join(project_root, "data", "topology_overrides.json"))
    lanelets = build_lanelets(doc, step_m=0.05, overrides=ov, ground_truth=gt)
    with open(out_json) as fh:
        plan = json.load(fh)
    seq = plan["lanelet_sequence"]
    assert len(seq) > 0
    broken = 0
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a == b:
            continue
        if a not in lanelets:
            broken += 1
            continue
        if b not in lanelets[a].successor_ids:
            broken += 1
    assert broken == 0, f"{broken} broken successor links in lanelet_sequence"
