"""Tests for src/traffic_light.py."""

import math
import os
import tempfile

import json

from src.traffic_light import TLCycle, TrafficLightModel, load_traffic_lights


def test_bfmc_cycle_expected_wait():
    c = TLCycle(green_s=5.0, yellow_s=3.0, red_s=3.0)
    # E[wait] = (3/11)*(1.5+3) + (3/11)*1.5 = (3*4.5 + 3*1.5)/11 = 18/11 ≈ 1.636
    expected = 18.0 / 11.0
    assert abs(c.expected_wait_s() - expected) < 1e-6


def test_all_green_zero_wait():
    c = TLCycle(green_s=10.0, yellow_s=0.0, red_s=0.0)
    assert c.expected_wait_s() == 0.0


def test_all_red_half_red():
    c = TLCycle(green_s=0.0, yellow_s=0.0, red_s=4.0)
    assert abs(c.expected_wait_s() - 2.0) < 1e-9


def test_load_traffic_lights_file(tmp_path):
    data = {
        "default_cycle": {"green_s": 4.0, "yellow_s": 2.0, "red_s": 2.0},
        "zones": {
            "G": {
                "cycle": {"green_s": 5.0, "yellow_s": 3.0, "red_s": 3.0},
                "lanelets": ["100", "200"]
            }
        }
    }
    path = tmp_path / "tl.json"
    path.write_text(json.dumps(data))
    model = load_traffic_lights(str(path))
    assert model.total_lanelets() == 2
    assert model.has("100")
    assert not model.has("999")
    assert abs(model.expected_wait_for("100") - 18.0 / 11.0) < 1e-6


def test_load_missing_file_returns_empty():
    model = load_traffic_lights("/nonexistent/path.json")
    assert model.total_lanelets() == 0
