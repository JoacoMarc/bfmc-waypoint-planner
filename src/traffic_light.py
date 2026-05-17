"""Probabilistic traffic-light wait model.

A traffic light cycles through green → yellow → red. The car must stop on yellow
and red and can pass on green. Assuming the car arrives at a uniformly random
phase of the cycle, we compute the expected wait time:

  P(arrive in green)  = green_s / cycle_s         -> wait = 0
  P(arrive in yellow) = yellow_s / cycle_s        -> wait avg = yellow_s/2 + red_s
  P(arrive in red)    = red_s / cycle_s           -> wait avg = red_s/2

  E[wait] = (yellow_s / cycle_s) * (yellow_s/2 + red_s)
          + (red_s    / cycle_s) * (red_s / 2)

For BFMC (G=5, Y=3, R=3): E[wait] = 18/11 ≈ 1.636 s, much lower than a hard 5 s.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TLCycle:
    green_s: float = 5.0
    yellow_s: float = 3.0
    red_s: float = 3.0

    @property
    def cycle_s(self) -> float:
        return self.green_s + self.yellow_s + self.red_s

    def expected_wait_s(self) -> float:
        c = self.cycle_s
        if c <= 0:
            return 0.0
        wait_yellow = (self.yellow_s / c) * (self.yellow_s / 2.0 + self.red_s)
        wait_red = (self.red_s / c) * (self.red_s / 2.0)
        return float(wait_yellow + wait_red)

    def to_dict(self) -> dict:
        return {"green_s": self.green_s, "yellow_s": self.yellow_s, "red_s": self.red_s}


@dataclass
class TrafficLightModel:
    """Maps lanelet_id -> TLCycle. Lanelets not in the map are not affected."""
    by_lanelet: dict[str, TLCycle] = field(default_factory=dict)
    default_cycle: TLCycle = field(default_factory=TLCycle)

    def has(self, lanelet_id: str) -> bool:
        return lanelet_id in self.by_lanelet

    def expected_wait_for(self, lanelet_id: str) -> float:
        cycle = self.by_lanelet.get(lanelet_id)
        if cycle is None:
            return 0.0
        return cycle.expected_wait_s()

    def total_lanelets(self) -> int:
        return len(self.by_lanelet)

    def to_summary_dict(self) -> dict:
        # Used for cache hashing — sorted, deterministic.
        return {
            "default": self.default_cycle.to_dict(),
            "by_lanelet": {
                lid: cycle.to_dict()
                for lid, cycle in sorted(self.by_lanelet.items(), key=lambda kv: kv[0])
            },
        }


def _parse_cycle(d: dict | None, fallback: TLCycle) -> TLCycle:
    if not d:
        return fallback
    return TLCycle(
        green_s=float(d.get("green_s", fallback.green_s)),
        yellow_s=float(d.get("yellow_s", fallback.yellow_s)),
        red_s=float(d.get("red_s", fallback.red_s)),
    )


def load_traffic_lights(path: str | None) -> TrafficLightModel:
    """Load traffic-light cycles per lanelet from JSON.

    Format:
        {
          "default_cycle": {"green_s": ..., "yellow_s": ..., "red_s": ...},
          "zones": {
            "<zone_name>": {
              "cycle": {...},
              "lanelets": ["lid1", "lid2", ...]
            }
          }
        }

    Returns an empty model if the file is missing.
    """
    if not path or not os.path.exists(path):
        return TrafficLightModel()
    with open(path) as fh:
        data = json.load(fh)
    default_cycle = _parse_cycle(data.get("default_cycle"), TLCycle())
    by_lanelet: dict[str, TLCycle] = {}
    zones = data.get("zones") or {}
    for _zname, zdata in zones.items():
        if not isinstance(zdata, dict):
            continue
        cycle = _parse_cycle(zdata.get("cycle"), default_cycle)
        for lid in zdata.get("lanelets", []) or []:
            by_lanelet[str(lid)] = cycle
    return TrafficLightModel(by_lanelet=by_lanelet, default_cycle=default_cycle)
