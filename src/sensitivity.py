"""Sensitivity analysis: vary one parameter at a time and measure score.

Usage: from `tools/analyze_plan.py` with --sensitivity.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .cost_matrix import CostParams, build_cost_matrix
from .orienteering_solver import solve_orienteering
from .plan_writer import SCORE_RANDOM_START, SCORE_WAYPOINT, build_plan
from .topology import Lanelet
from .waypoint_anchor import WaypointAnchor


@dataclass
class SensitivityAxis:
    param: str
    values: list[float]


@dataclass
class SensitivityRow:
    value: float
    score: int
    time_s: float
    distance_m: float
    n_waypoints: int


@dataclass
class SensitivityResult:
    axis: SensitivityAxis
    rows: list[SensitivityRow] = field(default_factory=list)


def _apply_param(base: CostParams, param: str, value: float) -> CostParams:
    p = copy.deepcopy(base)
    if not hasattr(p, param):
        raise ValueError(f"Unknown parameter: {param}")
    setattr(p, param, float(value))
    return p


def run_sensitivity_sweep(
    *,
    osm_path: str,
    lanelets: dict[str, Lanelet],
    anchors: list[WaypointAnchor],
    start_exit_lanelets: list[tuple[str, tuple[float, float], float]],
    base_params: CostParams,
    axes: list[SensitivityAxis],
    budget_s: float,
    restarts: int,
    sa_iterations: int,
    seed: int,
    start_bonus: int,
    cache_dir: str | None,
    waypoints,
    start_pose,
    verbose: bool = False,
) -> list[SensitivityResult]:
    N = len(anchors)
    scores = np.zeros(N + 1, dtype=float)
    scores[0] = float(start_bonus)
    for i in range(1, N + 1):
        scores[i] = float(SCORE_WAYPOINT)
    wp_id_per_node = [""] + [a.wp_id for a in anchors]

    results: list[SensitivityResult] = []
    for axis in axes:
        result = SensitivityResult(axis=axis)
        for v in axis.values:
            params = _apply_param(base_params, axis.param, v)
            cm = build_cost_matrix(
                lanelets, anchors, start_exit_lanelets, params, osm_path,
                cache_dir=cache_dir, verbose=False,
            )
            sol = solve_orienteering(
                cm.D, scores, wp_id_per_node, budget_s=budget_s,
                restarts=restarts, sa_iterations=sa_iterations,
                seed=seed, verbose=False,
            )
            plan = build_plan(
                osm_path=osm_path, params=params, budget_s=budget_s,
                solution=sol, cost_matrix=cm, anchors=anchors, waypoints=waypoints,
                start_pose=start_pose, start_bonus=start_bonus,
            )
            row = SensitivityRow(
                value=float(v),
                score=int(plan.expected_score),
                time_s=float(plan.expected_total_time_s),
                distance_m=float(plan.expected_distance_m),
                n_waypoints=int(plan.expected_waypoints_visited),
            )
            result.rows.append(row)
            if verbose:
                print(f"  {axis.param}={v}: score={row.score} t={row.time_s:.0f}s wps={row.n_waypoints}")
        results.append(result)
    return results


def write_sensitivity_markdown(results: list[SensitivityResult], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lines = ["# Sensitivity report", ""]
    for r in results:
        lines.append(f"## Axis: `{r.axis.param}`")
        lines.append("")
        lines.append("| value | score | time (s) | distance (m) | waypoints |")
        lines.append("|------:|------:|---------:|-------------:|----------:|")
        for row in r.rows:
            lines.append(
                f"| {row.value:.3g} | {row.score} | {row.time_s:.1f} | "
                f"{row.distance_m:.2f} | {row.n_waypoints} |"
            )
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def write_sensitivity_svg(results: list[SensitivityResult], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = len(results)
    fig, axes_plt = plt.subplots(n, 1, figsize=(8.0, 3.0 * max(1, n)))
    if n == 1:
        axes_plt = [axes_plt]
    for ax, r in zip(axes_plt, results):
        xs = [row.value for row in r.rows]
        ys = [row.score for row in r.rows]
        ax.plot(xs, ys, marker="o", color="#3aa0ff")
        ax.set_xlabel(r.axis.param)
        ax.set_ylabel("score (pts)")
        ax.set_title(f"Score vs {r.axis.param}")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
