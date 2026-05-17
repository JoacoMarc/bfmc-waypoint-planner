"""Reachability analysis: which waypoints are reachable from a start within a budget.

For each candidate budget, run a quick check: is there ANY path that visits a
given waypoint with total time <= budget? We use the cost matrix's D[0, j] for
single-waypoint reachability. For "score under budget" we re-run the OP solver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .cost_matrix import CostMatrix
from .orienteering_solver import solve_orienteering


INF = float("inf")


@dataclass
class ReachabilityRow:
    budget_s: float
    n_reachable_single_hop: int
    op_score: int
    op_waypoints: int


def reachable_single_hop(cost_matrix: CostMatrix, budget_s: float) -> set[str]:
    """Return wp_ids that are reachable from start (node 0) in budget_s, ignoring
    that the car would need to come back. This is an upper bound on what an OP
    plan could reach as a first-or-second waypoint."""
    out: set[str] = set()
    for j, anchor in enumerate(cost_matrix.index_to_anchor):
        if anchor is None:
            continue
        d = float(cost_matrix.D[0, j])
        if np.isfinite(d) and d <= budget_s:
            out.add(anchor.wp_id)
    return out


def reachability_curve(
    *, cost_matrix: CostMatrix, scores, wp_id_per_node, budgets: list[float],
    restarts: int, sa_iterations: int, seed: int,
) -> list[ReachabilityRow]:
    out: list[ReachabilityRow] = []
    for b in budgets:
        single = reachable_single_hop(cost_matrix, b)
        sol = solve_orienteering(
            cost_matrix.D, scores, wp_id_per_node, budget_s=b,
            restarts=restarts, sa_iterations=sa_iterations, seed=seed, verbose=False,
        )
        op_wps = sum(1 for n in sol.sequence if wp_id_per_node[n])
        out.append(ReachabilityRow(
            budget_s=float(b),
            n_reachable_single_hop=len(single),
            op_score=int(sol.score),
            op_waypoints=op_wps,
        ))
    return out


def write_reachability_svg(rows: list[ReachabilityRow], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    bs = [r.budget_s for r in rows]
    fig, ax1 = plt.subplots(figsize=(8.0, 4.0))
    ax1.plot(bs, [r.op_score for r in rows], marker="o", color="#3aa0ff", label="OP score (pts)")
    ax1.plot(bs, [r.op_waypoints for r in rows], marker="s", color="#22cc44", label="OP waypoints")
    ax1.plot(bs, [r.n_reachable_single_hop for r in rows], marker="^", color="#ffaa00", linestyle="--", label="Single-hop reachable")
    ax1.set_xlabel("Budget (s)")
    ax1.set_ylabel("Score / waypoints")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def write_reachability_markdown(rows: list[ReachabilityRow], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lines = ["# Reachability vs budget", "",
             "| budget (s) | OP score | OP waypoints | reachable in 1 hop |",
             "|----------:|---------:|-------------:|-------------------:|"]
    for r in rows:
        lines.append(f"| {r.budget_s:.0f} | {r.op_score} | {r.op_waypoints} | {r.n_reachable_single_hop} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
