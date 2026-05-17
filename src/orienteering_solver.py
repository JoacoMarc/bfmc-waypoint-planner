"""Orienteering Problem solver: maximize collected score within a time budget.

Pipeline:
    1. Greedy nearest-neighbor with ratio score / cost_marginal.
    2. 2-opt to reduce path cost (enables inserting more nodes).
    3. Or-opt (move a single node) to escape local optima.
    4. Try insertion of un-visited nodes after each improvement.
    5. Simulated Annealing with perturbations (swap, insert, delete-cheap).
    6. Random restart (k restarts) and keep the best.

The OP node 0 is the virtual start; nodes 1..N are anchors. Each anchor has a wp_id;
the solver enforces deduplication by wp_id (each wp_id visited at most once).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from .constraints import Constraints, is_candidate_allowed


INF = float("inf")


@dataclass
class OPSolution:
    sequence: list[int]
    score: float
    cost_s: float
    visited_wp_ids: list[str]


def _path_cost(D: np.ndarray, seq: list[int]) -> float:
    if len(seq) < 2:
        return 0.0
    return float(sum(D[seq[i], seq[i + 1]] for i in range(len(seq) - 1)))


def _path_score(scores: np.ndarray, seq: list[int]) -> float:
    return float(sum(scores[i] for i in seq))


def _valid(seq: list[int], wp_id_per_node: list[str]) -> bool:
    seen: set[str] = set()
    for i in seq:
        wid = wp_id_per_node[i]
        if not wid:
            continue
        if wid in seen:
            return False
        seen.add(wid)
    return True


def greedy_nn(
    D: np.ndarray,
    scores: np.ndarray,
    wp_id_per_node: list[str],
    budget_s: float,
    start: int = 0,
    rng: random.Random | None = None,
    randomness: float = 0.0,
    constraints: Constraints | None = None,
) -> list[int]:
    """Greedy nearest-neighbor with score/cost ratio.

    If randomness > 0, pick from the top-k=3 candidates uniformly at random.
    If budget_s is infinite, the budget constraint is ignored and the greedy
    visits every reachable scoring node (best for "max waypoints" mode).
    """
    rng = rng or random.Random(0)
    N = D.shape[0]
    seq = [start]
    visited_wp: set[str] = set()
    if wp_id_per_node[start]:
        visited_wp.add(wp_id_per_node[start])
    elapsed = 0.0
    unlimited = budget_s == INF
    while True:
        last = seq[-1]
        candidates: list[tuple[float, int, float]] = []
        for j in range(N):
            if j in seq:
                continue
            wid = wp_id_per_node[j]
            if wid and wid in visited_wp:
                continue
            cost = float(D[last, j])
            if cost == INF:
                continue
            if not unlimited and elapsed + cost > budget_s:
                continue
            if constraints is not None and not is_candidate_allowed(
                j, seq, wp_id_per_node, constraints
            ):
                continue
            s = float(scores[j])
            if cost <= 1e-6:
                ratio = s * 1e6
            else:
                ratio = s / cost
            candidates.append((ratio, j, cost))
        if not candidates:
            break
        candidates.sort(key=lambda t: -t[0])
        if randomness > 0 and len(candidates) > 1 and rng.random() < randomness:
            top = candidates[: min(3, len(candidates))]
            _, j, cost = top[rng.randrange(len(top))]
        else:
            _, j, cost = candidates[0]
        seq.append(j)
        elapsed += cost
        if wp_id_per_node[j]:
            visited_wp.add(wp_id_per_node[j])
    return seq


def two_opt(
    seq: list[int], D: np.ndarray, scores: np.ndarray, budget_s: float
) -> list[int]:
    """2-opt: reverse a sub-path if doing so reduces total cost (respecting budget)."""
    if len(seq) < 4:
        return seq
    improved = True
    seq = list(seq)
    while improved:
        improved = False
        cost = _path_cost(D, seq)
        for i in range(1, len(seq) - 2):
            for k in range(i + 1, len(seq)):
                new_seq = seq[:i] + seq[i : k + 1][::-1] + seq[k + 1 :]
                new_cost = _path_cost(D, new_seq)
                if new_cost + 1e-9 < cost and new_cost <= budget_s:
                    seq = new_seq
                    cost = new_cost
                    improved = True
                    break
            if improved:
                break
    return seq


def or_opt(
    seq: list[int], D: np.ndarray, scores: np.ndarray, budget_s: float
) -> list[int]:
    """Try moving a single node to a different position to reduce cost."""
    if len(seq) < 4:
        return seq
    improved = True
    seq = list(seq)
    while improved:
        improved = False
        cost = _path_cost(D, seq)
        for i in range(1, len(seq)):
            node = seq[i]
            for j in range(1, len(seq)):
                if j == i:
                    continue
                new_seq = seq[:i] + seq[i + 1 :]
                insert_at = j if j < i else j - 1
                new_seq = new_seq[:insert_at] + [node] + new_seq[insert_at:]
                new_cost = _path_cost(D, new_seq)
                if new_cost + 1e-9 < cost and new_cost <= budget_s:
                    seq = new_seq
                    cost = new_cost
                    improved = True
                    break
            if improved:
                break
    return seq


def try_insert_unvisited(
    seq: list[int],
    D: np.ndarray,
    scores: np.ndarray,
    wp_id_per_node: list[str],
    budget_s: float,
    constraints: Constraints | None = None,
) -> list[int]:
    """Try to insert any unvisited (wp_id not in path) node at the best position."""
    seq = list(seq)
    in_path = set(seq)
    visited_wp = {wp_id_per_node[i] for i in seq if wp_id_per_node[i]}
    N = D.shape[0]
    candidates = [
        j
        for j in range(N)
        if j not in in_path
        and (not wp_id_per_node[j] or wp_id_per_node[j] not in visited_wp)
        and scores[j] > 0
    ]
    candidates.sort(key=lambda j: -float(scores[j]))
    cost = _path_cost(D, seq)
    for j in candidates:
        wid = wp_id_per_node[j]
        if wid and wid in visited_wp:
            continue
        best_delta = INF
        best_pos = -1
        for pos in range(1, len(seq) + 1):
            prev = seq[pos - 1]
            nxt = seq[pos] if pos < len(seq) else None
            # Constraint check: simulate the sequence after insertion.
            if constraints is not None:
                trial = seq[:pos] + [j] + seq[pos:]
                # Validate every wp position is compatible.
                ok = True
                acc: list[int] = []
                for node in trial:
                    if not is_candidate_allowed(node, acc, wp_id_per_node, constraints):
                        ok = False
                        break
                    acc.append(node)
                if not ok:
                    continue
            if nxt is None:
                delta = float(D[prev, j])
            else:
                delta = float(D[prev, j] + D[j, nxt] - D[prev, nxt])
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
        if best_pos == -1:
            continue
        if cost + best_delta <= budget_s:
            seq.insert(best_pos, j)
            cost += best_delta
            if wid:
                visited_wp.add(wid)
            in_path.add(j)
    return seq


def _sequence_respects_constraints(
    seq: list[int],
    wp_id_per_node: list[str],
    constraints: Constraints | None,
) -> bool:
    if constraints is None or constraints.is_empty:
        return True
    acc: list[int] = []
    for node in seq:
        if not is_candidate_allowed(node, acc, wp_id_per_node, constraints):
            return False
        acc.append(node)
    return True


def simulated_annealing(
    seq: list[int],
    D: np.ndarray,
    scores: np.ndarray,
    wp_id_per_node: list[str],
    budget_s: float,
    rng: random.Random,
    iterations: int = 5000,
    T0: float = 5.0,
    T_min: float = 0.05,
    constraints: Constraints | None = None,
) -> list[int]:
    """SA. Perturbations: swap, insert unvisited, delete random. Constraint-aware."""
    N = D.shape[0]
    seq = list(seq)
    cost = _path_cost(D, seq)
    score = _path_score(scores, seq)
    best_seq = list(seq)
    best_score = score
    T = T0
    decay = (T_min / T0) ** (1.0 / max(1, iterations))
    for _ in range(iterations):
        op = rng.random()
        new_seq = list(seq)
        visited_wp = {wp_id_per_node[i] for i in new_seq if wp_id_per_node[i]}
        if op < 0.4 and len(new_seq) >= 4:
            i = rng.randint(1, len(new_seq) - 1)
            k = rng.randint(1, len(new_seq) - 1)
            if i != k:
                new_seq[i], new_seq[k] = new_seq[k], new_seq[i]
        elif op < 0.7:
            candidates = [
                j
                for j in range(N)
                if j not in new_seq
                and (not wp_id_per_node[j] or wp_id_per_node[j] not in visited_wp)
                and scores[j] > 0
            ]
            if not candidates:
                continue
            j = rng.choice(candidates)
            pos = rng.randint(1, len(new_seq))
            new_seq.insert(pos, j)
        elif len(new_seq) >= 3:
            i = rng.randint(1, len(new_seq) - 1)
            new_seq.pop(i)
        else:
            continue

        new_cost = _path_cost(D, new_seq)
        new_score = _path_score(scores, new_seq)
        if new_cost > budget_s:
            continue
        if not _sequence_respects_constraints(new_seq, wp_id_per_node, constraints):
            continue
        delta = new_score - score
        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-9)):
            seq = new_seq
            cost = new_cost
            score = new_score
            if score > best_score or (score == best_score and cost < _path_cost(D, best_seq)):
                best_seq = list(seq)
                best_score = score
        T *= decay
    return best_seq


def solve_orienteering(
    D: np.ndarray,
    scores: np.ndarray,
    wp_id_per_node: list[str],
    budget_s: float = 600.0,
    restarts: int = 30,
    sa_iterations: int = 5000,
    seed: int = 42,
    verbose: bool = False,
    constraints: Constraints | None = None,
) -> OPSolution:
    """Run greedy + 2-opt + or-opt + insertion + SA with random restarts.
    Optional `constraints` enforces must_visit_first / forbidden / before_after."""
    rng = random.Random(seed)
    best_seq: list[int] = [0]
    best_score = float(scores[0])
    best_cost = 0.0

    for r in range(restarts):
        local_rng = random.Random(seed + r)
        randomness = 0.0 if r == 0 else 0.5
        seq = greedy_nn(
            D, scores, wp_id_per_node, budget_s, start=0, rng=local_rng,
            randomness=randomness, constraints=constraints,
        )
        seq = two_opt(seq, D, scores, budget_s)
        seq = try_insert_unvisited(seq, D, scores, wp_id_per_node, budget_s, constraints=constraints)
        seq = or_opt(seq, D, scores, budget_s)
        seq = try_insert_unvisited(seq, D, scores, wp_id_per_node, budget_s, constraints=constraints)
        seq = simulated_annealing(
            seq, D, scores, wp_id_per_node, budget_s, local_rng, sa_iterations,
            constraints=constraints,
        )
        seq = two_opt(seq, D, scores, budget_s)
        seq = try_insert_unvisited(seq, D, scores, wp_id_per_node, budget_s, constraints=constraints)
        cost = _path_cost(D, seq)
        score = _path_score(scores, seq)
        valid = cost <= budget_s or budget_s == INF
        if not _sequence_respects_constraints(seq, wp_id_per_node, constraints):
            valid = False
        better = (score > best_score) or (score == best_score and cost < best_cost)
        if valid and better:
            best_seq = list(seq)
            best_score = score
            best_cost = cost
            if verbose:
                print(f"  [restart {r}] new best: score={best_score}, cost={best_cost:.1f}s, len={len(best_seq)}")

    visited_wp_ids = [wp_id_per_node[i] for i in best_seq if wp_id_per_node[i]]
    return OPSolution(
        sequence=best_seq,
        score=best_score,
        cost_s=best_cost,
        visited_wp_ids=visited_wp_ids,
    )
