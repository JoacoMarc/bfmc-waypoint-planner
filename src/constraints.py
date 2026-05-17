"""Order constraints for the Orienteering Problem.

Allowed fields (all optional):
  - must_visit_first: a single wp_id that MUST be the first visited waypoint
    (BFMC rule: when starting in random mode, the team declares the first
    valid checkpoint; anything else touched before doesn't count).
  - must_visit:  list[wp_id] that must appear in the solution.
  - forbidden:   list[wp_id] that must NOT appear in the solution.
  - before_after: list[[A, B]] meaning A must be visited before B.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class Constraints:
    must_visit_first: str | None = None
    must_visit: list[str] = field(default_factory=list)
    forbidden: set[str] = field(default_factory=set)
    before_after: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            self.must_visit_first is None
            and not self.must_visit
            and not self.forbidden
            and not self.before_after
        )

    def to_summary_dict(self) -> dict:
        return {
            "must_visit_first": self.must_visit_first,
            "must_visit": sorted(self.must_visit),
            "forbidden": sorted(self.forbidden),
            "before_after": sorted([list(pair) for pair in self.before_after]),
        }


def load_constraints(path: str | None) -> Constraints:
    if not path or not os.path.exists(path):
        return Constraints()
    with open(path) as fh:
        data = json.load(fh)
    raw_pairs = data.get("before_after") or []
    pairs: list[tuple[str, str]] = []
    for p in raw_pairs:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            pairs.append((str(p[0]), str(p[1])))
    return Constraints(
        must_visit_first=(str(data["must_visit_first"]) if data.get("must_visit_first") else None),
        must_visit=[str(x) for x in (data.get("must_visit") or [])],
        forbidden=set(str(x) for x in (data.get("forbidden") or [])),
        before_after=pairs,
    )


def is_candidate_allowed(
    j: int,
    current_seq: list[int],
    wp_id_per_node: list[str],
    constraints: Constraints,
) -> bool:
    """Predicate used by the solver: True if adding j to `current_seq` is still
    compatible with the constraints. Used during construction AND mutation."""
    if constraints.is_empty:
        return True
    wp_j = wp_id_per_node[j] if 0 <= j < len(wp_id_per_node) else ""

    # 1. Forbidden
    if wp_j and wp_j in constraints.forbidden:
        return False

    # 2. must_visit_first: the FIRST visited waypoint (i.e., the first non-start node) must
    #    be the declared one. start node = 0; if current_seq == [0] and we're about to add
    #    a node with a wp_id, that wp_id must match.
    if constraints.must_visit_first is not None:
        first_wp_node = None
        for node in current_seq[1:]:
            if 0 <= node < len(wp_id_per_node) and wp_id_per_node[node]:
                first_wp_node = wp_id_per_node[node]
                break
        if first_wp_node is None and wp_j:
            # We're about to set the first wp.
            if wp_j != constraints.must_visit_first:
                return False

    # 3. before_after pairs: if wp_j is the "B" of a pair, "A" must already be in seq.
    if wp_j:
        seq_wps = {wp_id_per_node[n] for n in current_seq if 0 <= n < len(wp_id_per_node) and wp_id_per_node[n]}
        for a, b in constraints.before_after:
            if wp_j == b and a not in seq_wps:
                return False
            # Symmetrically: if wp_j == a, b must not already be in seq.
            if wp_j == a and b in seq_wps:
                return False

    return True


def validate_plan(
    sequence: list[int], wp_id_per_node: list[str], constraints: Constraints
) -> tuple[bool, list[str]]:
    """Post-hoc validation. Returns (ok, list_of_errors)."""
    errors: list[str] = []
    if constraints.is_empty:
        return True, errors
    seq_wps_in_order = [
        wp_id_per_node[n] for n in sequence if 0 <= n < len(wp_id_per_node) and wp_id_per_node[n]
    ]
    seq_wp_set = set(seq_wps_in_order)
    seq_wp_position = {wp: idx for idx, wp in enumerate(seq_wps_in_order)}

    if constraints.must_visit_first is not None:
        if not seq_wps_in_order:
            errors.append(f"must_visit_first={constraints.must_visit_first} but plan has no waypoints")
        elif seq_wps_in_order[0] != constraints.must_visit_first:
            errors.append(
                f"must_visit_first={constraints.must_visit_first} but plan first waypoint is "
                f"{seq_wps_in_order[0]}"
            )

    for wp in constraints.must_visit:
        if wp not in seq_wp_set:
            errors.append(f"must_visit waypoint {wp} not in plan")

    for wp in constraints.forbidden & seq_wp_set:
        errors.append(f"forbidden waypoint {wp} appears in plan")

    for a, b in constraints.before_after:
        ia = seq_wp_position.get(a)
        ib = seq_wp_position.get(b)
        if ia is None and ib is None:
            continue  # neither visited
        if ia is None:
            errors.append(f"before_after: required {a} before {b}, but {a} is absent and {b} is present")
        elif ib is None:
            continue  # a visited, b absent — fine
        elif ia >= ib:
            errors.append(f"before_after: {a} must come before {b}, but indices were {ia} >= {ib}")

    return (not errors), errors
