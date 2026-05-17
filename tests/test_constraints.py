"""Tests for src/constraints.py."""

import json
from pathlib import Path

from src.constraints import (
    Constraints, is_candidate_allowed, load_constraints, validate_plan,
)


def test_empty_constraints():
    c = Constraints()
    assert c.is_empty
    assert is_candidate_allowed(5, [0], ["", "a", "b", "c", "d", "e"], c)


def test_forbidden_blocks_candidate():
    c = Constraints(forbidden={"b"})
    wp = ["", "a", "b", "c"]
    assert not is_candidate_allowed(2, [0], wp, c)
    assert is_candidate_allowed(1, [0], wp, c)


def test_must_visit_first():
    c = Constraints(must_visit_first="b")
    wp = ["", "a", "b", "c"]
    # First wp must be "b"; trying to insert "a" first fails.
    assert not is_candidate_allowed(1, [0], wp, c)
    assert is_candidate_allowed(2, [0], wp, c)
    # Once "b" is in seq, anything else is allowed.
    assert is_candidate_allowed(1, [0, 2], wp, c)


def test_before_after():
    c = Constraints(before_after=[("a", "b")])
    wp = ["", "a", "b", "c"]
    # adding b before a is forbidden
    assert not is_candidate_allowed(2, [0], wp, c)
    # adding a first ok
    assert is_candidate_allowed(1, [0], wp, c)
    # after a, b is allowed
    assert is_candidate_allowed(2, [0, 1], wp, c)


def test_validate_plan_detects_first_mismatch():
    c = Constraints(must_visit_first="b")
    wp = ["", "a", "b", "c"]
    ok, errors = validate_plan([0, 1, 2], wp, c)
    assert not ok
    assert any("must_visit_first" in e for e in errors)


def test_validate_plan_detects_order_violation():
    c = Constraints(before_after=[("a", "b")])
    wp = ["", "a", "b", "c"]
    ok, errors = validate_plan([0, 2, 1], wp, c)
    assert not ok
    assert any("before_after" in e for e in errors)


def test_load_constraints_file(tmp_path):
    data = {
        "must_visit_first": "1424",
        "must_visit": ["1434"],
        "forbidden": ["1772"],
        "before_after": [["1424", "1434"]]
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data))
    c = load_constraints(str(p))
    assert c.must_visit_first == "1424"
    assert "1434" in c.must_visit
    assert "1772" in c.forbidden
    assert ("1424", "1434") in c.before_after
