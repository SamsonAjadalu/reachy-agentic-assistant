"""Hungarian assignment and the no-match gate."""

from __future__ import annotations

import numpy as np

from vision.hungarian import UNMATCHED, assign_with_gate, linear_sum_assignment


def test_identity_matrix_assigns_diagonal() -> None:
    cost = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    rows, cols = linear_sum_assignment(cost)
    paired = dict(zip(rows.tolist(), cols.tolist(), strict=True))
    assert paired == {0: 0, 1: 1, 2: 2}


def test_rectangular_leaves_extra_column_unmatched() -> None:
    cost = np.array([[0.1, 0.9, 0.8], [0.9, 0.2, 0.7]])
    rows, _cols = linear_sum_assignment(cost)
    assert len(rows) == 2
    assert set(rows.tolist()) == {0, 1}


def test_gate_refuses_expensive_pairs() -> None:
    cost = np.array([[0.1, 0.9], [0.85, 0.9]])
    result = assign_with_gate(cost, unmatched_cost=0.4)
    assert result.row_to_col[0] == 0
    assert result.row_to_col[1] == UNMATCHED
    assert 1 in result.unmatched_rows


def test_empty_cost() -> None:
    result = assign_with_gate(np.zeros((0, 0)), unmatched_cost=0.5)
    assert result.matched == ()
