"""Linear assignment (Hungarian) without SciPy.

Used for within-scan tracking and evaluation matching. Rectangular cost
matrices are padded; infinite costs are treated as forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.intp]
FloatArray = NDArray[np.float64]

UNMATCHED = -1


@dataclass(frozen=True)
class AssignmentGate:
    row_to_col: dict[int, int]
    unmatched_rows: tuple[int, ...]
    unmatched_cols: tuple[int, ...]
    matched: tuple[tuple[int, int], ...]


def linear_sum_assignment(cost: np.ndarray) -> tuple[IntArray, IntArray]:
    """Return ``(row_ind, col_ind)`` minimizing the sum of ``cost[row, col]``.

    Forbidden pairs should be ``inf``. Empty input returns empty index arrays.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be a 2-D matrix")
    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    finite = matrix[np.isfinite(matrix)]
    fill = float(finite.max() + 1.0) if finite.size else 1.0
    size = max(n_rows, n_cols)
    padded = np.full((size, size), fill, dtype=np.float64)
    padded[:n_rows, :n_cols] = np.where(np.isfinite(matrix), matrix, fill * 2.0)

    assignment = _munkres(padded)
    rows: list[int] = []
    cols: list[int] = []
    for row, col in enumerate(assignment):
        if row < n_rows and col < n_cols and np.isfinite(matrix[row, col]):
            rows.append(row)
            cols.append(int(col))
    return np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)


def assign_with_gate(cost: np.ndarray, unmatched_cost: float = 0.5) -> AssignmentGate:
    """Hungarian plus a dummy no-match option per row.

    A pair is kept only when its cost is strictly cheaper than ``unmatched_cost``.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be a 2-D matrix")
    n_rows, n_cols = matrix.shape
    if n_rows == 0:
        return AssignmentGate({}, (), tuple(range(n_cols)), ())

    padded = np.full((n_rows, n_cols + n_rows), np.inf, dtype=np.float64)
    padded[:, :n_cols] = matrix
    for index in range(n_rows):
        padded[index, n_cols + index] = unmatched_cost

    rows, cols = linear_sum_assignment(padded)
    row_to_col: dict[int, int] = {}
    matched: list[tuple[int, int]] = []
    unmatched_rows: list[int] = []
    used_cols: set[int] = set()
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if col >= n_cols or float(matrix[row, col]) > unmatched_cost:
            row_to_col[row] = UNMATCHED
            unmatched_rows.append(row)
            continue
        row_to_col[row] = col
        matched.append((row, col))
        used_cols.add(col)
    for row in range(n_rows):
        if row not in row_to_col:
            row_to_col[row] = UNMATCHED
            unmatched_rows.append(row)
    unmatched_cols = tuple(col for col in range(n_cols) if col not in used_cols)
    return AssignmentGate(row_to_col, tuple(unmatched_rows), unmatched_cols, tuple(matched))


def _munkres(cost: FloatArray) -> IntArray:
    """Square Hungarian (Kuhn-Munkres) via shortest augmenting paths."""
    n = cost.shape[0]
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.intp)
    way = np.zeros(n + 1, dtype=np.intp)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = float(minv[j])
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.zeros(n, dtype=np.intp)
    for j in range(1, n + 1):
        assignment[int(p[j]) - 1] = j - 1
    return assignment
