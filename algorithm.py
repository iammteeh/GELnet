"""Core GELnet routines translated from the original Fortran code.

The original implementation lives inside the Fortran routine
``GLassoElnetFast``.  That routine exposes a block-coordinate descent
solver that repeatedly calls ``gelnet_loop1`` (together with the helper
routines ``connect`` and ``row``).  The functions implemented here are a
faithful NumPy translation of that logic, retaining the original control
flow and variable naming where possible to ease verification against the
Fortran source.

The translation follows the reference implementation shipped with the
GLassoElnetFast R package.  See the paper "Graphical Elastic Net"
(https://arxiv.org/pdf/2101.02148) for a detailed algorithmic
description.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import numpy as np


_EPS = 1.0e-16


@dataclass
class GelnetLoopResult:
    """Container storing the state returned by :func:`gelnet_loop1`.

    Attributes
    ----------
    theta : :class:`numpy.ndarray`
        Updated precision matrix.
    w : :class:`numpy.ndarray`
        Updated working matrix ``W`` used by the coordinate descent
        iterations.
    bold_matrix : :class:`numpy.ndarray`
        Auxiliary matrix corresponding to the scaled off-diagonal
        entries of ``theta``.
    outer_iterations : int
        Number of outer iterations executed.
    max_delta : float
        Maximal column update encountered in the last outer iteration.
    converged : bool
        ``True`` if the routine stopped early because the convergence
        criterion was satisfied.
    """

    theta: np.ndarray
    w: np.ndarray
    bold_matrix: np.ndarray
    outer_iterations: int
    max_delta: float
    converged: bool


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot product between two 1-D arrays.

    A tiny helper that mirrors the behaviour of Fortran's
    ``DOT_PRODUCT`` intrinsic while guaranteeing the return value is a
    Python ``float``.
    """

    return float(np.dot(a, b))


def gelnet_loop1(
    s: np.ndarray,
    l3: np.ndarray,
    l4: np.ndarray,
    theta: np.ndarray,
    w: np.ndarray,
    target: np.ndarray,
    max_outer: int,
    outer_threshold: float,
    max_inner: int,
    inner_threshold: float,
    penalise_diagonal: bool,
    iact: bool | None = None,
    eps: float = _EPS,
) -> GelnetLoopResult:
    """Run the ``gelnet_loop1`` routine.

    Parameters
    ----------
    s, l3, l4, theta, w, target : :class:`numpy.ndarray`
        Square matrices representing, respectively, the sample
        covariance matrix, the L1 penalty, the L2 penalty, the current
        precision estimate, the auxiliary matrix ``W`` and the optional
        target matrix used when the diagonal is penalised.
    max_outer, max_inner : int
        Maximum numbers of outer and inner iterations.
    outer_threshold, inner_threshold : float
        Convergence thresholds corresponding to the Fortran variables
        ``thr`` and ``thr2``.
    penalise_diagonal : bool
        Mirror of the ``ipen`` logical flag in the Fortran code.  When
        ``True`` the diagonal is penalised and the target matrix is
        taken into account.
    iact : bool, optional
        Kept for parity with the Fortran signature.  The argument is not
        used but maintained to simplify interoperability with existing
        callers.
    eps : float, optional
        Small numerical constant safeguarding the inner threshold.

    Returns
    -------
    :class:`GelnetLoopResult`
        Object containing the updated matrices together with diagnostic
        information about the iteration.
    """

    del iact  # The flag is unused in the original routine.

    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("'s' must be a square matrix")
    n = s.shape[0]
    for name, arr in {
        "l3": l3,
        "l4": l4,
        "theta": theta,
        "w": w,
        "target": target,
    }.items():
        if arr.shape != (n, n):
            raise ValueError(f"'{name}' must have shape {(n, n)}")

    theta = np.array(theta, dtype=float, copy=True)
    w = np.array(w, dtype=float, copy=True)
    l3 = np.array(l3, dtype=float, copy=False)
    l4 = np.array(l4, dtype=float, copy=False)
    s = np.array(s, dtype=float, copy=False)
    target = np.array(target, dtype=float, copy=False)

    shr = float(np.sum(np.abs(s))) - float(np.sum(np.abs(np.diag(s))))
    if shr == 0.0:
        bold_matrix = np.zeros_like(theta)
        return GelnetLoopResult(theta, w, bold_matrix, 0, 0.0, True)

    throut = outer_threshold * shr / (n - 1)
    thrin = inner_threshold * shr / (n - 1) / n
    thrin = max(thrin, 2 * eps)

    diag_theta = np.diag(theta).copy()
    if np.any(diag_theta == 0.0):
        raise ValueError("Diagonal entries of 'theta' must be non-zero")

    bold_matrix = -theta / diag_theta[np.newaxis, :]
    np.fill_diagonal(bold_matrix, 0.0)

    converged = False
    last_delta = 0.0

    for outer in range(1, max_outer + 1):
        dly = 0.0
        for j in range(n):
            b12 = bold_matrix[:, j].copy()
            b12[j] = 0.0
            w12 = np.zeros(n, dtype=float)
            non_zero = np.flatnonzero(b12)
            for idx in non_zero:
                w12 += w[:, idx] * b12[idx]

            for _inner in range(max_inner):
                dlx = 0.0
                for i in range(n):
                    if i == j:
                        continue
                    a = s[i, j] - w12[i] + w[i, i] * b12[i]
                    b_val = abs(a) - l3[i, j]
                    if b_val > 0.0:
                        denom = w[i, i] + l4[i, j] * theta[j, j]
                        if denom == 0.0:
                            raise ZeroDivisionError(
                                "Encountered zero denominator while updating b12"
                            )
                        c = math.copysign(b_val, a) / denom
                    else:
                        c = 0.0
                    delta = c - b12[i]
                    if delta != 0.0:
                        b12[i] = c
                        w12 += delta * w[:, i]
                        dlx = max(dlx, abs(delta))
                if dlx < thrin:
                    break

            bold_matrix[:, j] = b12
            w12[j] = w[j, j]

            if penalise_diagonal:
                if target[j, j] == 0.0:
                    if l4[j, j] == 0.0:
                        denom = s[j, j] + l3[j, j] - _dot(w12, b12)
                        if denom == 0.0:
                            raise ZeroDivisionError(
                                "Encountered zero denominator while updating theta"
                            )
                        theta_jj = 1.0 / denom
                    else:
                        helper = s[j, j] + l3[j, j] - _dot(w12, b12)
                        theta_jj = (
                            -helper + math.sqrt(helper ** 2 + 4 * l4[j, j])
                        ) / (2 * l4[j, j])
                    w_jj = s[j, j] + l3[j, j] + l4[j, j] * theta_jj
                else:
                    test = 1.0 / target[j, j] + _dot(w12, b12) - s[j, j]
                    if test > l3[j, j]:
                        if l4[j, j] == 0.0:
                            denom = s[j, j] + l3[j, j] - _dot(w12, b12)
                            if denom == 0.0:
                                raise ZeroDivisionError(
                                    "Encountered zero denominator while updating theta"
                                )
                            theta_jj = 1.0 / denom
                        else:
                            helper = (
                                s[j, j]
                                + l3[j, j]
                                - l4[j, j] * target[j, j]
                                - _dot(w12, b12)
                            )
                            theta_jj = (
                                -helper + math.sqrt(helper ** 2 + 4 * l4[j, j])
                            ) / (2 * l4[j, j])
                        w_jj = s[j, j] + l3[j, j] + l4[j, j] * max(0.0, theta_jj - target[j, j])
                    elif test < -l3[j, j]:
                        if l4[j, j] == 0.0:
                            denom = s[j, j] - l3[j, j] - _dot(w12, b12)
                            if denom == 0.0:
                                raise ZeroDivisionError(
                                    "Encountered zero denominator while updating theta"
                                )
                            theta_jj = 1.0 / denom
                        else:
                            helper = (
                                s[j, j]
                                - l3[j, j]
                                - l4[j, j] * target[j, j]
                                - _dot(w12, b12)
                            )
                            theta_jj = (
                                -helper + math.sqrt(helper ** 2 + 4 * l4[j, j])
                            ) / (2 * l4[j, j])
                        w_jj = s[j, j] + l4[j, j] * min(0.0, theta_jj - target[j, j]) - l3[j, j]
                    else:
                        theta_jj = target[j, j]
                        w_jj = s[j, j] + test
                w[j, j] = w_jj
            else:
                denom = s[j, j] - _dot(w12, b12)
                if denom == 0.0:
                    raise ZeroDivisionError(
                        "Encountered zero denominator while updating theta"
                    )
                theta_jj = 1.0 / denom

            theta_d = theta_jj
            diff = np.abs(w12 - w[:, j])
            dly = max(dly, float(np.sum(diff)))

            theta[j, :] = -theta_d * b12
            theta[:, j] = -theta_d * b12
            theta[j, j] = theta_d

            w[j, :] = w12
            w[:, j] = w12

        last_delta = dly
        if dly < throut:
            converged = True
            last_delta = dly / (n - 1)
            break

    if not converged and n > 1:
        last_delta = last_delta / (n - 1)

    return GelnetLoopResult(theta, w, bold_matrix, outer, last_delta, converged)


@dataclass
class GlassoElnetResult:
    """Result produced by :func:`glasso_elnet_fast`."""

    theta: np.ndarray
    w: np.ndarray
    outer_iterations: int
    max_delta: float
    converged: bool


def glasso_elnet_fast(
    s: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    theta: np.ndarray,
    w: np.ndarray,
    target: np.ndarray,
    max_outer: int,
    outer_threshold: float,
    max_inner: int,
    inner_threshold: float,
    penalise_diagonal: bool,
) -> GlassoElnetResult:
    """High level solver mirroring the ``GLassoElnetFast`` routine."""

    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("'s' must be a square matrix")
    n = s.shape[0]
    for name, arr in {"l1": l1, "l2": l2, "theta": theta, "w": w, "target": target}.items():
        if arr.shape != (n, n):
            raise ValueError(f"'{name}' must have shape {(n, n)}")

    theta = np.array(theta, dtype=float, copy=True)
    w = np.array(w, dtype=float, copy=True)
    l1 = np.array(l1, dtype=float, copy=False)
    l2 = np.array(l2, dtype=float, copy=False)
    s = np.array(s, dtype=float, copy=False)
    target = np.array(target, dtype=float, copy=False)

    i1, i2, _membership, n_components = connect(s, l1)
    if n_components == 0:
        component_indices = [np.arange(n, dtype=int)]
    else:
        component_indices = [i2[start : end + 1] for start, end in i1.T]

    converged = True
    outer_iterations = 0
    max_delta = 0.0

    for indices in component_indices:
        if indices.size == 0:
            continue
        res = gelnet_loop1(
            s[np.ix_(indices, indices)],
            l1[np.ix_(indices, indices)],
            l2[np.ix_(indices, indices)],
            theta[np.ix_(indices, indices)],
            w[np.ix_(indices, indices)],
            target[np.ix_(indices, indices)],
            max_outer,
            outer_threshold,
            max_inner,
            inner_threshold,
            penalise_diagonal,
        )
        theta[np.ix_(indices, indices)] = res.theta
        w[np.ix_(indices, indices)] = res.w
        converged = converged and res.converged
        outer_iterations = max(outer_iterations, res.outer_iterations)
        max_delta = max(max_delta, res.max_delta)

    return GlassoElnetResult(theta, w, outer_iterations, max_delta, converged)


def _row(
    component_id: int,
    frontier: Sequence[int],
    ss: np.ndarray,
    l1: np.ndarray,
    membership: np.ndarray,
) -> List[int]:
    """Python translation of the ``row`` helper subroutine.

    Parameters mirror the original Fortran code.  The function returns
    the newly discovered vertices of ``component_id``.
    """

    new_vertices: List[int] = []
    p = ss.shape[0]
    for k in frontier:
        for j in range(p):
            if membership[j] > 0:
                continue
            if j == k:
                continue
            if abs(ss[j, k]) <= l1[j, k]:
                continue
            membership[j] = component_id
            new_vertices.append(j)
    return new_vertices


def connect(ss: np.ndarray, l1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Identify connected components of the GELnet dependency graph.

    The routine mirrors the behaviour of the Fortran ``connect``
    subroutine.  It returns the same pieces of information but formatted
    in a Python friendly way.  Indices are zero based.

    Returns
    -------
    i1 : ndarray of shape ``(2, n_components)``
        Start (inclusive) and end (inclusive) indices into ``i2`` for
        each connected component.
    i2 : ndarray
        Vertices grouped by connected component.
    i3 : ndarray
        Component membership for each vertex.  Component ids start at 1
        to match the Fortran implementation.
    n_components : int
        Total number of connected components discovered.
    """

    if ss.shape != l1.shape:
        raise ValueError("'ss' and 'l1' must have identical shapes")
    if ss.ndim != 2 or ss.shape[0] != ss.shape[1]:
        raise ValueError("'ss' must be a square matrix")

    p = ss.shape[0]
    membership = np.zeros(p, dtype=int)
    ordering: List[int] = []
    ranges: List[Tuple[int, int]] = []

    for k in range(p):
        if membership[k] > 0:
            continue

        component_id = len(ranges) + 1
        start_idx = len(ordering)
        ordering.append(k)
        membership[k] = component_id

        frontier = [k]
        new_vertices = _row(component_id, frontier, ss, l1, membership)
        while new_vertices:
            ordering.extend(new_vertices)
            frontier = new_vertices
            new_vertices = _row(component_id, frontier, ss, l1, membership)

        end_idx = len(ordering) - 1
        ranges.append((start_idx, end_idx))

    if ranges:
        i1 = np.array(ranges, dtype=int).T
    else:
        i1 = np.zeros((2, 0), dtype=int)

    i2 = np.array(ordering, dtype=int)
    return i1, i2, membership, len(ranges)


__all__ = [
    "GelnetLoopResult",
    "gelnet_loop1",
    "GlassoElnetResult",
    "glasso_elnet_fast",
    "connect",
]
