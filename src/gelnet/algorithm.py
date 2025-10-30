"""Minimal GELnet solver packaged in a single module.

The goal of this module is to expose a compact Python translation of the
``GLassoElnetFast`` Fortran routine.  Only the pieces required to run the
block coordinate descent solver are retained so that downstream projects
can depend on a single import location.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple

import jax.numpy as np


_EPS = 1.0e-16
_BIG = 1.0e10


@dataclass
class GelnetLoopResult:
    """Container storing the outcome of :func:`gelnet_loop1`."""

    theta: np.ndarray
    w: np.ndarray
    bold_matrix: np.ndarray
    outer_iterations: int
    max_delta: float
    converged: bool


@dataclass
class GlassoElnetResult:
    """Result produced by :func:`glasso_elnet_fast`."""

    theta: np.ndarray
    w: np.ndarray
    outer_iterations: int
    max_delta: float
    converged: bool


@dataclass
class DpgelnetResult:
    """Container mirroring the R ``dpgelnet`` return structure."""

    theta: np.ndarray
    w: np.ndarray
    n_iter: int
    delta: float
    converged: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "Theta": self.theta,
            "W": self.w,
            "niter": self.n_iter,
            "del": self.delta,
            "conv": self.converged,
        }


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a floating point dot product."""

    return float(np.dot(a, b))


def _as_lambda_matrix(lambda_input: np.ndarray | Sequence[float] | float, p: int) -> np.ndarray:
    """Normalise the ``lambda`` argument to a symmetric penalty matrix."""

    lam = np.asarray(lambda_input, dtype=float)
    if lam.ndim == 0:
        matrix = np.full((p, p), float(lam))
    elif lam.ndim == 1:
        if lam.size != p:
            raise ValueError("Vector-valued 'lambda' must have length 'p'")
        sqrt_lam = np.sqrt(lam)
        matrix = np.outer(sqrt_lam, sqrt_lam)
    elif lam.shape == (p, p):
        matrix = lam.astype(float, copy=True)
    else:
        raise ValueError("'lambda' must be a scalar, vector of length p, or a pxp matrix")

    matrix = (matrix + matrix.T) / 2.0
    return matrix


def _apply_zero_constraints(lambda_matrix: np.ndarray, zero: Iterable[Sequence[int]] | None) -> None:
    if zero is None:
        return
    for pair in zero:
        if len(pair) != 2:
            raise ValueError("Entries in 'zero' must be index pairs")
        i, j = map(int, pair)
        if min(i, j) < 0 or i >= lambda_matrix.shape[0] or j >= lambda_matrix.shape[1]:
            raise IndexError("Zero constraint indices out of bounds")
        lambda_matrix[i, j] = _BIG
        lambda_matrix[j, i] = _BIG


def _row(
    component_id: int,
    frontier: Sequence[int],
    ss: np.ndarray,
    l1: np.ndarray,
    membership: np.ndarray,
) -> List[int]:
    """Python translation of the ``row`` helper subroutine."""

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
    """Identify connected components of the GELnet dependency graph."""

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
    """Run the ``gelnet_loop1`` routine."""

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
                        theta[j, j] = 1.0 / denom
                    else:
                        help_val = s[j, j] + l3[j, j] - _dot(w12, b12)
                        discr = help_val**2 + 4 * l4[j, j]
                        theta[j, j] = (-help_val + math.sqrt(discr)) / (2 * l4[j, j])
                    w[j, j] = s[j, j] + l3[j, j] + l4[j, j] * theta[j, j]
                else:
                    test = 1.0 / target[j, j] + _dot(w12, b12) - s[j, j]
                    if test > l3[j, j]:
                        if l4[j, j] == 0.0:
                            denom = s[j, j] + l3[j, j] - _dot(w12, b12)
                            if denom == 0.0:
                                raise ZeroDivisionError(
                                    "Encountered zero denominator while updating theta"
                                )
                            theta[j, j] = 1.0 / denom
                        else:
                            help_val = (
                                s[j, j]
                                + l3[j, j]
                                - l4[j, j] * target[j, j]
                                - _dot(w12, b12)
                            )
                            discr = help_val**2 + 4 * l4[j, j]
                            theta[j, j] = (-help_val + math.sqrt(discr)) / (2 * l4[j, j])
                        w[j, j] = s[j, j] + l3[j, j] + l4[j, j] * max(0.0, theta[j, j] - target[j, j])
                    elif test < -l3[j, j]:
                        if l4[j, j] == 0.0:
                            denom = s[j, j] - l3[j, j] - _dot(w12, b12)
                            if denom == 0.0:
                                raise ZeroDivisionError(
                                    "Encountered zero denominator while updating theta"
                                )
                            theta[j, j] = 1.0 / denom
                        else:
                            help_val = (
                                s[j, j]
                                - l3[j, j]
                                - l4[j, j] * target[j, j]
                                - _dot(w12, b12)
                            )
                            discr = help_val**2 + 4 * l4[j, j]
                            theta[j, j] = (-help_val + math.sqrt(discr)) / (2 * l4[j, j])
                        w[j, j] = s[j, j] - l3[j, j] + l4[j, j] * min(0.0, theta[j, j] - target[j, j])
                    else:
                        theta[j, j] = target[j, j]
                        w[j, j] = s[j, j] + test
            else:
                denom = s[j, j] - _dot(w12, b12)
                if denom == 0.0:
                    raise ZeroDivisionError("Encountered zero denominator while updating theta")
                theta[j, j] = 1.0 / denom
                w[j, j] = s[j, j]

            theta_diag = theta[j, j]
            w12[j] = w[j, j]
            dly = max(dly, float(np.sum(np.abs(w12 - w[:, j]))))
            theta[j, :] = -theta_diag * b12
            theta[:, j] = -theta_diag * b12
            theta[j, j] = theta_diag
            w[j, :] = w12
            w[:, j] = w12

        if dly < throut:
            converged = True
            last_delta = dly / (n - 1)
            break
        last_delta = dly / (n - 1)

    return GelnetLoopResult(theta, w, bold_matrix, outer, last_delta, converged)


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


def dpgelnet(
    s: np.ndarray,
    lambda_: np.ndarray | Sequence[float] | float,
    alpha: float,
    zero: Iterable[Sequence[int]] | None = None,
    theta: np.ndarray | None = None,
    w: np.ndarray | None = None,
    target: np.ndarray | None = None,
    outer_maxit: int = 1000,
    outer_thr: float = 1e-5,
    inner_maxit: int = 1000,
    inner_thr: float | None = None,
    penalise_diagonal: bool = True,
) -> DpgelnetResult:
    """Python equivalent of the R ``dpgelnet`` helper."""

    s = np.asarray(s, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("'s' must be a square matrix")
    p = s.shape[0]

    lambda_matrix = _as_lambda_matrix(lambda_, p)
    _apply_zero_constraints(lambda_matrix, zero)

    l1 = lambda_matrix * alpha
    l2 = lambda_matrix * (1.0 - alpha)
    if target is None:
        target_matrix = np.zeros_like(s)
    else:
        target_matrix = np.array(target, dtype=float, copy=True)
        if target_matrix.shape != (p, p):
            raise ValueError("'target' must have shape (p, p)")

    if inner_thr is None:
        inner_thr = outer_thr / 10.0

    if penalise_diagonal:
        lambda1_diag = np.diag(lambda_matrix) * alpha
        if theta is None:
            theta = np.diag(1.0 / (lambda1_diag + np.diag(s)))
        else:
            theta = np.array(theta, dtype=float, copy=True)
            if theta.shape != (p, p):
                raise ValueError("'theta' must have shape (p, p)")
        if w is None:
            w = s.copy()
            diag_w = np.diag(s) + lambda1_diag + np.diag(l2) * np.diag(theta)
            w[np.diag_indices_from(w)] = diag_w
        else:
            w = np.array(w, dtype=float, copy=True)
            if w.shape != (p, p):
                raise ValueError("'w' must have shape (p, p)")
    else:
        if theta is None:
            theta = np.diag(1.0 / np.diag(s))
        else:
            theta = np.array(theta, dtype=float, copy=True)
            if theta.shape != (p, p):
                raise ValueError("'theta' must have shape (p, p)")
        if w is None:
            w = s.copy()
        else:
            w = np.array(w, dtype=float, copy=True)
            if w.shape != (p, p):
                raise ValueError("'w' must have shape (p, p)")

    result = glasso_elnet_fast(
        s,
        l1,
        l2,
        theta,
        w,
        target_matrix,
        outer_maxit,
        outer_thr,
        inner_maxit,
        inner_thr,
        penalise_diagonal,
    )

    return DpgelnetResult(
        theta=result.theta,
        w=result.w,
        n_iter=result.outer_iterations,
        delta=result.max_delta,
        converged=result.converged,
    )


def gelnet(**kwargs) -> DpgelnetResult:
    """Convenience wrapper returning :func:`dpgelnet`."""

    return dpgelnet(**kwargs)


__all__ = [
    "GelnetLoopResult",
    "GlassoElnetResult",
    "DpgelnetResult",
    "connect",
    "gelnet_loop1",
    "glasso_elnet_fast",
    "dpgelnet",
    "gelnet",
]