"""High level helpers for running the GELnet solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import warnings

import numpy as np

from .algorithm import GlassoElnetResult, glasso_elnet_fast

try:  # pragma: no cover - optional dependency
    from sklearn.linear_model import LassoCV
except Exception:  # pragma: no cover - fallback when scikit-learn is unavailable
    LassoCV = None

_BIG = 1.0e10


@dataclass
class DpgelnetResult:
    """Container mirroring the list returned by the R ``dpgelnet`` helper."""

    theta: np.ndarray
    w: np.ndarray
    n_iter: int
    delta: float
    converged: bool

    def as_dict(self) -> dict[str, object]:
        """Return a dictionary compatible with the original R API."""

        return {
            "Theta": self.theta,
            "W": self.w,
            "niter": self.n_iter,
            "del": self.delta,
            "conv": self.converged,
        }


@dataclass
class CrossvalidationResult:
    """Container storing the outcome of :func:`crossvalidation`."""

    optimal: float
    lambda_path: np.ndarray
    cv_scores: np.ndarray
    theta: np.ndarray
    w: np.ndarray | None = None
    n_iter: int | None = None
    delta: float | None = None
    converged: bool | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a dictionary mirroring the original R API."""

        result: dict[str, object] = {
            "optimal": float(self.optimal),
            "lambda": np.asarray(self.lambda_path),
            "CV": np.asarray(self.cv_scores),
            "Theta": self.theta,
        }
        if self.w is not None:
            result["W"] = self.w
        if self.n_iter is not None:
            result["niter"] = int(self.n_iter)
        if self.delta is not None:
            result["del"] = float(self.delta)
        if self.converged is not None:
            result["conv"] = bool(self.converged)
        return result


def _as_lambda_matrix(lambda_input: np.ndarray | Sequence[float] | float, p: int) -> np.ndarray:
    """Normalise the ``lambda`` argument to a square penalty matrix."""

    lambda_arr = np.asarray(lambda_input, dtype=float)
    if lambda_arr.ndim == 0:
        lam = np.full((p, p), float(lambda_arr))
    elif lambda_arr.ndim == 1:
        if lambda_arr.size != p:
            raise ValueError("Vector-valued 'lambda' must have length equal to 'p'")
        sqrt_lambda = np.sqrt(lambda_arr)
        lam = np.outer(sqrt_lambda, sqrt_lambda)
    elif lambda_arr.shape == (p, p):
        lam = lambda_arr.astype(float, copy=True)
    else:
        raise ValueError("'lambda' must be a scalar, a vector of length p or a pxp matrix")

    lam = (lam + lam.T) / 2.0
    return lam


def _apply_zero_constraints(lambda_matrix: np.ndarray, zero: Iterable[Sequence[int]] | None) -> None:
    if zero is None:
        return
    for pair in zero:
        if len(pair) != 2:
            raise ValueError("Entries in 'zero' must be index pairs")
        i, j = map(int, pair)
        if i < 0 or j < 0 or i >= lambda_matrix.shape[0] or j >= lambda_matrix.shape[1]:
            raise IndexError("Zero constraints reference out-of-bounds indices")
        lambda_matrix[i, j] = _BIG
        lambda_matrix[j, i] = _BIG


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
    penalize_diagonal: bool = True,
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

    if penalize_diagonal:
        lambda1_diag = np.diag(lambda_matrix) * alpha
        if theta is None:
            theta = np.diag(1.0 / (lambda1_diag + np.diag(s)))
        else:
            theta = np.array(theta, dtype=float, copy=True)
            if theta.shape != (p, p):
                raise ValueError("'theta' must have shape (p, p)")
        if w is None:
            w = s.copy()
            diag_w = (
                np.diag(s)
                + lambda1_diag
                + np.diag(l2) * np.diag(theta)
            )
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

    result: GlassoElnetResult = glasso_elnet_fast(
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
        penalize_diagonal,
    )

    return DpgelnetResult(
        theta=result.theta,
        w=result.w,
        n_iter=result.outer_iterations,
        delta=result.max_delta,
        converged=result.converged,
    )


def gelnet(
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
    penalize_diagonal: bool = True,
) -> DpgelnetResult:
    """Convenience wrapper mirroring the public R ``gelnet`` helper."""

    return dpgelnet(
        s,
        lambda_,
        alpha,
        zero=zero,
        theta=theta,
        w=w,
        target=target,
        outer_maxit=outer_maxit,
        outer_thr=outer_thr,
        inner_maxit=inner_maxit,
        inner_thr=inner_thr,
        penalize_diagonal=penalize_diagonal,
    )


def rope(s: np.ndarray, lambda_: float, target: np.ndarray | None = None) -> np.ndarray:
    """Compute the ROPE precision estimate using a ridge-type penalty."""

    s = np.asarray(s, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("'s' must be a square matrix")
    if lambda_ < 0:
        raise ValueError("'lambda_' must be non-negative")

    p = s.shape[0]
    if target is None:
        target_matrix = np.zeros_like(s)
    else:
        target_matrix = np.asarray(target, dtype=float)
        if target_matrix.shape != (p, p):
            raise ValueError("'target' must have shape (p, p)")

    adjusted = s - lambda_ * target_matrix
    # ``np.linalg.eigh`` returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(adjusted)
    transformed = 2.0 / (eigenvalues + np.sqrt(eigenvalues**2 + 4.0 * lambda_))
    order = np.argsort(transformed)[::-1]
    lam_diag = np.diag(transformed[order])
    vectors_sorted = eigenvectors[:, order]
    theta = vectors_sorted @ lam_diag @ vectors_sorted.T
    return theta


def _covariance_matrix(data: np.ndarray) -> np.ndarray:
    return np.cov(data, rowvar=False, bias=False)


def _correlation_matrix(data: np.ndarray) -> np.ndarray:
    return np.corrcoef(data, rowvar=False, bias=False)


def _cov2cor(matrix: np.ndarray) -> np.ndarray:
    diag = np.diag(matrix)
    if np.any(diag <= 0):
        raise ValueError("Covariance matrix must have positive diagonal entries")
    scale = 1.0 / np.sqrt(diag)
    return matrix * scale[np.newaxis, :] * scale[:, np.newaxis]


def target(
    y: np.ndarray | None = None,
    s: np.ndarray | None = None,
    type: str | None = None,
    *,
    cor: bool = False,
    fraction: float = 1e-4,
    const: float = 1.0,
    safety_scaling: float = 1.0,
    nfolds_small: bool = True,
    nfolds_number: int | None = None,
) -> np.ndarray:
    """Compute target matrices mirroring the R helper of the same name."""

    if s is None and y is None:
        raise ValueError("Either 'y' or 's' must be provided")

    if s is not None:
        s = np.asarray(s, dtype=float)
        if s.ndim != 2 or s.shape[0] != s.shape[1]:
            raise ValueError("'s' must be a square matrix")
        p = s.shape[0]
    else:
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError("'y' must be a 2-D array")
        p = y.shape[1]

    if type is None:
        raise ValueError("'type' must be provided when calling target()")

    if cor:
        summary = _correlation_matrix
    else:
        summary = _covariance_matrix

    if type == "Identity":
        return np.eye(p)
    if type == "vI":
        if s is None:
            s = summary(y)
        mean_diag = np.mean(np.diag(s))
        if mean_diag == 0:
            raise ZeroDivisionError("Mean of the covariance diagonal is zero")
        return np.eye(p) / mean_diag
    if type == "Regression":
        if y is None:
            raise ValueError("'y' must be provided for Regression target")
        if safety_scaling < 0:
            raise ValueError("'safety_scaling' must be non-negative")
        if nfolds_small and nfolds_number is not None:
            warnings.warn(
                "'nfolds_number' ignored because 'nfolds_small' is True; using 10 folds",
                RuntimeWarning,
            )
        if LassoCV is None:
            raise ImportError(
                "scikit-learn is required for 'Regression' target computation"
            )
        folds = 10 if nfolds_small else int(nfolds_number or 10)
        tar = np.empty(p)
        rng = np.random.default_rng(0)
        for i in range(p):
            x = np.delete(y, i, axis=1)
            response = y[:, i]
            model = LassoCV(cv=folds, random_state=rng.integers(0, 2**32 - 1))
            model.fit(x, response)
            mse_path = model.mse_path_
            mean_mse = mse_path.mean(axis=1)
            std_mse = mse_path.std(axis=1, ddof=1)
            objective = mean_mse + safety_scaling * std_mse
            tar[i] = objective.min()
        if np.any(tar == 0):
            raise ZeroDivisionError("Encountered zero CV error while forming the target")
        return np.diag(1.0 / tar)
    if type == "Eigenvalue":
        if s is None:
            s = summary(y)
        eigenvalues = np.linalg.eigvalsh(s)
        largest = eigenvalues[-1]
        mask = eigenvalues >= largest * fraction
        if not np.any(mask):
            raise ValueError("No eigenvalues satisfy the fraction threshold")
        const_val = np.mean(1.0 / eigenvalues[mask])
        return const_val * np.eye(p)
    if type == "MSC":
        if s is None:
            s = summary(y)
        cor_matrix = _cov2cor(s)
        sorted_abs = np.sort(np.abs(cor_matrix), axis=0)
        second_largest = sorted_abs[-2, :]
        denom = np.diag(s) * (1.0 - second_largest**2)
        if np.any(denom == 0):
            raise ZeroDivisionError("Encountered zero denominator while forming MSC target")
        return np.diag(1.0 / denom)

    raise ValueError("Unsupported target type")


def crossvalidation(
    nfold: int,
    y: np.ndarray,
    lambda_path: Sequence[float],
    alpha: float,
    ind: Sequence[int] | None = None,
    *,
    type: str | None = None,
    target_matrix: np.ndarray | None = None,
    outer_maxit: int = 1000,
    outer_thr: float = 1e-5,
    inner_maxit: int = 1000,
    inner_thr: float | None = None,
    penalize_diagonal: bool = True,
    cor: bool = False,
    rope_mode: bool = False,
) -> CrossvalidationResult:
    """Cross-validate over lambda values mirroring the R implementation."""

    if y is None:
        raise ValueError("'y' is required as input")
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError("'y' must be a 2-D array")
    if nfold < 2:
        raise ValueError("'nfold' must be at least 2")

    lambda_arr = np.asarray(lambda_path, dtype=float)
    if lambda_arr.ndim != 1:
        raise ValueError("'lambda_path' must be a one-dimensional sequence")
    if lambda_arr.size == 0:
        raise ValueError("'lambda_path' must contain at least one value")
    lambda_arr = np.sort(lambda_arr)

    n_samples, p = y.shape
    if cor:
        summary = _correlation_matrix
    else:
        summary = _covariance_matrix

    if inner_thr is None:
        inner_thr = outer_thr / 10.0

    if ind is None:
        indices = np.arange(n_samples)
    else:
        indices = np.asarray(ind, dtype=int)
        if indices.shape != (n_samples,):
            raise ValueError("'ind' must be a permutation of length n")

    fold_sizes = [n_samples // nfold] * nfold
    for i in range(n_samples % nfold):
        fold_sizes[i] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start : start + size])
        start += size

    if type is None:
        if target_matrix is None:
            base_target = np.zeros((p, p))
        else:
            base_target = np.asarray(target_matrix, dtype=float)
            if base_target.shape != (p, p):
                raise ValueError("'target_matrix' must have shape (p, p)")

        def fold_target(data: np.ndarray) -> np.ndarray:
            _ = data  # unused
            return base_target

    else:

        def fold_target(data: np.ndarray) -> np.ndarray:
            return target(y=data, type=type, cor=cor)

    fold_targets = []
    for fold in folds:
        train_indices = np.setdiff1d(indices, fold, assume_unique=True)
        train_data = y[train_indices]
        fold_targets.append(fold_target(train_data))

    cv_scores = np.zeros_like(lambda_arr)

    def objective(theta: np.ndarray, sigma: np.ndarray) -> float:
        sign, logdet = np.linalg.slogdet(theta)
        if sign <= 0:
            raise ValueError("Precision matrix must be positive definite")
        return logdet - np.trace(sigma @ theta)

    for j, lam in enumerate(lambda_arr):
        for fold_indices, tar in zip(folds, fold_targets):
            train_indices = np.setdiff1d(indices, fold_indices, assume_unique=True)
            train_data = y[train_indices]
            test_data = y[fold_indices]
            strain = summary(train_data)
            if rope_mode:
                theta_train = rope(strain, lam, tar)
            else:
                result = gelnet(
                    strain,
                    lambda_=lam,
                    alpha=alpha,
                    target=tar,
                    outer_maxit=outer_maxit,
                    outer_thr=outer_thr,
                    inner_maxit=inner_maxit,
                    inner_thr=inner_thr,
                    penalize_diagonal=penalize_diagonal,
                )
                theta_train = result.theta
            sigma_test = summary(test_data)
            cv_scores[j] += objective(theta_train, sigma_test)

    cv_scores /= nfold
    best_index = int(np.argmax(cv_scores))
    if best_index == 0 or best_index == lambda_arr.size - 1:
        warnings.warn(
            "The optimal lambda value is a boundary value; consider widening the search range",
            RuntimeWarning,
        )
    optimal = float(lambda_arr[best_index])

    if rope_mode:
        full_sigma = summary(y)
        final_theta = rope(full_sigma, optimal, fold_target(y))
        result = CrossvalidationResult(
            optimal=optimal,
            lambda_path=lambda_arr,
            cv_scores=cv_scores,
            theta=final_theta,
        )
    else:
        full_sigma = summary(y)
        final = gelnet(
            full_sigma,
            lambda_=optimal,
            alpha=alpha,
            target=fold_target(y),
            outer_maxit=outer_maxit,
            outer_thr=outer_thr,
            inner_maxit=inner_maxit,
            inner_thr=inner_thr,
            penalize_diagonal=penalize_diagonal,
        )
        result = CrossvalidationResult(
            optimal=optimal,
            lambda_path=lambda_arr,
            cv_scores=cv_scores,
            theta=final.theta,
            w=final.w,
            n_iter=final.n_iter,
            delta=final.delta,
            converged=final.converged,
        )

    return result


__all__ = [
    "CrossvalidationResult",
    "DpgelnetResult",
    "crossvalidation",
    "dpgelnet",
    "gelnet",
    "rope",
    "target",
]