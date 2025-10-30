import numpy as np

from src.gelnet.algorithm import (
    GelnetLoopResult,
    GlassoElnetResult,
    DpgelnetResult,
    gelnet,
    dpgelnet,
    connect,
    gelnet_loop1,
    glasso_elnet_fast,
)
from src.gelnet.solver import (
    CrossvalidationResult,
    crossvalidation,
    rope,
    target,
)


def _make_positive_definite_matrix(size: int) -> np.ndarray:
    rng = np.random.default_rng(1234)
    a = rng.standard_normal((size, size))
    mat = a @ a.T
    # Ensure diagonal dominance for numerical stability.
    mat += size * np.eye(size)
    return mat


def test_connect_identifies_components():
    ss = np.array(
        [
            [2.0, 0.9, 0.0, 0.0],
            [0.9, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 1.1],
            [0.0, 0.0, 1.1, 2.0],
        ]
    )
    l1 = np.full_like(ss, 0.5)

    i1, i2, membership, n_components = connect(ss, l1)

    # The matrix consists of two disconnected blocks: (0, 1) and (2, 3).
    assert n_components == 2
    np.testing.assert_array_equal(i2, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(i1, np.array([[0, 2], [1, 3]]))
    np.testing.assert_array_equal(membership, np.array([1, 1, 2, 2]))


def test_gelnet_loop1_runs_and_preserves_symmetry():
    size = 4
    s = _make_positive_definite_matrix(size)
    theta0 = np.linalg.inv(s)
    w0 = s.copy()
    l3 = np.full_like(s, 0.05)
    l4 = np.full_like(s, 0.1)
    target = np.zeros_like(s)

    result = gelnet_loop1(
        s,
        l3,
        l4,
        theta0,
        w0,
        target,
        max_outer=10,
        outer_threshold=1e-4,
        max_inner=20,
        inner_threshold=1e-5,
        penalise_diagonal=True,
    )

    assert isinstance(result, GelnetLoopResult)
    assert result.theta.shape == (size, size)
    assert result.w.shape == (size, size)
    np.testing.assert_allclose(result.theta, result.theta.T, atol=1e-10)
    np.testing.assert_allclose(result.w, result.w.T, atol=1e-10)
    assert np.all(np.diag(result.theta) > 0)


def test_glasso_elnet_fast_handles_block_structure():
    s = np.block(
        [
            [np.array([[2.5, 0.8], [0.8, 2.4]]), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.array([[2.2, 0.6], [0.6, 2.3]])],
        ]
    )
    lam = np.full_like(s, 0.1)
    alpha = 0.7
    l1 = lam * alpha
    l2 = lam * (1.0 - alpha)

    theta0 = np.diag(1.0 / (np.diag(s) + np.diag(l1)))
    w0 = s.copy()
    diag_w = np.diag(s) + np.diag(l1) + np.diag(l2) * np.diag(theta0)
    w0[np.diag_indices_from(w0)] = diag_w

    result = glasso_elnet_fast(
        s,
        l1,
        l2,
        theta0,
        w0,
        np.zeros_like(s),
        max_outer=20,
        outer_threshold=1e-5,
        max_inner=30,
        inner_threshold=1e-6,
        penalise_diagonal=True,
    )

    assert isinstance(result, GlassoElnetResult)
    np.testing.assert_allclose(result.theta[:2, 2:], np.zeros((2, 2)), atol=1e-12)
    np.testing.assert_allclose(result.theta[2:, :2], np.zeros((2, 2)), atol=1e-12)
    assert result.outer_iterations <= 20


def test_dpgelnet_runs_end_to_end():
    size = 5
    s = _make_positive_definite_matrix(size)
    zeros = [(0, 1)]

    result = dpgelnet(
        s,
        lambda_=0.25,
        alpha=0.6,
        zero=zeros,
        outer_maxit=30,
        outer_thr=1e-4,
        inner_maxit=40,
    )

    assert isinstance(result, DpgelnetResult)
    np.testing.assert_allclose(result.theta, result.theta.T, atol=1e-10)
    assert np.all(np.diag(result.theta) > 0)
    assert result.n_iter <= 30
    assert result.delta >= 0
    # Zero constrained edges should remain inactive.
    assert abs(result.theta[0, 1]) < 1e-10


def test_gelnet_aliases_dpgelnet():
    size = 4
    s = _make_positive_definite_matrix(size)

    direct = dpgelnet(s, lambda_=0.2, alpha=0.5, outer_maxit=20, outer_thr=1e-4)
    alias = gelnet(s, lambda_=0.2, alpha=0.5, outer_maxit=20, outer_thr=1e-4)

    assert isinstance(alias, DpgelnetResult)
    np.testing.assert_allclose(alias.theta, direct.theta)


def test_rope_returns_positive_definite_precision():
    s = _make_positive_definite_matrix(3)
    theta = rope(s, lambda_=0.5)

    np.testing.assert_allclose(theta, theta.T, atol=1e-10)
    sign, _ = np.linalg.slogdet(theta)
    assert sign > 0


def test_target_identity_and_vI():
    s = _make_positive_definite_matrix(4)
    identity = target(s=s, type="Identity")
    np.testing.assert_array_equal(identity, np.eye(4))

    vi = target(s=s, type="vI")
    expected = np.eye(4) / np.mean(np.diag(s))
    np.testing.assert_allclose(vi, expected)


def test_crossvalidation_returns_result():
    rng = np.random.default_rng(2024)
    y = rng.standard_normal((12, 4))
    lambdas = [0.1, 0.2, 0.3]

    result = crossvalidation(
        nfold=3,
        y=y,
        lambda_path=lambdas,
        alpha=0.6,
        outer_maxit=15,
        outer_thr=1e-4,
        inner_maxit=20,
    )

    assert isinstance(result, CrossvalidationResult)
    assert result.lambda_path[0] <= result.lambda_path[-1]
    np.testing.assert_allclose(result.theta, result.theta.T, atol=1e-10)
    assert result.w is not None
    assert result.n_iter is not None
