import sys
import os
import numpy as np
import torch
import pytest

# Make local modules importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from implementation._torch_gmm_em import (
    _estimate_log_gaussian_prob_diag,
    _estimate_log_gaussian_prob_spherical,
    _estimate_log_gaussian_prob_tied,
    _estimate_log_gaussian_prob_full,
    _expectation_step,
    _maximization_step
)
from sklearn_mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _estimate_log_gaussian_prob as _sk_estimate_log_gaussian_prob,
    _estimate_gaussian_parameters as _sk_estimate_gaussian_parameters,
)

torch.set_default_dtype(torch.float64)


def _random_cov(rng, K, D, cov_type):
    reg = 1e-6
    if cov_type == "diag":
        return (0.5 + rng.rand(K, D)) ** 2
    if cov_type == "spherical":
        return (0.5 + rng.rand(K)) ** 2
    if cov_type == "tied":
        A = rng.randn(D, D)
        C = A @ A.T
        C /= np.trace(C) / D
        return C + reg * np.eye(D)
    if cov_type == "full":
        covs = []
        for _ in range(K):
            A = rng.randn(D, D)
            C = A @ A.T
            C /= np.trace(C) / D
            covs.append(C + reg * np.eye(D))
        return np.stack(covs, axis=0)
    raise ValueError(cov_type)


def _torch_log_prob(X_t, means_t, cov_t, cov_type):
    if cov_type == "diag":
        return _estimate_log_gaussian_prob_diag(X_t, means_t, cov_t)
    if cov_type == "spherical":
        return _estimate_log_gaussian_prob_spherical(X_t, means_t, cov_t)
    if cov_type == "tied":
        return _estimate_log_gaussian_prob_tied(X_t, means_t, cov_t)
    return _estimate_log_gaussian_prob_full(X_t, means_t, cov_t)


@pytest.mark.parametrize("cov_type", ["diag", "spherical", "tied", "full"])
def test_log_prob_matches_sklearn(cov_type):
    rng = np.random.RandomState(0)
    N, K, D = 40, 3, 4

    X = rng.randn(N, D)
    means = rng.randn(K, D) * 2.0
    cov = _random_cov(rng, K, D, cov_type)

    # Torch side
    X_t = torch.from_numpy(X)
    means_t = torch.from_numpy(means)
    cov_t = torch.from_numpy(cov)
    log_prob_t = _torch_log_prob(X_t, means_t, cov_t, cov_type).numpy()

    # Sklearn side
    prec_chol = _compute_precision_cholesky(cov, cov_type)
    log_prob_sk = _sk_estimate_log_gaussian_prob(X, means, prec_chol, cov_type)

    np.testing.assert_allclose(log_prob_t, log_prob_sk, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cov_type", ["diag", "spherical", "tied", "full"])
def test_responsibilities_match_sklearn(cov_type):
    rng = np.random.RandomState(1)
    N, K, D = 30, 2, 3

    X = rng.randn(N, D)
    means = rng.randn(K, D)
    cov = _random_cov(rng, K, D, cov_type)
    weights = rng.rand(K)
    weights /= weights.sum()

    # Torch responsibilities
    X_t = torch.from_numpy(X)
    means_t = torch.from_numpy(means)
    cov_t = torch.from_numpy(cov)
    weights_t = torch.from_numpy(weights)
    resp_t = _expectation_step(X_t, means_t, cov_t, weights_t, cov_type).numpy()

    # Sklearn responsibilities (softmax over log prob + log weights)
    prec_chol = _compute_precision_cholesky(cov, cov_type)
    log_prob_sk = _sk_estimate_log_gaussian_prob(X, means, prec_chol, cov_type)
    log_resp_unnorm = log_prob_sk + np.log(weights)[None, :]
    log_norm = np.logaddexp.reduce(log_resp_unnorm, axis=1, keepdims=True)
    resp_sk = np.exp(log_resp_unnorm - log_norm)

    np.testing.assert_allclose(resp_t, resp_sk, rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize("cov_type", ["diag", "spherical", "tied", "full"])
def test_m_step_matches_sklearn(cov_type):
    rng = np.random.RandomState(2)
    N, K, D = 50, 3, 4
    reg_covar = 1e-6

    X = rng.randn(N, D)
    resp = rng.rand(N, K)
    resp /= resp.sum(axis=1, keepdims=True)

    X_t = torch.from_numpy(X)
    resp_t = torch.from_numpy(resp)
    dummy_means = torch.zeros(K, D, dtype=torch.float64)
    dummy_cov = {
        "diag": torch.ones(K, D, dtype=torch.float64),
        "spherical": torch.ones(K, dtype=torch.float64),
        "tied": torch.eye(D, dtype=torch.float64),
        "full": torch.stack([torch.eye(D, dtype=torch.float64) for _ in range(K)]),
    }[cov_type]
    dummy_weights = torch.full((K,), 1.0 / K, dtype=torch.float64)

    means_t, cov_t, weights_t = _maximization_step(
        X_t, dummy_means, dummy_cov, dummy_weights, resp_t, cov_type, reg_covar=reg_covar
    )

    nk_sk, means_sk, cov_sk = _sk_estimate_gaussian_parameters(
        X, resp, reg_covar, cov_type
    )
    weights_sk = nk_sk / nk_sk.sum()

    np.testing.assert_allclose(means_t.numpy(), means_sk, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cov_t.numpy(), cov_sk, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(weights_t.numpy(), weights_sk, rtol=1e-5, atol=1e-6)

def _sk_e_step(X, means, cov, weights, cov_type):
    prec_chol = _compute_precision_cholesky(cov, cov_type)
    log_prob = _sk_estimate_log_gaussian_prob(X, means, prec_chol, cov_type)
    log_resp_unnorm = log_prob + np.log(weights)[None, :]
    log_norm = np.logaddexp.reduce(log_resp_unnorm, axis=1, keepdims=True)
    resp = np.exp(log_resp_unnorm - log_norm)
    return resp


@pytest.mark.parametrize("cov_type", ["diag", "spherical", "tied", "full"])
def test_em_loop_matches_sklearn(cov_type):
    rng = np.random.RandomState(3)
    N, K, D = 60, 3, 4
    reg_covar = 1e-6
    n_iter = 4

    X = rng.randn(N, D)
    means0 = rng.randn(K, D)
    cov0 = _random_cov(rng, K, D, cov_type)
    weights0 = rng.rand(K)
    weights0 /= weights0.sum()

    # Torch state
    means_t = torch.from_numpy(means0.copy())
    cov_t = torch.from_numpy(cov0.copy())
    weights_t = torch.from_numpy(weights0.copy())
    X_t = torch.from_numpy(X)

    # Sklearn state
    means_sk = means0.copy()
    cov_sk = cov0.copy()
    weights_sk = weights0.copy()

    for _ in range(n_iter):
        # E-step
        resp_t = _expectation_step(X_t, means_t, cov_t, weights_t, cov_type)
        resp_sk = _sk_e_step(X, means_sk, cov_sk, weights_sk, cov_type)

        # M-step
        means_t, cov_t, weights_t = _maximization_step(
            X_t, means_t, cov_t, weights_t, resp_t, cov_type, reg_covar=reg_covar
        )
        nk_sk, means_sk, cov_sk = _sk_estimate_gaussian_parameters(
            X, resp_sk, reg_covar, cov_type
        )
        weights_sk = nk_sk / nk_sk.sum()

    np.testing.assert_allclose(means_t.numpy(), means_sk, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cov_t.numpy(), cov_sk, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(weights_t.numpy(), weights_sk, rtol=1e-4, atol=1e-5)

def _torch_log_likelihood(X_t, means_t, cov_t, weights_t, cov_type):
    log_prob = _torch_log_prob(X_t, means_t, cov_t, cov_type)
    log_w = torch.log(weights_t).unsqueeze(0)
    return torch.logsumexp(log_prob + log_w, dim=1).mean().item()


def _sk_log_likelihood(X, means, cov, weights, cov_type):
    prec_chol = _compute_precision_cholesky(cov, cov_type)
    log_prob = _sk_estimate_log_gaussian_prob(X, means, prec_chol, cov_type)
    log_w = np.log(weights)[None, :]
    return np.logaddexp.reduce(log_prob + log_w, axis=1).mean()


def _run_em_torch(X, means, cov, weights, cov_type, reg_covar, tol=1e-6, max_iter=50):
    X_t = torch.from_numpy(X)
    means_t = torch.from_numpy(means.copy())
    cov_t = torch.from_numpy(cov.copy())
    weights_t = torch.from_numpy(weights.copy())
    ll_hist = []
    for _ in range(max_iter):
        resp_t = _expectation_step(X_t, means_t, cov_t, weights_t, cov_type)
        means_t, cov_t, weights_t = _maximization_step(
            X_t, means_t, cov_t, weights_t, resp_t, cov_type, reg_covar=reg_covar
        )
        ll = _torch_log_likelihood(X_t, means_t, cov_t, weights_t, cov_type)
        ll_hist.append(ll)
        if len(ll_hist) > 1 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            break
    return means_t.numpy(), cov_t.numpy(), weights_t.numpy(), ll_hist


def _run_em_sklearn(X, means, cov, weights, cov_type, reg_covar, tol=1e-6, max_iter=50):
    means_sk = means.copy()
    cov_sk = cov.copy()
    weights_sk = weights.copy()
    ll_hist = []
    for _ in range(max_iter):
        resp_sk = _sk_e_step(X, means_sk, cov_sk, weights_sk, cov_type)
        nk_sk, means_sk, cov_sk = _sk_estimate_gaussian_parameters(
            X, resp_sk, reg_covar, cov_type
        )
        weights_sk = nk_sk / nk_sk.sum()
        ll = _sk_log_likelihood(X, means_sk, cov_sk, weights_sk, cov_type)
        ll_hist.append(ll)
        if len(ll_hist) > 1 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            break
    return means_sk, cov_sk, weights_sk, ll_hist


@pytest.mark.parametrize("cov_type", ["diag", "spherical", "tied", "full"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_em_equivalence(cov_type, seed):
    rng = np.random.RandomState(seed)
    N, K, D = 80, 3, 4
    reg_covar = 1e-6

    X = rng.randn(N, D)
    means0 = rng.randn(K, D)
    cov0 = _random_cov(rng, K, D, cov_type)
    weights0 = rng.rand(K)
    weights0 /= weights0.sum()

    m_t, c_t, w_t, ll_t = _run_em_torch(X, means0, cov0, weights0, cov_type, reg_covar)
    m_s, c_s, w_s, ll_s = _run_em_sklearn(X, means0, cov0, weights0, cov_type, reg_covar)

    np.testing.assert_allclose(m_t, m_s, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(c_t, c_s, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(w_t, w_s, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(ll_t[-1], ll_s[-1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cov_type", ["diag", "spherical", "tied", "full"])
def test_torch_em_monotone_likelihood(cov_type):
    rng = np.random.RandomState(42)
    N, K, D = 60, 3, 4
    reg_covar = 1e-6

    X = rng.randn(N, D)
    means0 = rng.randn(K, D)
    cov0 = _random_cov(rng, K, D, cov_type)
    weights0 = rng.rand(K)
    weights0 /= weights0.sum()

    _, _, _, ll_hist = _run_em_torch(X, means0, cov0, weights0, cov_type, reg_covar, max_iter=20)
    assert all(ll_hist[i] >= ll_hist[i - 1] - 1e-10 for i in range(1, len(ll_hist)))