#!/usr/bin/env python3
"""E-step and M-step runtime benchmark: sklearn vs _v0_ref vs _v1.
Fixed: K=5, D=50. N in [1e4, 1e5, 1e6, 1e7].
10 timed runs per configuration.
Outputs CSV to stdout; progress and summary table to stderr.
"""

import sys
import os
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
    _estimate_log_gaussian_prob,
    _compute_precision_cholesky,
)
from implementation import _v0_ref
from implementation import _v1

K = 5
D = 50
N_VALUES = [10_000, 100_000, 1_000_000, 10_000_000]
N_RUNS = 10
COV_TYPE = "full"
REG_COVAR = 1e-6


def make_spd_covs(K, D, rng):
    covs = []
    for _ in range(K):
        A = rng.standard_normal((D, D))
        covs.append(A @ A.T + np.eye(D))
    return np.stack(covs).astype(np.float64)


def make_datasets(N, rng):
    """Pre-generate N_RUNS datasets for a given N, shared across all implementations."""
    datasets = []
    for _ in range(N_RUNS):
        weights_np = np.ones(K, dtype=np.float64) / K
        means_np = rng.standard_normal((K, D)).astype(np.float64)
        covs_np = make_spd_covs(K, D, rng)
        X_np = rng.standard_normal((N, D)).astype(np.float64)

        # sklearn: precisions_cholesky_ = inv(L).T  (upper-triangular, shape K×D×D)
        prec_chol_sk = np.zeros((K, D, D), dtype=np.float64)
        for k in range(K):
            L = np.linalg.cholesky(covs_np[k])
            prec_chol_sk[k] = np.linalg.solve(L, np.eye(D)).T

        # v0_ref / v1: precisions_cholesky = inv(L)  (lower-triangular, shape K×D×D)
        covs_t = torch.from_numpy(covs_np)
        prec_chol_t = _v0_ref._compute_precisions_cholesky(covs_t, COV_TYPE)

        datasets.append({
            "weights_np": weights_np,
            "means_np": means_np,
            "prec_chol_sk": prec_chol_sk,
            "covs_t": covs_t,
            "X_np": X_np,
            "X_t": torch.from_numpy(X_np),
            "means_t": torch.from_numpy(means_np),
            "weights_t": torch.from_numpy(weights_np),
            "prec_chol_t": prec_chol_t,
        })
    return datasets


# ---------------------------------------------------------------------------
# Sub-operation helpers (match each implementation's exact approach)
# ---------------------------------------------------------------------------

def _sk_mean_update(resp_np, X_np, nk_np):
    return np.dot(resp_np.T, X_np) / nk_np[:, None]


def _torch_mean_update(resp_t, X_t, nk_t):
    return (resp_t.T @ X_t) / nk_t.unsqueeze(1)


def _sk_cov_update(X_np, resp_np, nk_np, new_means_np):
    """numpy loop matching sklearn's _estimate_gaussian_covariances_full."""
    new_cov = np.empty((K, D, D))
    for k in range(K):
        diff_k = X_np - new_means_np[k]
        new_cov[k] = np.dot((resp_np[:, k:k+1] * diff_k).T, diff_k) / nk_np[k]
        new_cov[k].flat[::D + 1] += REG_COVAR
    return new_cov


def _v0_cov_update(X_t, resp_t, nk_t, new_means_t):
    """torch loop matching v0_ref's full covariance update."""
    diff = X_t.unsqueeze(1) - new_means_t.unsqueeze(0)  # (N,K,D)
    new_cov = torch.empty((K, D, D), device=X_t.device, dtype=X_t.dtype)
    eye = torch.eye(D, device=X_t.device, dtype=X_t.dtype)
    for k in range(K):
        wdiff = diff[:, k, :] * resp_t[:, k].unsqueeze(1)
        new_cov[k] = (wdiff.T @ diff[:, k, :]) / nk_t[k] + REG_COVAR * eye
    return new_cov


def _v1_cov_update(X_t, resp_t, nk_t, new_means_t):
    """torch einsum matching v1's full covariance update."""
    diff = X_t.unsqueeze(1) - new_means_t.unsqueeze(0)  # (N,K,D)
    cov_sum = torch.einsum('nk,nkd,nke->kde', resp_t, diff, diff)
    new_cov = cov_sum / nk_t.unsqueeze(1).unsqueeze(2)
    eye = torch.eye(D, device=X_t.device, dtype=X_t.dtype)
    return new_cov + REG_COVAR * eye.unsqueeze(0)


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _print_table(all_rows):
    ops = [
        ("E-step",      "sk_e",    "v0_e",    "v1_e"),
        ("M-step",      "sk_m",    "v0_m",    "v1_m"),
        ("Mahalanobis", "sk_mahal","v0_mahal","v1_mahal"),
        ("Mean update", "sk_mean", "v0_mean", "v1_mean"),
        ("Cov update",  "sk_cov",  "v0_cov",  "v1_cov"),
        ("Cholesky",    "sk_chol", "v0_chol", "v1_chol"),
    ]
    W = 76
    sep = "-" * W
    header = f"  {'Operation':<16}{'sklearn (ms)':>18}{'v0_ref (ms)':>20}{'v1 (ms)':>20}"

    print("", file=sys.stderr)
    print("=" * W, file=sys.stderr)
    print(f"  BENCHMARK RESULTS  K={K}, D={D}, COV_TYPE={COV_TYPE}, runs={N_RUNS}", file=sys.stderr)
    print("=" * W, file=sys.stderr)

    for row in all_rows:
        N = row["N"]
        print(f"\n  N = {N:,}", file=sys.stderr)
        print(sep, file=sys.stderr)
        print(header, file=sys.stderr)
        print(sep, file=sys.stderr)
        for op_name, sk_key, v0_key, v1_key in ops:
            sk_s = f"{row[f'{sk_key}_mean_ms']:.3f} ± {row[f'{sk_key}_std_ms']:.3f}"
            v0_s = f"{row[f'{v0_key}_mean_ms']:.3f} ± {row[f'{v0_key}_std_ms']:.3f}"
            v1_s = f"{row[f'{v1_key}_mean_ms']:.3f} ± {row[f'{v1_key}_std_ms']:.3f}"
            print(f"  {op_name:<16}{sk_s:>18}{v0_s:>20}{v1_s:>20}", file=sys.stderr)
        print(sep, file=sys.stderr)

    print("", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    all_rows = []

    for i, N in enumerate(N_VALUES):
        print(f"[{i+1}/{len(N_VALUES)}] N={N:,} — generating datasets...", file=sys.stderr, flush=True)
        datasets = make_datasets(N, rng)
        print(f"[{i+1}/{len(N_VALUES)}] N={N:,} — running benchmarks...", file=sys.stderr, flush=True)

        # --- Warmup on first dataset (untimed) ---
        ds0 = datasets[0]
        wu_model = GaussianMixture(n_components=K, covariance_type=COV_TYPE)
        wu_model.weights_ = ds0["weights_np"]
        wu_model.means_ = ds0["means_np"]
        wu_model.precisions_cholesky_ = ds0["prec_chol_sk"]

        _, wu_log_resp_sk = wu_model._e_step(ds0["X_np"])
        wu_model._m_step(ds0["X_np"], wu_log_resp_sk)

        _, wu_log_resp_v0 = _v0_ref._expectation_step_precchol(
            ds0["X_t"], ds0["means_t"], ds0["prec_chol_t"], ds0["weights_t"], COV_TYPE
        )
        _v0_ref._maximization_step(
            ds0["X_t"], ds0["means_t"], ds0["covs_t"], ds0["weights_t"], wu_log_resp_v0, COV_TYPE
        )

        _, wu_log_resp_v1 = _v1._expectation_step_precchol(
            ds0["X_t"], ds0["means_t"], ds0["prec_chol_t"], ds0["weights_t"], COV_TYPE
        )
        _v1._maximization_step(
            ds0["X_t"], ds0["means_t"], ds0["covs_t"], ds0["weights_t"], wu_log_resp_v1, COV_TYPE
        )

        # Warmup sub-operations
        wu_resp_np = np.exp(wu_log_resp_sk)
        wu_nk_np   = wu_resp_np.sum(0) + 10.0 * np.finfo(np.float64).eps
        wu_means_np = _sk_mean_update(wu_resp_np, ds0["X_np"], wu_nk_np)
        wu_resp_t  = wu_log_resp_v0.exp()
        wu_nk_t    = wu_resp_t.sum(0) + float(_v0_ref._nk_eps(wu_resp_t.dtype))
        wu_means_t  = _torch_mean_update(wu_resp_t, ds0["X_t"], wu_nk_t)

        _estimate_log_gaussian_prob(ds0["X_np"], ds0["means_np"], ds0["prec_chol_sk"], COV_TYPE)
        _v0_ref._estimate_log_gaussian_prob_full_precchol(ds0["X_t"], ds0["means_t"], ds0["prec_chol_t"])
        _v1._estimate_log_gaussian_prob_full_precchol(ds0["X_t"], ds0["means_t"], ds0["prec_chol_t"])

        _sk_mean_update(wu_resp_np, ds0["X_np"], wu_nk_np)
        _torch_mean_update(wu_resp_t, ds0["X_t"], wu_nk_t)

        wu_new_cov_sk = _sk_cov_update(ds0["X_np"], wu_resp_np, wu_nk_np, wu_means_np)
        wu_new_cov_v0 = _v0_cov_update(ds0["X_t"], wu_resp_t, wu_nk_t, wu_means_t)
        wu_new_cov_v1 = _v1_cov_update(ds0["X_t"], wu_resp_t, wu_nk_t, wu_means_t)

        _compute_precision_cholesky(wu_new_cov_sk, COV_TYPE)
        _v0_ref._compute_precisions_cholesky(wu_new_cov_v0, COV_TYPE)
        _v1._compute_precisions_cholesky(wu_new_cov_v1, COV_TYPE)

        # --- Timing lists ---
        sk_e_times, v0_e_times, v1_e_times         = [], [], []
        sk_m_times, v0_m_times, v1_m_times         = [], [], []
        sk_mahal_times, v0_mahal_times, v1_mahal_times = [], [], []
        sk_mean_times, v0_mean_times, v1_mean_times = [], [], []
        sk_cov_times, v0_cov_times, v1_cov_times   = [], [], []
        sk_chol_times, v0_chol_times, v1_chol_times = [], [], []

        for ds in datasets:
            X_t  = ds["X_t"]
            X_np = ds["X_np"]

            model = GaussianMixture(n_components=K, covariance_type=COV_TYPE)
            model.weights_ = ds["weights_np"]
            model.means_   = ds["means_np"]
            model.precisions_cholesky_ = ds["prec_chol_sk"]

            # --- E-step ---
            t0 = time.perf_counter()
            _, log_resp_sk = model._e_step(X_np)
            sk_e_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _, log_resp_v0 = _v0_ref._expectation_step_precchol(
                X_t, ds["means_t"], ds["prec_chol_t"], ds["weights_t"], COV_TYPE
            )
            v0_e_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _, log_resp_v1 = _v1._expectation_step_precchol(
                X_t, ds["means_t"], ds["prec_chol_t"], ds["weights_t"], COV_TYPE
            )
            v1_e_times.append((time.perf_counter() - t0) * 1e3)

            # --- M-step ---
            t0 = time.perf_counter()
            model._m_step(X_np, log_resp_sk)
            sk_m_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _v0_ref._maximization_step(
                X_t, ds["means_t"], ds["covs_t"], ds["weights_t"], log_resp_v0, COV_TYPE
            )
            v0_m_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _v1._maximization_step(
                X_t, ds["means_t"], ds["covs_t"], ds["weights_t"], log_resp_v1, COV_TYPE
            )
            v1_m_times.append((time.perf_counter() - t0) * 1e3)

            # --- Pre-compute intermediates (untimed) ---
            resp_np  = np.exp(log_resp_sk)
            nk_np    = resp_np.sum(0) + 10.0 * np.finfo(np.float64).eps
            means_np = _sk_mean_update(resp_np, X_np, nk_np)

            resp_t   = log_resp_v0.exp()
            nk_t     = resp_t.sum(0) + float(_v0_ref._nk_eps(resp_t.dtype))
            means_t  = _torch_mean_update(resp_t, X_t, nk_t)

            # --- Mahalanobis distance (E-step log-prob computation) ---
            t0 = time.perf_counter()
            _estimate_log_gaussian_prob(X_np, ds["means_np"], ds["prec_chol_sk"], COV_TYPE)
            sk_mahal_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _v0_ref._estimate_log_gaussian_prob_full_precchol(X_t, ds["means_t"], ds["prec_chol_t"])
            v0_mahal_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _v1._estimate_log_gaussian_prob_full_precchol(X_t, ds["means_t"], ds["prec_chol_t"])
            v1_mahal_times.append((time.perf_counter() - t0) * 1e3)

            # --- Mean update (M-step) ---
            t0 = time.perf_counter()
            _sk_mean_update(resp_np, X_np, nk_np)
            sk_mean_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _torch_mean_update(resp_t, X_t, nk_t)
            v0_mean_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _torch_mean_update(resp_t, X_t, nk_t)
            v1_mean_times.append((time.perf_counter() - t0) * 1e3)

            # --- Covariance update (M-step); output reused as cholesky input ---
            t0 = time.perf_counter()
            new_cov_sk = _sk_cov_update(X_np, resp_np, nk_np, means_np)
            sk_cov_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            new_cov_v0 = _v0_cov_update(X_t, resp_t, nk_t, means_t)
            v0_cov_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            new_cov_v1 = _v1_cov_update(X_t, resp_t, nk_t, means_t)
            v1_cov_times.append((time.perf_counter() - t0) * 1e3)

            # --- Cholesky factorization (M-step) ---
            t0 = time.perf_counter()
            _compute_precision_cholesky(new_cov_sk, COV_TYPE)
            sk_chol_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _v0_ref._compute_precisions_cholesky(new_cov_v0, COV_TYPE)
            v0_chol_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _v1._compute_precisions_cholesky(new_cov_v1, COV_TYPE)
            v1_chol_times.append((time.perf_counter() - t0) * 1e3)

        print(f"[{i+1}/{len(N_VALUES)}] N={N:,} — done.", file=sys.stderr, flush=True)

        def agg(lst):
            return round(float(np.mean(lst)), 4), round(float(np.std(lst)), 4)

        sk_e_m,     sk_e_s     = agg(sk_e_times)
        v0_e_m,     v0_e_s     = agg(v0_e_times)
        v1_e_m,     v1_e_s     = agg(v1_e_times)
        sk_m_m,     sk_m_s     = agg(sk_m_times)
        v0_m_m,     v0_m_s     = agg(v0_m_times)
        v1_m_m,     v1_m_s     = agg(v1_m_times)
        sk_mahal_m, sk_mahal_s = agg(sk_mahal_times)
        v0_mahal_m, v0_mahal_s = agg(v0_mahal_times)
        v1_mahal_m, v1_mahal_s = agg(v1_mahal_times)
        sk_mean_m,  sk_mean_s  = agg(sk_mean_times)
        v0_mean_m,  v0_mean_s  = agg(v0_mean_times)
        v1_mean_m,  v1_mean_s  = agg(v1_mean_times)
        sk_cov_m,   sk_cov_s   = agg(sk_cov_times)
        v0_cov_m,   v0_cov_s   = agg(v0_cov_times)
        v1_cov_m,   v1_cov_s   = agg(v1_cov_times)
        sk_chol_m,  sk_chol_s  = agg(sk_chol_times)
        v0_chol_m,  v0_chol_s  = agg(v0_chol_times)
        v1_chol_m,  v1_chol_s  = agg(v1_chol_times)

        all_rows.append({
            "N": N, "D": D, "K": K,
            "sk_e_mean_ms":     sk_e_m,     "sk_e_std_ms":     sk_e_s,
            "v0_e_mean_ms":     v0_e_m,     "v0_e_std_ms":     v0_e_s,
            "v1_e_mean_ms":     v1_e_m,     "v1_e_std_ms":     v1_e_s,
            "sk_m_mean_ms":     sk_m_m,     "sk_m_std_ms":     sk_m_s,
            "v0_m_mean_ms":     v0_m_m,     "v0_m_std_ms":     v0_m_s,
            "v1_m_mean_ms":     v1_m_m,     "v1_m_std_ms":     v1_m_s,
            "sk_mahal_mean_ms": sk_mahal_m, "sk_mahal_std_ms": sk_mahal_s,
            "v0_mahal_mean_ms": v0_mahal_m, "v0_mahal_std_ms": v0_mahal_s,
            "v1_mahal_mean_ms": v1_mahal_m, "v1_mahal_std_ms": v1_mahal_s,
            "sk_mean_mean_ms":  sk_mean_m,  "sk_mean_std_ms":  sk_mean_s,
            "v0_mean_mean_ms":  v0_mean_m,  "v0_mean_std_ms":  v0_mean_s,
            "v1_mean_mean_ms":  v1_mean_m,  "v1_mean_std_ms":  v1_mean_s,
            "sk_cov_mean_ms":   sk_cov_m,   "sk_cov_std_ms":   sk_cov_s,
            "v0_cov_mean_ms":   v0_cov_m,   "v0_cov_std_ms":   v0_cov_s,
            "v1_cov_mean_ms":   v1_cov_m,   "v1_cov_std_ms":   v1_cov_s,
            "sk_chol_mean_ms":  sk_chol_m,  "sk_chol_std_ms":  sk_chol_s,
            "v0_chol_mean_ms":  v0_chol_m,  "v0_chol_std_ms":  v0_chol_s,
            "v1_chol_mean_ms":  v1_chol_m,  "v1_chol_std_ms":  v1_chol_s,
        })

    _print_table(all_rows)

    df = pd.DataFrame(all_rows)
    output_path = os.path.join(os.path.dirname(__file__), "estep_benchmark.csv")
    df.to_csv(output_path, index=False)
    print(df.to_csv(index=False), end="")
    print(f"CSV saved to: {output_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
