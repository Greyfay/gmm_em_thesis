#!/usr/bin/env python3
"""E-step and M-step runtime benchmark: sklearn vs _v0_ref vs _v1.
Fixed: K=5, N=1000. D in [5, 10, 20, 50, 100, 500, 1000, 5000].
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
N = 1000
D_VALUES = [5, 10, 20, 50, 100, 500, 1000, 5000]
N_RUNS = 30
COV_TYPE = "full"

# Single epsilon constant used for nk smoothing across both numpy and torch paths.
# Matches sklearn's convention: 10 * machine epsilon for float32.
_NK_EPS = float(10.0 * np.finfo(np.float32).eps)


def make_spd_covs(K, D, rng):
    covs = []
    for _ in range(K):
        A = rng.standard_normal((D, D))
        covs.append(A @ A.T + np.eye(D))
    return np.stack(covs).astype(np.float32)


def make_datasets(N, K, D, rng):
    """Pre-generate N_RUNS datasets for a given D, shared across all implementations."""
    datasets = []
    for _ in range(N_RUNS):
        weights_np = np.ones(K, dtype=np.float32) / K
        means_np = rng.standard_normal((K, D)).astype(np.float32)
        covs_np = make_spd_covs(K, D, rng)
        X_np = rng.standard_normal((N, D)).astype(np.float32)

        # sklearn: precisions_cholesky_ = inv(L).T  (upper-triangular, shape K×D×D)
        prec_chol_sk = np.zeros((K, D, D), dtype=np.float32)
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
# D is passed explicitly since it varies across the outer loop.
# ---------------------------------------------------------------------------

def _sk_mean_update(resp_np, X_np, nk_np):
    return np.dot(resp_np.T, X_np) / nk_np[:, None]


def _torch_mean_update(resp_t, X_t, nk_t):
    return (resp_t.T @ X_t) / nk_t.unsqueeze(1)


def _sk_cov_update(X_np, resp_np, nk_np, new_means_np, K, D, reg_covar=1e-6):
    """numpy loop matching sklearn's _estimate_gaussian_covariances_full."""
    new_cov = np.empty((K, D, D))
    for k in range(K):
        diff_k = X_np - new_means_np[k]
        new_cov[k] = np.dot((resp_np[:, k:k+1] * diff_k).T, diff_k) / nk_np[k]
        new_cov[k].flat[::D + 1] += reg_covar
    return new_cov


def _v0_cov_update(X_t, resp_t, nk_t, new_means_t, K, D, reg_covar=1e-4):
    """torch loop matching v0_ref's full covariance update."""
    diff = X_t.unsqueeze(1) - new_means_t.unsqueeze(0)  # (N,K,D)
    new_cov = torch.empty((K, D, D), device=X_t.device, dtype=X_t.dtype)
    eye = torch.eye(D, device=X_t.device, dtype=X_t.dtype)
    for k in range(K):
        wdiff = diff[:, k, :] * resp_t[:, k].unsqueeze(1)
        new_cov[k] = (wdiff.T @ diff[:, k, :]) / nk_t[k]
    # Add regularisation in one shot outside the loop, matching v0_ref exactly.
    return new_cov + reg_covar * eye


def _v1_cov_update(X_t, resp_t, nk_t, new_means_t, K, D, reg_covar=1e-4):
    """torch einsum matching v1's full covariance update."""
    diff = X_t.unsqueeze(1) - new_means_t.unsqueeze(0)  # (N,K,D)
    cov_sum = torch.einsum('nk,nkd,nke->kde', resp_t, diff, diff)
    new_cov = cov_sum / nk_t.unsqueeze(1).unsqueeze(2)
    eye = torch.eye(D, device=X_t.device, dtype=X_t.dtype)
    return new_cov + reg_covar * eye.unsqueeze(0)


def _agg(lst):
    return round(float(np.mean(lst)), 4), round(float(np.std(lst)), 4)


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _print_table(all_rows):
    ops = [
        ("E-step",      "sk_e",    "v0_e",    "v1_e"),
        ("M-step",      "sk_m",    "v0_m",    "v1_m"),
        ("Log-prob",    "sk_mahal","v0_mahal","v1_mahal"),
        ("Mean update", "sk_mean", "v0_mean", "v1_mean"),
        ("Cov update",  "sk_cov",  "v0_cov",  "v1_cov"),
        ("Cholesky",    "sk_chol", "v0_chol", "v1_chol"),
    ]
    W = 76
    sep = "-" * W
    header = f"  {'Operation':<16}{'sklearn (ms)':>18}{'v0_ref (ms)':>20}{'v1 (ms)':>20}"

    print("", file=sys.stderr)
    print("=" * W, file=sys.stderr)
    print(f"  BENCHMARK RESULTS  K={K}, N={N}, COV_TYPE={COV_TYPE}, runs={N_RUNS}", file=sys.stderr)
    print("=" * W, file=sys.stderr)

    for row in all_rows:
        D = row["D"]
        print(f"\n  D = {D:,}", file=sys.stderr)
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

    all_rows = []

    for i, D in enumerate(D_VALUES):
        print(f"[{i+1}/{len(D_VALUES)}] D={D:,} — generating datasets...", file=sys.stderr, flush=True)
        datasets = make_datasets(N, K, D, rng)
        print(f"[{i+1}/{len(D_VALUES)}] D={D:,} — running benchmarks...", file=sys.stderr, flush=True)

        # --- Warmup on first dataset (untimed) ---
        ds0 = datasets[0]

        # GaussianMixture is constructed once per D block and reused.
        # sklearn default reg_covar=1e-6, but we use 1e-4 for high-D stability
        model = GaussianMixture(n_components=K, covariance_type=COV_TYPE, reg_covar=1e-4)
        model.weights_ = ds0["weights_np"]
        model.means_   = ds0["means_np"]
        model.precisions_cholesky_ = ds0["prec_chol_sk"]

        _, wu_log_resp_sk = model._e_step(ds0["X_np"])
        model._m_step(ds0["X_np"], wu_log_resp_sk)

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

        # Warmup sub-operations — each implementation uses its own log_resp.
        wu_resp_np    = np.exp(wu_log_resp_sk)
        wu_nk_np      = wu_resp_np.sum(0) + _NK_EPS
        wu_means_np   = _sk_mean_update(wu_resp_np, ds0["X_np"], wu_nk_np)

        wu_resp_t_v0  = wu_log_resp_v0.exp()
        wu_nk_t_v0    = wu_resp_t_v0.sum(0) + _NK_EPS
        wu_means_t_v0 = _torch_mean_update(wu_resp_t_v0, ds0["X_t"], wu_nk_t_v0)

        wu_resp_t_v1  = wu_log_resp_v1.exp()
        wu_nk_t_v1    = wu_resp_t_v1.sum(0) + _NK_EPS
        wu_means_t_v1 = _torch_mean_update(wu_resp_t_v1, ds0["X_t"], wu_nk_t_v1)

        _estimate_log_gaussian_prob(ds0["X_np"], ds0["means_np"], ds0["prec_chol_sk"], COV_TYPE)
        _v0_ref._estimate_log_gaussian_prob_full_precchol(ds0["X_t"], ds0["means_t"], ds0["prec_chol_t"])
        _v1._estimate_log_gaussian_prob_full_precchol(ds0["X_t"], ds0["means_t"], ds0["prec_chol_t"])

        wu_new_cov_sk = _sk_cov_update(ds0["X_np"], wu_resp_np, wu_nk_np, wu_means_np, K, D, reg_covar=1e-4)
        wu_new_cov_v0 = _v0_cov_update(ds0["X_t"], wu_resp_t_v0, wu_nk_t_v0, wu_means_t_v0, K, D, reg_covar=1e-4)
        wu_new_cov_v1 = _v1_cov_update(ds0["X_t"], wu_resp_t_v1, wu_nk_t_v1, wu_means_t_v1, K, D, reg_covar=1e-4)

        _compute_precision_cholesky(wu_new_cov_sk, COV_TYPE)
        _v0_ref._compute_precisions_cholesky(wu_new_cov_v0, COV_TYPE)
        _v1._compute_precisions_cholesky(wu_new_cov_v1, COV_TYPE)

        # --- Timing lists ---
        sk_e_times, v0_e_times, v1_e_times             = [], [], []
        sk_m_times, v0_m_times, v1_m_times             = [], [], []
        sk_mahal_times, v0_mahal_times, v1_mahal_times = [], [], []
        sk_mean_times, v0_mean_times, v1_mean_times     = [], [], []
        sk_cov_times, v0_cov_times, v1_cov_times       = [], [], []
        sk_chol_times, v0_chol_times, v1_chol_times     = [], [], []

        for run_idx, ds in enumerate(datasets, start=1):
            print(
                f"[{i+1}/{len(D_VALUES)}] D={D:,} — run {run_idx}/{N_RUNS}",
                file=sys.stderr,
                flush=True,
            )
            X_t  = ds["X_t"]
            X_np = ds["X_np"]

            # Reuse the single model instance; just overwrite the parameters.
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
            # NOTE: sklearn's _m_step internally calls _compute_precision_cholesky, so
            # sk_m includes cholesky time while v0_m / v1_m do not (their _maximization_step
            # returns raw covariances without factorizing). M-step totals are not directly
            # comparable across implementations; use the sub-timers for a fair comparison.
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

            # --- Pre-compute intermediates (untimed).
            # Each implementation uses its own log_resp so sub-timer inputs are symmetric.
            resp_np   = np.exp(log_resp_sk)
            nk_np     = resp_np.sum(0) + _NK_EPS
            means_np  = _sk_mean_update(resp_np, X_np, nk_np)

            resp_t_v0  = log_resp_v0.exp()
            nk_t_v0    = resp_t_v0.sum(0) + _NK_EPS
            means_t_v0 = _torch_mean_update(resp_t_v0, X_t, nk_t_v0)

            resp_t_v1  = log_resp_v1.exp()
            nk_t_v1    = resp_t_v1.sum(0) + _NK_EPS
            means_t_v1 = _torch_mean_update(resp_t_v1, X_t, nk_t_v1)

            # --- Log-prob computation (E-step; times Mahalanobis + log-det + const) ---
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
            _torch_mean_update(resp_t_v0, X_t, nk_t_v0)
            v0_mean_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _torch_mean_update(resp_t_v1, X_t, nk_t_v1)
            v1_mean_times.append((time.perf_counter() - t0) * 1e3)

            # --- Covariance update (M-step); output reused as cholesky input ---
            t0 = time.perf_counter()
            new_cov_sk = _sk_cov_update(X_np, resp_np, nk_np, means_np, K, D, reg_covar=1e-4)
            sk_cov_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            new_cov_v0 = _v0_cov_update(X_t, resp_t_v0, nk_t_v0, means_t_v0, K, D, reg_covar=1e-4)
            v0_cov_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            new_cov_v1 = _v1_cov_update(X_t, resp_t_v1, nk_t_v1, means_t_v1, K, D, reg_covar=1e-4)
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

        print(f"[{i+1}/{len(D_VALUES)}] D={D:,} — done.", file=sys.stderr, flush=True)

        row = {"N": N, "D": D, "K": K}
        for prefix, lst in [
            ("sk_e",     sk_e_times),     ("v0_e",     v0_e_times),     ("v1_e",     v1_e_times),
            ("sk_m",     sk_m_times),     ("v0_m",     v0_m_times),     ("v1_m",     v1_m_times),
            ("sk_mahal", sk_mahal_times), ("v0_mahal", v0_mahal_times), ("v1_mahal", v1_mahal_times),
            ("sk_mean",  sk_mean_times),  ("v0_mean",  v0_mean_times),  ("v1_mean",  v1_mean_times),
            ("sk_cov",   sk_cov_times),   ("v0_cov",   v0_cov_times),   ("v1_cov",   v1_cov_times),
            ("sk_chol",  sk_chol_times),  ("v0_chol",  v0_chol_times),  ("v1_chol",  v1_chol_times),
        ]:
            row[f"{prefix}_mean_ms"], row[f"{prefix}_std_ms"] = _agg(lst)
        all_rows.append(row)

    _print_table(all_rows)

    df = pd.DataFrame(all_rows)
    output_path = os.path.join(os.path.dirname(__file__), "scaling_d_v1_v0.csv")
    df.to_csv(output_path, index=False)
    print(df.to_csv(index=False), end="")
    print(f"CSV saved to: {output_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
