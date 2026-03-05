#!/usr/bin/env python3
"""E-step runtime benchmark: sklearn vs _v0_ref vs _v1.
Fixed: K=5, D=50. N in [1e4, 1e5, 1e6, 1e7].
10 timed runs per configuration; outputs CSV only.
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
from implementation import _v0_ref
from implementation import _v1

K = 5
D = 50
N_VALUES = [10_000, 100_000, 1_000_000, 10_000_000]
N_RUNS = 10
COV_TYPE = "full"


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
            "X_np": X_np,
            "X_t": torch.from_numpy(X_np),
            "means_t": torch.from_numpy(means_np),
            "weights_t": torch.from_numpy(weights_np),
            "prec_chol_t": prec_chol_t,
        })
    return datasets


def main():
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    rows = []

    for i, N in enumerate(N_VALUES):
        print(f"[{i+1}/{len(N_VALUES)}] N={N:,} — generating datasets...", file=sys.stderr, flush=True)
        datasets = make_datasets(N, rng)
        print(f"[{i+1}/{len(N_VALUES)}] N={N:,} — running benchmarks...", file=sys.stderr, flush=True)

        # One warmup iteration on the first dataset (untimed)
        ds0 = datasets[0]
        warmup_model = GaussianMixture(n_components=K, covariance_type=COV_TYPE)
        warmup_model.weights_ = ds0["weights_np"]
        warmup_model.means_ = ds0["means_np"]
        warmup_model.precisions_cholesky_ = ds0["prec_chol_sk"]
        warmup_model._e_step(ds0["X_np"])
        _v0_ref._expectation_step_precchol(
            ds0["X_t"], ds0["means_t"], ds0["prec_chol_t"], ds0["weights_t"], COV_TYPE
        )
        _v1._expectation_step_precchol(
            ds0["X_t"], ds0["means_t"], ds0["prec_chol_t"], ds0["weights_t"], COV_TYPE
        )

        sk_times, v0_times, v1_times = [], [], []

        for ds in datasets:
            # Build sklearn model before timing so only _e_step is measured
            model = GaussianMixture(n_components=K, covariance_type=COV_TYPE)
            model.weights_ = ds["weights_np"]
            model.means_ = ds["means_np"]
            model.precisions_cholesky_ = ds["prec_chol_sk"]

            t0 = time.perf_counter()
            model._e_step(ds["X_np"])
            sk_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _v0_ref._expectation_step_precchol(
                ds["X_t"], ds["means_t"], ds["prec_chol_t"], ds["weights_t"], COV_TYPE
            )
            v0_times.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _v1._expectation_step_precchol(
                ds["X_t"], ds["means_t"], ds["prec_chol_t"], ds["weights_t"], COV_TYPE
            )
            v1_times.append((time.perf_counter() - t0) * 1e3)

        print(f"[{i+1}/{len(N_VALUES)}] N={N:,} — done.", file=sys.stderr, flush=True)
        rows.append({
            "N": N,
            "D": D,
            "K": K,
            "sklearn_mean_ms": round(float(np.mean(sk_times)), 4),
            "sklearn_std_ms": round(float(np.std(sk_times)), 4),
            "v0_ref_mean_ms": round(float(np.mean(v0_times)), 4),
            "v0_ref_std_ms": round(float(np.std(v0_times)), 4),
            "v1_mean_ms": round(float(np.mean(v1_times)), 4),
            "v1_std_ms": round(float(np.std(v1_times)), 4),
        })

    df = pd.DataFrame(rows)
    output_path = os.path.join(os.path.dirname(__file__), "estep_benchmark.csv")
    df.to_csv(output_path, index=False)
    print(df.to_csv(index=False), end="")
    print(f"CSV saved to: {output_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
