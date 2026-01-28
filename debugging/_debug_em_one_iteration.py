import numpy as np

# If your file is named differently, change this import.
from sklearn_mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _estimate_log_gaussian_prob,
    _estimate_gaussian_parameters,
)

def logsumexp(a, axis=None, keepdims=False):
    """Minimal logsumexp to avoid extra deps."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)

def pretty(name, arr):
    print(f"\n{name}:")
    print(arr)
    print(f"shape={arr.shape}, dtype={arr.dtype}")

def main():
    np.set_printoptions(precision=6, suppress=True)

    # -----------------------------
    # 1) Hard-coded tiny dataset (2D)
    # -----------------------------
    X = np.array([
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-2.0, -2.0],
        [ 2.0,  1.0],
        [ 1.0,  2.0],
        [ 2.0,  2.0],
    ], dtype=np.float64)

    n_samples, n_features = X.shape
    K = 2
    cov_type = "full"
    reg_covar = 1e-6

    # -----------------------------
    # 2) Hard-coded starting params
    # -----------------------------
    weights = np.array([0.5, 0.5], dtype=np.float64)
    means = np.array([
        [-1.5, -1.5],
        [ 1.5,  1.5],
    ], dtype=np.float64)

    covariances = np.array([
        [[0.5, 0.0],
         [0.0, 0.5]],
        [[0.5, 0.0],
         [0.0, 0.5]],
    ], dtype=np.float64)

    pretty("X", X)
    pretty("weights (pi)", weights)
    pretty("means (mu)", means)
    pretty("covariances (Sigma)", covariances)

    # -----------------------------
    # 3) E-step pieces
    #    log p(x|k) and responsibilities
    # -----------------------------
    precisions_chol = _compute_precision_cholesky(covariances, cov_type, xp=np)  # :contentReference[oaicite:2]{index=2}
    pretty("precisions_cholesky", precisions_chol)

    log_prob = _estimate_log_gaussian_prob(X, means, precisions_chol, cov_type, xp=np)  # :contentReference[oaicite:3]{index=3}
    pretty("log_prob = log N(x|mu_k,Sigma_k)", log_prob)

    log_weights = np.log(weights)
    pretty("log_weights = log pi_k", log_weights)

    # unnormalized log responsibilities: log r~(n,k) = log pi_k + log N(x_n|...)
    log_resp_unnorm = log_prob + log_weights
    pretty("log_resp_unnorm = log pi_k + log_prob", log_resp_unnorm)

    # normalize across k:
    log_norm = logsumexp(log_resp_unnorm, axis=1, keepdims=True)
    log_resp = log_resp_unnorm - log_norm
    resp = np.exp(log_resp)

    pretty("log_norm = log sum_k exp(log_resp_unnorm)", log_norm.squeeze())
    pretty("log_resp (normalized)", log_resp)
    pretty("resp = responsibilities", resp)

    print("\nSanity check: each row of resp should sum to 1:")
    print(resp.sum(axis=1))

    # -----------------------------
    # 4) M-step (from responsibilities)
    # -----------------------------
    nk, new_means, new_covs = _estimate_gaussian_parameters(
        X, resp, reg_covar=reg_covar, covariance_type=cov_type, xp=np
    )  # :contentReference[oaicite:4]{index=4}

    new_weights = nk / nk.sum()

    pretty("nk = sum_n r_nk", nk)
    pretty("new_weights = nk / sum(nk)", new_weights)
    pretty("new_means", new_means)
    pretty("new_covariances", new_covs)

if __name__ == "__main__":
    main()
