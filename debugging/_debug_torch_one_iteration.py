import numpy as np

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from implementation._v1 import (
    _expectation_step,
    _maximization_step,
    pretty
)

def main():


    np.set_printoptions(precision=6, suppress=True)

    # -----------------------------
    # 1) Hard-coded tiny dataset (2D)
    # -----------------------------
    X = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=np.float64)

    n_samples, n_features = X.shape
    K = 2
    cov_type = "diag"
    reg_covar = 1e-6

    # -----------------------------
    # 2) Hard-coded starting params
    # -----------------------------
    weights = np.array([0.5, 0.5], dtype=np.float64)
    means = np.array([
        [0.45, 0.15],
        [0.55,  0.85],
    ], dtype=np.float64)

    covariances = np.array([
        [[0.2475, -0.0675],
         [-0.0675, 0.1275]],
        [[0.2475, -0.0675],
         [-0.0675, 0.1275]],
    ], dtype=np.float64)

    # Convert to torch
    X = torch.from_numpy(X)
    means = torch.from_numpy(means)
    covariances = torch.from_numpy(covariances)
    weights = torch.from_numpy(weights)

    pretty("X", X)
    pretty("weights (pi)", weights)
    pretty("means (mu)", means)
    pretty("covariances (Sigma)", covariances)

    # -----------------------------
    # 3) E-step

    resp = _expectation_step(X, means, covariances, weights)
    
    _maximization_step(X, means, covariances, weights, resp)

if __name__ == "__main__":
    main()