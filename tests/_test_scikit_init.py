"""Test script to verify scikit_random and scikit_kmeans initialization options."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from implementation._torch_gmm_em import TorchGaussianMixture

# Generate some test data
torch.manual_seed(42)
np.random.seed(42)

N, D, K = 100, 5, 3
X = torch.randn(N, D)

print("Testing scikit_random initialization...")
gmm_scikit_random = TorchGaussianMixture(
    n_components=K,
    covariance_type="diag",
    init_params="scikit_random",
    max_iter=10
)
gmm_scikit_random.fit(X)
print(f"✓ scikit_random initialization succeeded")
print(f"  Converged: {gmm_scikit_random.converged_}")
print(f"  Iterations: {gmm_scikit_random.n_iter_}")
print(f"  Lower bound: {gmm_scikit_random.lower_bound_:.4f}")
print()

print("Testing scikit_kmeans initialization...")
gmm_scikit_kmeans = TorchGaussianMixture(
    n_components=K,
    covariance_type="diag",
    init_params="scikit_kmeans",
    max_iter=10
)
gmm_scikit_kmeans.fit(X)
print(f"✓ scikit_kmeans initialization succeeded")
print(f"  Converged: {gmm_scikit_kmeans.converged_}")
print(f"  Iterations: {gmm_scikit_kmeans.n_iter_}")
print(f"  Lower bound: {gmm_scikit_kmeans.lower_bound_:.4f}")
print()

# Compare with regular kmeans and random
print("Comparing with regular PyTorch kmeans...")
gmm_kmeans = TorchGaussianMixture(
    n_components=K,
    covariance_type="diag",
    init_params="kmeans",
    max_iter=10
)
gmm_kmeans.fit(X)
print(f"  PyTorch kmeans lower bound: {gmm_kmeans.lower_bound_:.4f}")
print()

print("Comparing with regular PyTorch random...")
gmm_random = TorchGaussianMixture(
    n_components=K,
    covariance_type="diag",
    init_params="random",
    max_iter=10
)
gmm_random.fit(X)
print(f"  PyTorch random lower bound: {gmm_random.lower_bound_:.4f}")
print()

print("All initialization methods work successfully!")
