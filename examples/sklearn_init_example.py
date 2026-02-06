"""
Example: Using scikit_random and scikit_kmeans initialization options

This demonstrates how to use the new sklearn-based initialization options
to ensure that PyTorch GMM initialization matches sklearn exactly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from implementation._torch_gmm_em import TorchGaussianMixture

# Generate synthetic data
np.random.seed(123)
torch.manual_seed(123)

N, D, K = 500, 20, 8
X = torch.randn(N, D)

print("="*80)
print("PyTorch GMM - Sklearn-Based Initialization Options")
print("="*80)
print()
print(f"Data: {N} samples, {D} dimensions, {K} components")
print()

# Example 1: Using scikit_kmeans initialization
print("Example 1: Using 'scikit_kmeans' initialization")
print("-" * 80)
gmm1 = TorchGaussianMixture(
    n_components=K,
    covariance_type="diag",
    init_params="scikit_kmeans",  # <-- Use sklearn's KMeans
    max_iter=50,
    tol=1e-3
)
gmm1.fit(X)
print(f"Converged: {gmm1.converged_}")
print(f"Iterations: {gmm1.n_iter_}")
print(f"Final log-likelihood: {gmm1.lower_bound_:.4f}")
print(f"Weights shape: {gmm1.weights_.shape}")
print(f"Means shape: {gmm1.means_.shape}")
print(f"Covariances shape: {gmm1.covariances_.shape}")
print()

# Example 2: Using scikit_random initialization
print("Example 2: Using 'scikit_random' initialization")
print("-" * 80)
gmm2 = TorchGaussianMixture(
    n_components=K,
    covariance_type="full",
    init_params="scikit_random",  # <-- Use sklearn's random init
    max_iter=50,
    tol=1e-3
)
gmm2.fit(X)
print(f"Converged: {gmm2.converged_}")
print(f"Iterations: {gmm2.n_iter_}")
print(f"Final log-likelihood: {gmm2.lower_bound_:.4f}")
print(f"Weights shape: {gmm2.weights_.shape}")
print(f"Means shape: {gmm2.means_.shape}")
print(f"Covariances shape: {gmm2.covariances_.shape}")
print()

# Example 3: Comparing all initialization methods
print("Example 3: Comparing all initialization methods")
print("-" * 80)

init_methods = ['kmeans', 'k-means++', 'random', 'random_from_data', 
                'scikit_kmeans', 'scikit_random']

results = {}
for init_method in init_methods:
    gmm = TorchGaussianMixture(
        n_components=K,
        covariance_type="diag",
        init_params=init_method,
        max_iter=30,
        tol=1e-3
    )
    gmm.fit(X)
    results[init_method] = {
        'lower_bound': gmm.lower_bound_,
        'converged': gmm.converged_,
        'n_iter': gmm.n_iter_
    }
    print(f"{init_method:20s}: LL={gmm.lower_bound_:10.4f}, "
          f"converged={str(gmm.converged_):5s}, iter={gmm.n_iter_:3d}")

print()
print("="*80)
print("Key Benefits of scikit_kmeans and scikit_random:")
print("="*80)
print("• They use sklearn's exact initialization logic")
print("• Helps align results between sklearn and PyTorch implementations")
print("• Useful for debugging and comparing implementations")
print("• Can help identify whether differences are due to initialization or EM algorithm")
print()
