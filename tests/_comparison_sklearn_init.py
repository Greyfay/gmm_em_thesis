"""Comparison script showing sklearn vs PyTorch initialization alignment."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.mixture import GaussianMixture as SklearnGMM
from implementation._torch_gmm_em import TorchGaussianMixture

# Generate test data
np.random.seed(42)
torch.manual_seed(42)

N, D, K = 200, 10, 5
X_np = np.random.randn(N, D)
X_torch = torch.from_numpy(X_np).float()

print("="*70)
print("Comparing Sklearn vs PyTorch GMM with Different Initializations")
print("="*70)
print()

# Fit sklearn GMM with kmeans initialization
print("1. Fitting Sklearn GMM with kmeans initialization...")
sklearn_gmm = SklearnGMM(n_components=K, covariance_type='diag', 
                          init_params='kmeans', n_init=1, max_iter=20)
sklearn_gmm.fit(X_np)
print(f"   Sklearn lower bound: {sklearn_gmm.lower_bound_:.6f}")
print(f"   Sklearn converged: {sklearn_gmm.converged_}")
print(f"   Sklearn n_iter: {sklearn_gmm.n_iter_}")
print()

# Fit PyTorch GMM with PyTorch kmeans (original)
print("2. Fitting PyTorch GMM with PyTorch kmeans initialization...")
torch_gmm_pytorch_init = TorchGaussianMixture(
    n_components=K, covariance_type='diag',
    init_params='kmeans', n_init=1, max_iter=20
)
torch_gmm_pytorch_init.fit(X_torch)
print(f"   PyTorch lower bound: {torch_gmm_pytorch_init.lower_bound_:.6f}")
print(f"   PyTorch converged: {torch_gmm_pytorch_init.converged_}")
print(f"   PyTorch n_iter: {torch_gmm_pytorch_init.n_iter_}")
print(f"   Difference: {abs(sklearn_gmm.lower_bound_ - torch_gmm_pytorch_init.lower_bound_):.6f}")
print()

# Fit PyTorch GMM with sklearn kmeans (NEW!)
print("3. Fitting PyTorch GMM with SKLEARN kmeans initialization...")
torch_gmm_sklearn_init = TorchGaussianMixture(
    n_components=K, covariance_type='diag',
    init_params='scikit_kmeans', n_init=1, max_iter=20
)
torch_gmm_sklearn_init.fit(X_torch)
print(f"   PyTorch lower bound: {torch_gmm_sklearn_init.lower_bound_:.6f}")
print(f"   PyTorch converged: {torch_gmm_sklearn_init.converged_}")
print(f"   PyTorch n_iter: {torch_gmm_sklearn_init.n_iter_}")
print(f"   Difference: {abs(sklearn_gmm.lower_bound_ - torch_gmm_sklearn_init.lower_bound_):.6f}")
print()

print("="*70)
print("Summary")
print("="*70)
print(f"Sklearn lower bound:                 {sklearn_gmm.lower_bound_:.6f}")
print(f"PyTorch (PyTorch init) lower bound:  {torch_gmm_pytorch_init.lower_bound_:.6f}")
print(f"PyTorch (Sklearn init) lower bound:  {torch_gmm_sklearn_init.lower_bound_:.6f}")
print()
print(f"Difference (PyTorch init vs Sklearn): {abs(sklearn_gmm.lower_bound_ - torch_gmm_pytorch_init.lower_bound_):.6f}")
print(f"Difference (Sklearn init vs Sklearn): {abs(sklearn_gmm.lower_bound_ - torch_gmm_sklearn_init.lower_bound_):.6f}")
print()

if abs(sklearn_gmm.lower_bound_ - torch_gmm_sklearn_init.lower_bound_) < abs(sklearn_gmm.lower_bound_ - torch_gmm_pytorch_init.lower_bound_):
    print("✓ Using scikit_kmeans initialization brings PyTorch closer to Sklearn results!")
else:
    print("ℹ Results may vary due to random seeds and EM convergence paths.")
