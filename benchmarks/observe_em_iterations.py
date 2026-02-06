"""Observe EM parameter updates iteration-by-iteration."""

import os
import sys
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation._torch_gmm_em import TorchGaussianMixture

def observe_sklearn():
    """Observe scikit-learn EM iterations with fixed initialization."""
    print("="*70)
    print("SCIKIT-LEARN GMM - Observing EM Iterations (HARDCODED DATA)")
    print("="*70)
    
    # Small hardcoded example
    n_samples = 30
    n_dims = 2
    n_components = 2
    
    # Hardcoded data: 30 samples, 2 dimensions
    # First 15 samples cluster around (0, 0), next 15 around (5, 5)
    X = np.array([
        [0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.3], [-0.1, -0.3],
        [0.2, 0.1], [-0.4, 0.2], [0.3, 0.5], [-0.2, -0.1], [0.1, -0.2],
        [0.4, -0.3], [-0.3, 0.1], [0.2, 0.4], [-0.1, 0.2], [0.3, 0.0],
        [5.1, 5.2], [4.8, 5.1], [5.3, 4.9], [4.9, 5.3], [5.2, 4.8],
        [5.0, 5.0], [4.7, 5.2], [5.4, 4.7], [4.8, 4.9], [5.1, 5.1],
        [5.3, 5.3], [4.9, 4.8], [5.2, 5.1], [4.8, 5.3], [5.0, 4.9]
    ], dtype=np.float32)
    
    print(f"\nConfiguration: N={n_samples}, D={n_dims}, K={n_components}")
    print(f"Covariance: full, max_iter=20, init=kmeans")
    print(f"\nHardcoded data (all {n_samples} samples):\n{X}")
    print()
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=20,
        n_init=1,
        init_params="kmeans",
        tol=1e-3,
        random_state=99,  # Fixed seed for reproducibility
        verbose=2,
        verbose_interval=1,
    )
    
    gmm.fit(X)
    
    print(f"\n{'='*70}")
    print(f"Final Results:")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Iterations: {gmm.n_iter_}")
    print(f"  Final log-likelihood: {gmm.lower_bound_:.6f}")
    print(f"\n  Final Weights: {gmm.weights_}")
    print(f"\n  Final Means (all dims):")
    for k in range(n_components):
        print(f"    Component {k}: {gmm.means_[k]}")
    print("="*70 + "\n")


def observe_torch():
    """Observe PyTorch EM iterations with fixed initialization."""
    print("="*70)
    print("PYTORCH GMM - Observing EM Iterations (HARDCODED DATA)")
    print("="*70)
    
    # Same configuration as scikit
    n_samples = 30
    n_dims = 2
    n_components = 2
    
    # Hardcoded data: same as sklearn version
    X_np = np.array([
        [0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.3], [-0.1, -0.3],
        [0.2, 0.1], [-0.4, 0.2], [0.3, 0.5], [-0.2, -0.1], [0.1, -0.2],
        [0.4, -0.3], [-0.3, 0.1], [0.2, 0.4], [-0.1, 0.2], [0.3, 0.0],
        [5.1, 5.2], [4.8, 5.1], [5.3, 4.9], [4.9, 5.3], [5.2, 4.8],
        [5.0, 5.0], [4.7, 5.2], [5.4, 4.7], [4.8, 4.9], [5.1, 5.1],
        [5.3, 5.3], [4.9, 4.8], [5.2, 5.1], [4.8, 5.3], [5.0, 4.9]
    ], dtype=np.float32)
    X = torch.from_numpy(X_np).to(device="cuda", dtype=torch.float32)
    
    print(f"\nConfiguration: N={n_samples}, D={n_dims}, K={n_components}")
    print(f"Covariance: full, max_iter=20, init=kmeans")
    print(f"\nHardcoded data (all {n_samples} samples):\n{X.cpu().numpy()}")
    print()
    
    gmm = TorchGaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=20,
        n_init=1,
        init_params="kmeans",
        tol=1e-3,
        device="cuda",
        dtype=torch.float32,
    )
    
    # Manually set initialization to match sklearn
    torch.manual_seed(99)
    torch.cuda.manual_seed(99)
    
    print("Running EM algorithm...\n")
    
    # Fit the model
    gmm.fit(X)
    
    print(f"\n{'='*70}")
    print(f"Final Results:")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Iterations: {gmm.n_iter_}")
    
    if hasattr(gmm, 'lower_bound_'):
        print(f"  Final log-likelihood: {gmm.lower_bound_:.6f}")
    
    print(f"\n  Final Weights: {gmm.weights_.cpu().numpy()}")
    print(f"\n  Final Means (all dims):")
    for k in range(n_components):
        print(f"    Component {k}: {gmm.means_[k].cpu().numpy()}")
    
    print("\n  Comparison with scikit-learn:")
    print("  If the final means are similar, the implementations are equivalent.")
    print("  If they differ significantly, there may be algorithmic differences.")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run scikit-learn observation
    observe_sklearn()
    
    print("\n" + "="*70)
    print("="*70)
    print("\n")
    
    # Run PyTorch observation
    if torch.cuda.is_available():
        observe_torch()
    else:
        print("CUDA not available - skipping PyTorch demonstration")
