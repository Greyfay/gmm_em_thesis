"""Observe EM parameter updates iteration-by-iteration."""

import os
import sys
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation._torch_gmm_em import TorchGaussianMixture

def observe_sklearn():
    """Observe scikit-learn EM iterations."""
    print("="*70)
    print("SCIKIT-LEARN GMM - Observing EM Iterations")
    print("="*70)
    
    # Moderate-sized example
    np.random.seed(42)
    n_samples = 5000
    n_dims = 20
    n_components = 3
    
    print(f"\nConfiguration: N={n_samples}, D={n_dims}, K={n_components}")
    print(f"Covariance: full, max_iter=20, init=kmeans\n")
    
    X = np.random.randn(n_samples, n_dims).astype(np.float32)
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=20,
        n_init=1,
        init_params="kmeans",
        tol=1e-3,
        random_state=42,
        verbose=2,  # Print iteration info
        verbose_interval=1,  # Print every iteration
    )
    
    gmm.fit(X)
    
    print(f"\n{'='*70}")
    print(f"Final Results:")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Iterations: {gmm.n_iter_}")
    print(f"  Final log-likelihood: {gmm.lower_bound_:.6f}")
    print(f"\n  Final Weights: {gmm.weights_}")
    print(f"\n  Final Means (first 5 dims of each component):")
    for k in range(n_components):
        print(f"    Component {k}: {gmm.means_[k][:5]}")
    print("="*70 + "\n")


def observe_torch():
    """Observe PyTorch EM iterations."""
    print("="*70)
    print("PYTORCH GMM - Observing EM Iterations")
    print("="*70)
    
    # Same configuration as scikit
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    n_samples = 5000
    n_dims = 20
    n_components = 3
    
    print(f"\nConfiguration: N={n_samples}, D={n_dims}, K={n_components}")
    print(f"Covariance: full, max_iter=20, init=kmeans\n")
    
    X = torch.randn(n_samples, n_dims, device="cuda", dtype=torch.float32)
    
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
    
    print("Running EM algorithm...\n")
    
    # Fit the model
    gmm.fit(X)
    
    print(f"{'='*70}")
    print(f"Final Results:")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Iterations: {gmm.n_iter_}")
    
    if hasattr(gmm, 'lower_bound_'):
        print(f"  Final log-likelihood: {gmm.lower_bound_:.6f}")
    
    print(f"\n  Final Weights: {gmm.weights_.cpu().numpy()}")
    print(f"\n  Final Means (first 5 dims of each component):")
    for k in range(n_components):
        print(f"    Component {k}: {gmm.means_[k][:5].cpu().numpy()}")
    
    print(f"\n  Note: For detailed iteration-by-iteration output,")
    print(f"        check if your TorchGaussianMixture supports verbose mode")
    print(f"        or add print statements inside the _em_step() method.")
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
