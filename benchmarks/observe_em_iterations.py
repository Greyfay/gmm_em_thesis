"""Observe EM parameter updates iteration-by-iteration."""

import os
import sys
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation._torch_gmm_em import TorchGaussianMixture


def reorder_gmm_components(weights, means, covariances=None):
    """Reorder GMM components by sorting on the first dimension of means.
    
    Args:
        weights: Component weights (numpy array or torch tensor)
        means: Component means, shape (n_components, n_dims)
        covariances: Optional covariances (for future use)
    
    Returns:
        Reordered weights, means, and covariances
    """
    # Get sorting indices based on first dimension of means
    if isinstance(means, torch.Tensor):
        sort_idx = torch.argsort(means[:, 0])
        sorted_weights = weights[sort_idx]
        sorted_means = means[sort_idx]
        sorted_covariances = covariances[sort_idx] if covariances is not None else None
    else:
        sort_idx = np.argsort(means[:, 0])
        sorted_weights = weights[sort_idx]
        sorted_means = means[sort_idx]
        sorted_covariances = covariances[sort_idx] if covariances is not None else None
    
    return sorted_weights, sorted_means, sorted_covariances


def observe_sklearn():
    """Observe scikit-learn EM iterations with fixed initialization."""
    print("="*70)
    print("SCIKIT-LEARN GMM - Observing EM Iterations (RANDOM DATA)")
    print("="*70)
    
    # Small random example with fixed seed
    np.random.seed(42)
    n_samples = 5000
    n_dims = 20
    n_components = 2
    
    # Generate random data with fixed seed
    X = np.random.randn(n_samples, n_dims).astype(np.float32)
    
    print(f"\nConfiguration: N={n_samples}, D={n_dims}, K={n_components}")
    print(f"Covariance: full, max_iter=100, init=random (fixed seed)")
    print(f"\nRandom data (first 5 samples):\n{X[:5]}")
    print()
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=100,
        n_init=1,
        init_params="random",  # Use random init with fixed seed
        tol=1e-3,
        random_state=99,  # Fixed seed for reproducibility
        verbose=2,
        verbose_interval=1,
    )
    
    gmm.fit(X)
    
    # Reorder components for consistent comparison
    sorted_weights, sorted_means, _ = reorder_gmm_components(
        gmm.weights_, gmm.means_, gmm.covariances_
    )
    
    print(f"\n{'='*70}")
    print(f"Final Results (components sorted by mean[0]):")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Iterations: {gmm.n_iter_}")
    print(f"  Final log-likelihood: {gmm.lower_bound_:.6f}")
    print(f"\n  Final Weights: {sorted_weights}")
    print(f"\n  Final Means (all dims):")
    for k in range(n_components):
        print(f"    Component {k}: {sorted_means[k]}")
    print("="*70 + "\n")


def observe_torch():
    """Observe PyTorch EM iterations with fixed initialization."""
    print("="*70)
    print("PYTORCH GMM - Observing EM Iterations (RANDOM DATA)")
    print("="*70)
    
    # Same configuration as scikit with same random seed
    np.random.seed(42)
    n_samples = 5000
    n_dims = 20
    n_components = 2
    
    # Generate random data with same seed as sklearn
    X_np = np.random.randn(n_samples, n_dims).astype(np.float32)
    X = torch.from_numpy(X_np).to(device="cuda", dtype=torch.float32)
    
    print(f"\nConfiguration: N={n_samples}, D={n_dims}, K={n_components}")
    print(f"Covariance: full, max_iter=100, init=random (fixed seed)")
    print(f"\nRandom data (first 5 samples):\n{X[:5].cpu().numpy()}")
    print()
    
    gmm = TorchGaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=100,
        n_init=1,
        init_params="random",
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
    
    # Reorder components for consistent comparison
    sorted_weights, sorted_means, _ = reorder_gmm_components(
        gmm.weights_, gmm.means_, gmm.covariances_ if hasattr(gmm, 'covariances_') else None
    )
    
    print(f"\n{'='*70}")
    print(f"Final Results (components sorted by mean[0]):")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Iterations: {gmm.n_iter_}")
    
    if hasattr(gmm, 'lower_bound_'):
        print(f"  Final log-likelihood: {gmm.lower_bound_:.6f}")
    
    print(f"\n  Final Weights: {sorted_weights.cpu().numpy()}")
    print(f"\n  Final Means (all dims):")
    for k in range(n_components):
        print(f"    Component {k}: {sorted_means[k].cpu().numpy()}")
    
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
