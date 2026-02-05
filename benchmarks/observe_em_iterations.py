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
    """Observe PyTorch EM iterations with custom verbose output."""
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
        verbose=2,  # If your implementation supports it
    )
    
    # Monkey-patch the fit method to print iteration details
    original_fit = gmm.fit
    
    def verbose_fit(X):
        # Initialize
        original_fit(X)
        return gmm
    
    # Alternative: manually run EM loop with prints
    print("Running EM algorithm with verbose output...\n")
    
    # Initialize using kmeans
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=1)
    labels = kmeans.fit_predict(X.cpu().numpy())
    
    # Initialize parameters
    gmm.weights_ = torch.ones(n_components, device=X.device, dtype=X.dtype) / n_components
    gmm.means_ = torch.from_numpy(kmeans.cluster_centers_).to(X.device, dtype=X.dtype)
    
    # Initialize covariances as identity scaled by data variance
    data_var = torch.var(X, dim=0).mean()
    gmm.covariances_ = torch.eye(n_dims, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(n_components, 1, 1) * data_var
    
    print(f"Initialization:")
    print(f"  Weights: {gmm.weights_.cpu().numpy()}")
    print(f"  Means (first 5 dims):")
    for k in range(n_components):
        print(f"    Component {k}: {gmm.means_[k][:5].cpu().numpy()}")
    print()
    
    # Run EM manually to show iteration details
    prev_log_likelihood = -np.inf
    
    for iteration in range(20):
        # E-step: compute responsibilities
        log_resp = gmm._estimate_log_resp(X)
        log_likelihood = torch.logsumexp(log_resp, dim=1).mean().item()
        
        # Check convergence
        delta = log_likelihood - prev_log_likelihood
        print(f"Iteration {iteration+1:2d}: log-likelihood = {log_likelihood:12.6f} (delta = {delta:+.6e})")
        
        if iteration > 0 and abs(delta) < 1e-3:
            print(f"  -> Converged!")
            break
        
        # M-step: update parameters
        resp = torch.exp(log_resp)
        gmm._m_step(X, resp)
        
        # Print updated parameters every few iterations
        if iteration % 5 == 4 or iteration == 0:
            print(f"    Weights: {gmm.weights_.cpu().numpy()}")
            print(f"    Means (component 0, first 5 dims): {gmm.means_[0][:5].cpu().numpy()}")
        
        prev_log_likelihood = log_likelihood
        print()
    
    print(f"{'='*70}")
    print(f"Final Results:")
    print(f"  Iterations: {iteration+1}")
    print(f"  Final log-likelihood: {log_likelihood:.6f}")
    print(f"\n  Final Weights: {gmm.weights_.cpu().numpy()}")
    print(f"\n  Final Means (first 5 dims of each component):")
    for k in range(n_components):
        print(f"    Component {k}: {gmm.means_[k][:5].cpu().numpy()}")
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
