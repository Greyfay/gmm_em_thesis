# Sklearn-Based Initialization for PyTorch GMM

## Summary

Added two new initialization options to the PyTorch GMM implementation:
- `scikit_kmeans`: Uses sklearn's KMeans for initialization
- `scikit_random`: Uses sklearn's random uniform initialization

These options help align the starting values between sklearn and PyTorch implementations, making it easier to debug and compare results.

## Changes Made

### 1. Added Dependencies
- Added `numpy` import
- Added `sklearn.cluster.KMeans` and `kmeans_plusplus` imports

### 2. New Helper Function
Added `_initialize_from_sklearn()` function that:
- Converts PyTorch tensors to numpy arrays
- Runs sklearn's initialization (KMeans or random)
- Converts results back to PyTorch format
- Returns log-responsibilities for the EM algorithm

### 3. Updated Initialization Logic
- Updated `init_params` validation to accept `'scikit_kmeans'` and `'scikit_random'`
- Added handling for these new options in the `_initialize()` method
- Updated module docstring to document the new options

## Usage Examples

### Basic Usage

```python
from implementation._torch_gmm_em import TorchGaussianMixture

# Use sklearn's KMeans initialization
gmm = TorchGaussianMixture(
    n_components=5,
    covariance_type="diag",
    init_params="scikit_kmeans"  # <-- Use sklearn init
)
gmm.fit(X)

# Use sklearn's random initialization
gmm = TorchGaussianMixture(
    n_components=5,
    covariance_type="full",
    init_params="scikit_random"  # <-- Use sklearn init
)
gmm.fit(X)
```

### Comparing with Sklearn

```python
import torch
import numpy as np
from sklearn.mixture import GaussianMixture as SklearnGMM
from implementation._torch_gmm_em import TorchGaussianMixture

# Generate data
X_np = np.random.randn(100, 10)
X_torch = torch.from_numpy(X_np).float()

# Fit sklearn GMM
sklearn_gmm = SklearnGMM(n_components=3, init_params='kmeans', n_init=1)
sklearn_gmm.fit(X_np)

# Fit PyTorch GMM with sklearn initialization
torch_gmm = TorchGaussianMixture(
    n_components=3,
    init_params='scikit_kmeans',
    n_init=1
)
torch_gmm.fit(X_torch)

# Now both should start from similar initial values!
print(f"Sklearn: {sklearn_gmm.lower_bound_:.4f}")
print(f"PyTorch: {torch_gmm.lower_bound_:.4f}")
```

## Available Initialization Options

| Option | Description | Source |
|--------|-------------|--------|
| `'kmeans'` | PyTorch k-means++ + Lloyd iterations | PyTorch |
| `'k-means++'` | PyTorch k-means++ seeding only | PyTorch |
| `'random'` | Random uniform responsibilities | PyTorch |
| `'random_from_data'` | Random data points as means | PyTorch |
| `'scikit_kmeans'` | **Sklearn's KMeans** | **Sklearn** |
| `'scikit_random'` | **Sklearn's random uniform** | **Sklearn** |

## Benefits

1. **Exact Alignment**: Use sklearn's initialization to ensure starting points match exactly
2. **Debugging**: Helps identify whether differences are due to initialization or EM algorithm
3. **Reproducibility**: Makes it easier to compare and validate against sklearn results
4. **Flexibility**: Still have access to native PyTorch initialization when desired

## Files Modified

- `/home/salam/thesis/implementation/_torch_gmm_em.py`

## Test Files Created

- `/home/salam/thesis/tests/_test_scikit_init.py` - Basic functionality test
- `/home/salam/thesis/tests/_comparison_sklearn_init.py` - Comparison between init methods
- `/home/salam/thesis/examples/sklearn_init_example.py` - Comprehensive usage examples

## Notes

- The sklearn initialization converts data to numpy, runs sklearn functions, then converts back to PyTorch
- This adds a small CPU overhead but is only done once during initialization
- The rest of the EM algorithm remains in pure PyTorch
- Works with all covariance types: 'diag', 'full', 'tied', 'spherical'
