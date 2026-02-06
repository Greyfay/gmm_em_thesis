import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from implementation._torch_gmm_em import TorchGaussianMixture
import torch

X = torch.randn(50, 5)
print('Testing all init methods...')
methods = ['kmeans', 'k-means++', 'random', 'random_from_data', 'scikit_kmeans', 'scikit_random']
for m in methods:
    TorchGaussianMixture(n_components=3, init_params=m, max_iter=5).fit(X)
print('✓ All initialization methods work correctly!')
