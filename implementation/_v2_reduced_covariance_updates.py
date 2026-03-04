"""Gaussian Mixture Model with reduced covariance update frequency.

Variant of v1 where the covariance computation in the M-step is skipped on
iterations that are not multiples of `covariance_update_frequency`. Means and
weights are still updated every iteration; only the expensive covariance
recomputation is throttled.

The goal is to trade convergence rate (more EM iterations needed) against
per-iteration cost (cheaper M-step on skipped iterations) and see whether
total wall-clock time decreases.

Exposes an additional fitted attribute after fit():
  covariance_updates_: int  -- how many times covariance was actually updated
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from implementation._v1 import (
    GMMParams,
    TorchGaussianMixture as _BaseGMM,
    _compute_precisions,
    _expectation_step_precchol,
    _maximization_step,
    _nk_eps,
)


# ---------------------------
# M-step with optional covariance skip
# ---------------------------

def _maximization_step_reduced(
    X: torch.Tensor,
    means: torch.Tensor,
    cov: torch.Tensor,
    weights: torch.Tensor,
    log_resp: torch.Tensor,
    cov_type: str,
    reg_covar: float = 1e-6,
    update_covariance: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M-step that optionally skips the covariance recomputation.

    When update_covariance=True: delegates to v1's full _maximization_step.
    When update_covariance=False: updates only means and weights, returns the
      old covariance unchanged. This avoids the O(N*D^2*K) covariance sum.
    """
    if update_covariance:
        return _maximization_step(X, means, cov, weights, log_resp, cov_type, reg_covar=reg_covar)

    # Means and weights only (no covariance computation)
    resp = log_resp.exp()                                       # (N, K)
    nk = resp.sum(dim=0) + _nk_eps(resp.dtype)                 # (K,)
    new_weights = nk / nk.sum()
    new_means = (resp.T @ X) / nk.unsqueeze(1)                 # (K, D)
    return new_means, cov, new_weights


# ---------------------------
# Model wrapper
# ---------------------------

class TorchGaussianMixture(_BaseGMM):
    """GMM variant that updates the covariance only every N iterations.

    All parameters are identical to the base v1 TorchGaussianMixture except:

    covariance_update_frequency : int, default=1
        Update the covariance matrix every this many EM iterations.
        1 means every iteration (identical to v1).
        2 means every other iteration, etc.

    After fit(), the additional attribute covariance_updates_ records how many
    times the covariance was actually recomputed.
    """

    def __init__(
        self,
        *args,
        covariance_update_frequency: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if covariance_update_frequency < 1:
            raise ValueError(
                f"covariance_update_frequency must be >= 1, got {covariance_update_frequency}"
            )
        self.covariance_update_frequency = covariance_update_frequency
        self.covariance_updates_: int = 0

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "TorchGaussianMixture":
        X = self._to_device_dtype(X)

        best_lower = torch.tensor(float("-inf"), device=X.device, dtype=X.dtype)
        best_params: Optional[GMMParams] = None
        best_n_iter = 0
        best_converged = False
        best_history: List[float] = []
        best_cov_updates = 0

        n_init = 1 if (self.warm_start and self._params is not None) else self.n_init

        for _ in range(n_init):
            p = self._params if (self.warm_start and self._params is not None) else self._initialize(X)

            prev_lower = torch.tensor(float("-inf"), device=X.device, dtype=X.dtype)
            converged = False
            history: List[float] = []
            covariance_updates = 0

            for it in range(self.max_iter):
                lower, log_resp = _expectation_step_precchol(
                    X, p.means, p.prec_chol, p.weights, p.cov_type
                )
                history.append(float(lower.item()))

                update_cov = (it % self.covariance_update_frequency == 0)
                means, cov, weights = _maximization_step_reduced(
                    X, p.means, p.cov, p.weights, log_resp, p.cov_type,
                    reg_covar=self.reg_covar,
                    update_covariance=update_cov,
                )
                if update_cov:
                    covariance_updates += 1

                p = self._make_params(weights, means, cov)

                change = lower - prev_lower
                if torch.abs(change) < self.tol:
                    converged = True
                    break
                prev_lower = lower

            if prev_lower > best_lower:
                best_lower = prev_lower
                best_params = p
                best_n_iter = it + 1
                best_converged = converged
                best_history = history
                best_cov_updates = covariance_updates

        assert best_params is not None

        self._params = best_params
        self.weights_ = best_params.weights
        self.means_ = best_params.means
        self.covariances_ = best_params.cov
        self.precisions_cholesky_ = best_params.prec_chol
        self.precisions_ = _compute_precisions(best_params.prec_chol, best_params.cov_type)

        self.lower_bound_ = float(best_lower.item())
        self.lower_bounds_ = best_history
        self.n_iter_ = best_n_iter
        self.converged_ = best_converged
        self.covariance_updates_ = best_cov_updates

        return self
