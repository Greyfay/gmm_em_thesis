#!/usr/bin/env python3
"""Comprehensive parallelization analysis comparing old vs new GMM implementations.

This script analyzes:
1. Amdahl's Law: Serial fractions, theoretical vs actual speedup
2. GPU Occupancy: Hardware utilization, memory bandwidth, compute vs memory-bound
3. Kernel Fusion: Launch overhead, memory traffic reduction

Outputs results to parallelization_analysis.xlsx
"""

import sys
import os
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Add parent directory to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import implementations
from implementation import _v0_ref as old_impl
from implementation import _v1 as new_impl


# ============================================================================
# Data Generation & Utilities
# ============================================================================

def generate_test_data(
    N: int = 1000,
    D: int = 50,
    K: int = 5,
    device: str = "cpu",
    dtype=torch.float64,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic test data for benchmarking."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X = torch.randn(N, D, device=device, dtype=dtype)
    means = torch.randn(K, D, device=device, dtype=dtype)
    weights = torch.softmax(torch.randn(K, device=device, dtype=dtype), dim=0)
    
    # Generate covariances
    cov_full = torch.stack([
        torch.eye(D, device=device, dtype=dtype) + 0.1 * torch.randn(D, D, device=device, dtype=dtype)
        for _ in range(K)
    ])
    cov_full = torch.bmm(cov_full, cov_full.transpose(-1, -2))
    
    cov_tied = torch.eye(D, device=device, dtype=dtype) + 0.1 * torch.randn(D, D, device=device, dtype=dtype)
    cov_tied = cov_tied @ cov_tied.T
    
    cov_diag = torch.rand(K, D, device=device, dtype=dtype) + 0.5
    cov_spherical = torch.rand(K, device=device, dtype=dtype) + 0.5
    
    log_resp = torch.randn(N, K, device=device, dtype=dtype)
    log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
    
    return {
        "X": X,
        "means": means,
        "weights": weights,
        "cov_full": cov_full,
        "cov_tied": cov_tied,
        "cov_diag": cov_diag,
        "cov_spherical": cov_spherical,
        "log_resp": log_resp,
        "resp": torch.exp(log_resp),
    }


def timer(func, *args, n_runs: int = 10, warmup: int = 2, **kwargs) -> Tuple[float, float]:
    """Time a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return np.mean(times), np.std(times)


def get_gpu_specs(peak_bandwidth: Optional[float] = None) -> Dict[str, Any]:
    """Get GPU specifications if available.
    
    Args:
        peak_bandwidth: Peak memory bandwidth in GB/s. If None, estimates based on compute capability.
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    device_count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    
    # Use provided bandwidth or estimate based on GPU generation
    if peak_bandwidth is None:
        # Rough estimates by compute capability
        estimated_bandwidth = 900 if props.major >= 8 else 600 if props.major >= 7 else 300
        bandwidth_note = "estimated"
    else:
        estimated_bandwidth = peak_bandwidth
        bandwidth_note = "user-provided"
    
    return {
        "available": True,
        "device_count": device_count,
        "device_id": 0,  # Analysis runs on GPU 0
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / (1024**3),
        "multiprocessor_count": props.multi_processor_count,
        "max_threads_per_multiprocessor": props.max_threads_per_multi_processor,
        "memory_bandwidth_gbs": estimated_bandwidth,
        "bandwidth_note": bandwidth_note,
    }


# ============================================================================
# 1. Amdahl's Law Analysis
# ============================================================================

@dataclass
class ComponentTiming:
    """Timing breakdown for a component."""
    name: str
    time_ms: float
    std_ms: float
    fraction: float = 0.0  # Will be computed
    parallel_potential: float = 1.0  # 1.0 = fully parallelizable, 0.0 = serial


class AmdahlAnalyzer:
    """Analyze code using Amdahl's Law to determine speedup limits."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = []
    
    def profile_component_breakdown(
        self,
        impl_name: str,
        impl_module,
        data: Dict[str, torch.Tensor],
        cov_type: str = "full",
    ) -> List[ComponentTiming]:
        """Profile individual components of EM iteration."""
        components = []
        
        X = data["X"]
        means = data["means"]
        weights = data["weights"]
        log_resp = data["log_resp"]
        
        if cov_type == "full":
            cov = data["cov_full"]
        elif cov_type == "tied":
            cov = data["cov_tied"]
        elif cov_type == "diag":
            cov = data["cov_diag"]
        else:
            cov = data["cov_spherical"]
        
        # 1. Precision computation
        t_mean, t_std = timer(impl_module._compute_precisions_cholesky, cov, cov_type, n_runs=20)
        components.append(ComponentTiming(
            "compute_precisions_cholesky",
            t_mean, t_std,
            parallel_potential=1.0  # Fully parallelizable (matrix ops)
        ))
        
        prec_chol = impl_module._compute_precisions_cholesky(cov, cov_type)
        
        # 2. Log probability computation (E-step core)
        if cov_type == "full":
            log_prob_func = impl_module._estimate_log_gaussian_prob_full_precchol
        elif cov_type == "tied":
            log_prob_func = impl_module._estimate_log_gaussian_prob_tied_precchol
        elif cov_type == "diag":
            log_prob_func = impl_module._estimate_log_gaussian_prob_diag_precchol
        else:
            log_prob_func = impl_module._estimate_log_gaussian_prob_spherical_precchol
        
        t_mean, t_std = timer(log_prob_func, X, means, prec_chol, n_runs=20)
        components.append(ComponentTiming(
            f"log_prob_{cov_type}",
            t_mean, t_std,
            parallel_potential=0.95  # Mostly parallel, some overhead
        ))
        
        # 3. E-step (complete)
        t_mean, t_std = timer(impl_module._expectation_step_precchol, X, means, prec_chol, weights, cov_type, n_runs=20)
        components.append(ComponentTiming(
            "expectation_step",
            t_mean, t_std,
            parallel_potential=0.95  # Mostly parallel
        ))
        
        # 4. M-step (complete - where most parallelization happens)
        t_mean, t_std = timer(impl_module._maximization_step, X, means, cov, weights, log_resp, cov_type, 1e-6, n_runs=20)
        components.append(ComponentTiming(
            f"maximization_step_{cov_type}",
            t_mean, t_std,
            parallel_potential=0.85  # Mix of parallel and serial operations
        ))
        
        # Compute fractions
        total_time = sum(c.time_ms for c in components)
        for c in components:
            c.fraction = c.time_ms / total_time
        
        return components
    
    def compute_amdahl_metrics(
        self,
        components: List[ComponentTiming],
        p_cores: int = None,
    ) -> Dict[str, float]:
        """Compute Amdahl's Law metrics from component breakdown."""
        total_time = sum(c.time_ms for c in components)
        
        # Calculate serial fraction (weighted by parallel potential)
        serial_time = sum(c.time_ms * (1 - c.parallel_potential) for c in components)
        parallel_time = sum(c.time_ms * c.parallel_potential for c in components)
        
        f_serial = serial_time / total_time
        f_parallel = parallel_time / total_time
        
        # Theoretical speedup with infinite processors
        speedup_infinite = 1.0 / f_serial if f_serial < 1.0 else float('inf')
        
        # Theoretical speedup with p processors
        if p_cores is None:
            p_cores = os.cpu_count() or 8
        speedup_p = 1.0 / (f_serial + f_parallel / p_cores)
        
        # Parallel efficiency
        efficiency_p = speedup_p / p_cores
        
        return {
            "total_time_ms": total_time,
            "serial_fraction": f_serial,
            "parallel_fraction": f_parallel,
            "serial_time_ms": serial_time,
            "parallel_time_ms": parallel_time,
            "speedup_limit_infinite": speedup_infinite,
            "speedup_limit_p_cores": speedup_p,
            "p_cores": p_cores,
            "parallel_efficiency": efficiency_p,
        }
    
    def analyze_implementations(
        self,
        test_configs: List[Tuple[int, int, int, str]],
    ) -> pd.DataFrame:
        """Compare old vs new implementations with Amdahl's Law analysis."""
        results = []
        
        for N, D, K, cov_type in test_configs:
            print(f"\nAnalyzing N={N}, D={D}, K={K}, cov_type={cov_type}")
            
            data = generate_test_data(N, D, K, device=self.device)
            
            # Analyze old implementation
            old_components = self.profile_component_breakdown("old", old_impl, data, cov_type)
            old_metrics = self.compute_amdahl_metrics(old_components)
            
            # Analyze new implementation
            new_components = self.profile_component_breakdown("new", new_impl, data, cov_type)
            new_metrics = self.compute_amdahl_metrics(new_components)
            
            # Compute actual speedup
            actual_speedup = old_metrics["total_time_ms"] / new_metrics["total_time_ms"]
            
            # Speedup efficiency (actual vs theoretical limit)
            theoretical_limit = old_metrics["speedup_limit_p_cores"]
            speedup_efficiency = (actual_speedup / theoretical_limit) * 100 if theoretical_limit > 0 else 0
            
            result = {
                "N": N,
                "D": D,
                "K": K,
                "cov_type": cov_type,
                "old_total_time_ms": old_metrics["total_time_ms"],
                "new_total_time_ms": new_metrics["total_time_ms"],
                "old_serial_fraction": old_metrics["serial_fraction"],
                "new_serial_fraction": new_metrics["serial_fraction"],
                "old_serial_time_ms": old_metrics["serial_time_ms"],
                "new_serial_time_ms": new_metrics["serial_time_ms"],
                "old_speedup_limit": old_metrics["speedup_limit_infinite"],
                "new_speedup_limit": new_metrics["speedup_limit_infinite"],
                "theoretical_speedup_p_cores": theoretical_limit,
                "actual_speedup": actual_speedup,
                "speedup_efficiency_pct": speedup_efficiency,
                "p_cores": old_metrics["p_cores"],
            }
            
            results.append(result)
            
            # Store component breakdowns for detailed analysis
            for comp in old_components:
                self.results.append({
                    "impl": "old",
                    "N": N, "D": D, "K": K,
                    "cov_type": cov_type,
                    "component": comp.name,
                    "time_ms": comp.time_ms,
                    "std_ms": comp.std_ms,
                    "fraction": comp.fraction,
                    "parallel_potential": comp.parallel_potential,
                })
            
            for comp in new_components:
                self.results.append({
                    "impl": "new",
                    "N": N, "D": D, "K": K,
                    "cov_type": cov_type,
                    "component": comp.name,
                    "time_ms": comp.time_ms,
                    "std_ms": comp.std_ms,
                    "fraction": comp.fraction,
                    "parallel_potential": comp.parallel_potential,
                })
        
        return pd.DataFrame(results)
    
    def get_component_breakdown(self) -> pd.DataFrame:
        """Get detailed component breakdown."""
        return pd.DataFrame(self.results)


# ============================================================================
# 2. GPU Occupancy & Memory Bandwidth Analysis
# ============================================================================

class GPUProfiler:
    """Profile GPU occupancy and memory bandwidth utilization."""
    
    def __init__(self, peak_bandwidth: Optional[float] = None):
        self.gpu_specs = get_gpu_specs(peak_bandwidth)
        self.results = []
    
    def profile_with_pytorch_profiler(
        self,
        impl_name: str,
        func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Profile function using PyTorch profiler."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            with record_function(f"{impl_name}"):
                result = func(*args, **kwargs)
                torch.cuda.synchronize()
        
        # Extract metrics
        events = prof.key_averages()
        
        cuda_time = 0
        cpu_time = 0
        for evt in events:
            # Use self_cuda_time_total and self_cpu_time_total for averaged events
            cuda_time += getattr(evt, 'self_cuda_time_total', 0) or getattr(evt, 'cuda_time_total', 0)
            cpu_time += getattr(evt, 'self_cpu_time_total', 0) or getattr(evt, 'cpu_time_total', 0)
        
        # Convert to ms
        cuda_time_ms = cuda_time / 1000
        cpu_time_ms = cpu_time / 1000
        
        return {
            "cuda_time_ms": cuda_time_ms,
            "cpu_time_ms": cpu_time_ms,
            "total_time_ms": cpu_time_ms + cuda_time_ms,
            "gpu_utilization_pct": (cuda_time_ms / (cpu_time_ms + cuda_time_ms)) * 100 if (cpu_time_ms + cuda_time_ms) > 0 else 0,
        }
    
    def estimate_memory_bandwidth(
        self,
        impl_name: str,
        func,
        data_size_bytes: int,
        *args,
        n_runs: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        """Estimate memory bandwidth utilization."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        # Time the operation
        torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        
        # Achieved bandwidth (GB/s)
        # Assume we read and write the data
        achieved_bandwidth_gbs = (data_size_bytes * 2) / (avg_time * 1e9)
        
        # Peak bandwidth (from specs or typical values)
        peak_bandwidth = self.gpu_specs.get("memory_bandwidth_gbs", 900)  # Typical for modern GPUs
        
        bandwidth_utilization = (achieved_bandwidth_gbs / peak_bandwidth) * 100
        
        return {
            "achieved_bandwidth_gbs": achieved_bandwidth_gbs,
            "peak_bandwidth_gbs": peak_bandwidth,
            "bandwidth_utilization_pct": bandwidth_utilization,
            "avg_time_ms": avg_time * 1000,
        }
    
    def analyze_gpu_utilization(
        self,
        test_configs: List[Tuple[int, int, int, str]],
        device: str = "cuda",
    ) -> pd.DataFrame:
        """Analyze GPU utilization for different operations."""
        print(f"\nDEBUG: analyze_gpu_utilization called with device={device}")
        print(f"DEBUG: torch.cuda.is_available()={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("DEBUG: CUDA not available in analyze_gpu_utilization, returning empty dataframe")
            return pd.DataFrame()
        
        results = []
        
        for N, D, K, cov_type in test_configs:
            print(f"\nGPU Profiling N={N}, D={D}, K={K}, cov_type={cov_type}")
            
            data = generate_test_data(N, D, K, device=device)
            
            X = data["X"]
            means = data["means"]
            weights = data["weights"]
            log_resp = data["log_resp"]
            
            if cov_type == "full":
                cov = data["cov_full"]
            elif cov_type == "tied":
                cov = data["cov_tied"]
            elif cov_type == "diag":
                cov = data["cov_diag"]
            else:
                cov = data["cov_spherical"]
            
            # Calculate data sizes for bandwidth estimation
            x_size = X.element_size() * X.nelement()
            cov_size = cov.element_size() * cov.nelement()
            
            # Get precision for log prob test
            prec_chol = old_impl._compute_precisions_cholesky(cov, cov_type)
            
            # Profile key operations for both implementations
            operations = [
                ("compute_precisions_cholesky", lambda impl: impl._compute_precisions_cholesky(cov, cov_type), cov_size),
                ("e_step", lambda impl: impl._expectation_step_precchol(X, means, prec_chol, weights, cov_type), x_size),
                ("m_step", lambda impl: impl._maximization_step(X, means, cov, weights, log_resp, cov_type, 1e-6), x_size),
            ]
            
            for op_name, op_func, data_size in operations:
                # Old implementation
                old_prof = self.profile_with_pytorch_profiler("old_" + op_name, op_func, old_impl)
                old_bw = self.estimate_memory_bandwidth("old_" + op_name, op_func, data_size, old_impl)
                
                # New implementation
                new_prof = self.profile_with_pytorch_profiler("new_" + op_name, op_func, new_impl)
                new_bw = self.estimate_memory_bandwidth("new_" + op_name, op_func, data_size, new_impl)
                
                results.append({
                    "N": N, "D": D, "K": K,
                    "cov_type": cov_type,
                    "operation": op_name,
                    "old_cuda_time_ms": old_prof.get("cuda_time_ms", 0),
                    "new_cuda_time_ms": new_prof.get("cuda_time_ms", 0),
                    "old_cpu_time_ms": old_prof.get("cpu_time_ms", 0),
                    "new_cpu_time_ms": new_prof.get("cpu_time_ms", 0),
                    "old_gpu_util_pct": old_prof.get("gpu_utilization_pct", 0),
                    "new_gpu_util_pct": new_prof.get("gpu_utilization_pct", 0),
                    "old_bandwidth_gbs": old_bw.get("achieved_bandwidth_gbs", 0),
                    "new_bandwidth_gbs": new_bw.get("achieved_bandwidth_gbs", 0),
                    "old_bandwidth_util_pct": old_bw.get("bandwidth_utilization_pct", 0),
                    "new_bandwidth_util_pct": new_bw.get("bandwidth_utilization_pct", 0),
                    "peak_bandwidth_gbs": old_bw.get("peak_bandwidth_gbs", 0),
                })
        
        return pd.DataFrame(results)


# ============================================================================
# 3. Kernel Fusion Impact Analysis
# ============================================================================

class KernelFusionAnalyzer:
    """Analyze kernel launch overhead and fusion opportunities."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = []
    
    def count_kernel_launches(
        self,
        impl_name: str,
        func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Count kernel launches using profiler."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        with profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
        
        # Count CUDA kernels
        kernel_count = 0
        total_kernel_time = 0
        kernel_times = []
        
        for evt in prof.key_averages():
            if evt.device_type == torch.profiler.DeviceType.CUDA:
                kernel_count += 1
                # Use self_cuda_time_total for averaged events
                kernel_time = getattr(evt, 'self_cuda_time_total', 0) or getattr(evt, 'cuda_time_total', 0)
                kernel_time_ms = kernel_time / 1000  # Convert to ms
                total_kernel_time += kernel_time_ms
                kernel_times.append(kernel_time_ms)
        
        avg_kernel_time = total_kernel_time / kernel_count if kernel_count > 0 else 0
        
        return {
            "kernel_count": kernel_count,
            "total_kernel_time_ms": total_kernel_time,
            "avg_kernel_time_ms": avg_kernel_time,
            "kernel_times": kernel_times,
        }
    
    def estimate_launch_overhead(
        self,
        kernel_count: int,
        total_time_ms: float,
        total_kernel_time_ms: float,
    ) -> Dict[str, float]:
        """Estimate kernel launch overhead."""
        # Typical kernel launch overhead is ~5-10 microseconds
        # This is a rough estimate
        estimated_launch_overhead_per_kernel_us = 7.0
        total_launch_overhead_ms = (kernel_count * estimated_launch_overhead_per_kernel_us) / 1000
        
        # Overhead as percentage of total time
        overhead_pct = (total_launch_overhead_ms / total_time_ms) * 100 if total_time_ms > 0 else 0
        
        return {
            "estimated_launch_overhead_ms": total_launch_overhead_ms,
            "launch_overhead_pct": overhead_pct,
        }
    
    def analyze_memory_traffic(
        self,
        impl_name: str,
        N: int,
        D: int,
        K: int,
        cov_type: str,
        kernel_count: int,
    ) -> Dict[str, float]:
        """Estimate memory traffic and fusion potential."""
        dtype_size = 8  # float64
        
        # Estimate data sizes
        x_size = N * D * dtype_size
        means_size = K * D * dtype_size
        resp_size = N * K * dtype_size
        
        if cov_type == "full":
            cov_size = K * D * D * dtype_size
        elif cov_type == "tied":
            cov_size = D * D * dtype_size
        elif cov_type == "diag":
            cov_size = K * D * dtype_size
        else:
            cov_size = K * dtype_size
        
        # Total working set
        total_data_mb = (x_size + means_size + resp_size + cov_size) / (1024 ** 2)
        
        # Estimate redundant memory traffic
        # Each kernel launch typically reads/writes data
        # Old implementation with loops -> more launches -> more traffic
        estimated_traffic_mb = kernel_count * total_data_mb * 0.3  # Conservative estimate
        
        # Fusion potential: if we reduce kernel count by 50%, we save memory traffic
        fusion_potential_pct = 50.0  # Typical fusion can reduce by 30-70%
        potential_saved_mb = estimated_traffic_mb * (fusion_potential_pct / 100)
        
        return {
            "total_data_mb": total_data_mb,
            "estimated_traffic_mb": estimated_traffic_mb,
            "fusion_potential_pct": fusion_potential_pct,
            "potential_saved_mb": potential_saved_mb,
            "traffic_per_kernel_mb": estimated_traffic_mb / kernel_count if kernel_count > 0 else 0,
        }
    
    def analyze_fusion_opportunities(
        self,
        test_configs: List[Tuple[int, int, int, str]],
    ) -> pd.DataFrame:
        """Analyze kernel fusion opportunities."""
        print(f"\nDEBUG: analyze_fusion_opportunities called")
        print(f"DEBUG: torch.cuda.is_available()={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("DEBUG: CUDA not available in analyze_fusion_opportunities, returning empty dataframe")
            return pd.DataFrame()
        
        results = []
        
        for N, D, K, cov_type in test_configs:
            print(f"\nKernel Fusion Analysis N={N}, D={D}, K={K}, cov_type={cov_type}")
            
            data = generate_test_data(N, D, K, device=self.device)
            
            X = data["X"]
            means = data["means"]
            weights = data["weights"]
            log_resp = data["log_resp"]
            
            if cov_type == "full":
                cov = data["cov_full"]
            elif cov_type == "tied":
                cov = data["cov_tied"]
            elif cov_type == "diag":
                cov = data["cov_diag"]
            else:
                cov = data["cov_spherical"]
            
            # Analyze M-step (covariance update) for kernel fusion potential
            # This is where the old version has loops that launch many kernels
            cov_func = lambda impl: impl._maximization_step(X, means, cov, weights, log_resp, cov_type, 1e-6)
            
            # Count kernels for old implementation
            old_kernel_info = self.count_kernel_launches("old", cov_func, old_impl)
            old_time, _ = timer(cov_func, old_impl, n_runs=10)
            
            # Count kernels for new implementation
            new_kernel_info = self.count_kernel_launches("new", cov_func, new_impl)
            new_time, _ = timer(cov_func, new_impl, n_runs=10)
            
            # Estimate overhead
            old_overhead = self.estimate_launch_overhead(
                old_kernel_info.get("kernel_count", 0),
                old_time,
                old_kernel_info.get("total_kernel_time_ms", 0)
            )
            
            new_overhead = self.estimate_launch_overhead(
                new_kernel_info.get("kernel_count", 0),
                new_time,
                new_kernel_info.get("total_kernel_time_ms", 0)
            )
            
            # Analyze memory traffic
            old_traffic = self.analyze_memory_traffic(
                "old", N, D, K, cov_type,
                old_kernel_info.get("kernel_count", 0)
            )
            
            new_traffic = self.analyze_memory_traffic(
                "new", N, D, K, cov_type,
                new_kernel_info.get("kernel_count", 0)
            )
            
            # Kernel reduction benefit
            kernel_reduction_pct = ((old_kernel_info.get("kernel_count", 0) - new_kernel_info.get("kernel_count", 0)) / 
                                   old_kernel_info.get("kernel_count", 1)) * 100
            
            results.append({
                "N": N, "D": D, "K": K,
                "cov_type": cov_type,
                "old_kernel_count": old_kernel_info.get("kernel_count", 0),
                "new_kernel_count": new_kernel_info.get("kernel_count", 0),
                "kernel_reduction_pct": kernel_reduction_pct,
                "old_total_time_ms": old_time,
                "new_total_time_ms": new_time,
                "old_launch_overhead_ms": old_overhead["estimated_launch_overhead_ms"],
                "new_launch_overhead_ms": new_overhead["estimated_launch_overhead_ms"],
                "old_launch_overhead_pct": old_overhead["launch_overhead_pct"],
                "new_launch_overhead_pct": new_overhead["launch_overhead_pct"],
                "old_estimated_traffic_mb": old_traffic["estimated_traffic_mb"],
                "new_estimated_traffic_mb": new_traffic["estimated_traffic_mb"],
                "traffic_reduction_mb": old_traffic["estimated_traffic_mb"] - new_traffic["estimated_traffic_mb"],
                "traffic_reduction_pct": ((old_traffic["estimated_traffic_mb"] - new_traffic["estimated_traffic_mb"]) / 
                                         old_traffic["estimated_traffic_mb"]) * 100 if old_traffic["estimated_traffic_mb"] > 0 else 0,
            })
        
        return pd.DataFrame(results)


# ============================================================================
# Main Analysis & Excel Export
# ============================================================================

def run_comprehensive_analysis(device: str = "cpu", output_file: str = "parallelization_analysis.xlsx", 
                              peak_bandwidth: Optional[float] = None):
    """Run all analyses and export to Excel."""
    print("="*80)
    print("COMPREHENSIVE PARALLELIZATION ANALYSIS")
    print("="*80)
    print(f"Device: {device}")
    print(f"Output: {output_file}")
    
    # GPU specs
    gpu_specs = get_gpu_specs(peak_bandwidth)
    print(f"\nGPU Available: {gpu_specs['available']}")
    if gpu_specs['available']:
        print(f"GPU: {gpu_specs['name']}")
        if gpu_specs.get('device_count', 1) > 1:
            print(f"Devices: {gpu_specs['device_count']} GPUs detected (using GPU 0)")
        print(f"Memory: {gpu_specs['total_memory_gb']:.2f} GB")
        print(f"SMs: {gpu_specs['multiprocessor_count']}")
        print(f"Peak Bandwidth: {gpu_specs['memory_bandwidth_gbs']:.1f} GB/s ({gpu_specs.get('bandwidth_note', 'estimated')})")
    
    # Test configurations - same 3 configurations for each covariance type
    test_configs = [
        # Full covariance
        (1000, 50, 5, "full"),
        (2000, 100, 10, "full"),
        (5000, 50, 5, "full"),
        # Diagonal covariance
        (1000, 50, 5, "diag"),
        (2000, 100, 10, "diag"),
        (5000, 50, 5, "diag"),
        # Tied covariance
        (1000, 50, 5, "tied"),
        (2000, 100, 10, "tied"),
        (5000, 50, 5, "tied"),
        # Spherical covariance
        (1000, 50, 5, "spherical"),
        (2000, 100, 10, "spherical"),
        (5000, 50, 5, "spherical"),
    ]
    
    # 1. Amdahl's Law Analysis
    print("\n" + "="*80)
    print("1. AMDAHL'S LAW ANALYSIS")
    print("="*80)
    amdahl = AmdahlAnalyzer(device=device)
    amdahl_df = amdahl.analyze_implementations(test_configs)
    component_df = amdahl.get_component_breakdown()
    
    # 2. GPU Occupancy Analysis
    gpu_df = pd.DataFrame()
    print(f"\nDEBUG GPU: device={device}, torch.cuda.is_available()={torch.cuda.is_available()}")
    if device == "cuda" and torch.cuda.is_available():
        try:
            print("DEBUG GPU: Entering GPU analysis...")
            print("\n" + "="*80)
            print("2. GPU OCCUPANCY & MEMORY BANDWIDTH ANALYSIS")
            print("="*80)
            gpu_profiler = GPUProfiler(peak_bandwidth)
            print(f"DEBUG GPU: Created profiler with peak_bandwidth={peak_bandwidth}")
            gpu_df = gpu_profiler.analyze_gpu_utilization(test_configs, device=device)
            print(f"DEBUG GPU: Completed analysis, returned {len(gpu_df)} rows")
            if gpu_df.empty:
                print("⚠ Warning: GPU analysis returned no data")
        except Exception as e:
            print(f"⚠ GPU occupancy analysis failed: {e}")
            print("Continuing with remaining analyses...")
            import traceback
            traceback.print_exc()
    else:
        print(f"DEBUG GPU: Skipping GPU analysis - device={device}, cuda_available={torch.cuda.is_available()}")
    
    # 3. Kernel Fusion Analysis
    fusion_df = pd.DataFrame()
    print(f"\nDEBUG FUSION: device={device}, torch.cuda.is_available()={torch.cuda.is_available()}")
    if device == "cuda" and torch.cuda.is_available():
        try:
            print("DEBUG FUSION: Entering kernel fusion analysis...")
            print("\n" + "="*80)
            print("3. KERNEL FUSION IMPACT ANALYSIS")
            print("="*80)
            fusion_analyzer = KernelFusionAnalyzer(device=device)
            print("DEBUG FUSION: Created analyzer")
            fusion_df = fusion_analyzer.analyze_fusion_opportunities(test_configs)
            print(f"DEBUG FUSION: Completed analysis, returned {len(fusion_df)} rows")
            if fusion_df.empty:
                print("⚠ Warning: Kernel fusion analysis returned no data")
        except Exception as e:
            print(f"⚠ Kernel fusion analysis failed: {e}")
            print("Continuing with export...")
            import traceback
            traceback.print_exc()
    else:
        print(f"DEBUG FUSION: Skipping kernel fusion analysis - device={device}, cuda_available={torch.cuda.is_available()}")
    
    # Export to Excel
    print("\n" + "="*80)
    print("EXPORTING TO EXCEL")
    print("="*80)
    print(f"DEBUG EXPORT: gpu_df.empty={gpu_df.empty}, len(gpu_df)={len(gpu_df)}")
    print(f"DEBUG EXPORT: fusion_df.empty={fusion_df.empty}, len(fusion_df)={len(fusion_df)}")
    print(f"DEBUG EXPORT: amdahl_df.empty={amdahl_df.empty}, len(amdahl_df)={len(amdahl_df)}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            "Analysis": ["Amdahl's Law", "GPU Occupancy", "Kernel Fusion"],
            "Status": [
                "Complete",
                "Complete" if not gpu_df.empty else "Skipped (No GPU)",
                "Complete" if not fusion_df.empty else "Skipped (No GPU)",
            ],
            "Sheet Name": ["Amdahl_Summary", "GPU_Analysis", "Kernel_Fusion"],
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        
        # Amdahl's Law results
        amdahl_df.to_excel(writer, sheet_name="Amdahl_Summary", index=False)
        component_df.to_excel(writer, sheet_name="Amdahl_Components", index=False)
        
        # GPU results
        if not gpu_df.empty:
            gpu_df.to_excel(writer, sheet_name="GPU_Analysis", index=False)
        
        # Kernel fusion results
        if not fusion_df.empty:
            fusion_df.to_excel(writer, sheet_name="Kernel_Fusion", index=False)
        
        # GPU specs
        if gpu_specs['available']:
            gpu_specs_df = pd.DataFrame([gpu_specs])
            gpu_specs_df.to_excel(writer, sheet_name="GPU_Specs", index=False)
    
    print(f"✓ Analysis complete! Results saved to: {output_file}")
    print("\nSheets created:")
    print("  - Summary: Overview of analyses")
    print("  - Amdahl_Summary: Speedup limits and efficiency")
    print("  - Amdahl_Components: Detailed component breakdown")
    if not gpu_df.empty:
        print("  - GPU_Analysis: GPU utilization and memory bandwidth")
    if not fusion_df.empty:
        print("  - Kernel_Fusion: Kernel launch and fusion analysis")
    if gpu_specs['available']:
        print("  - GPU_Specs: GPU hardware specifications")


def main():
    """Main entry point."""
    # Hardcoded for consistent SSH setup
    device = "cuda"
    peak_bandwidth = 616  # RTX 2080 Ti
    output_file = "results/parallelization_analysis.xlsx"
    
    run_comprehensive_analysis(device=device, output_file=output_file, 
                              peak_bandwidth=peak_bandwidth)


if __name__ == "__main__":
    main()
