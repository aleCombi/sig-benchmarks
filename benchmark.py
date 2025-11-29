"""
Minimal benchmarks: chen-signatures vs iisignature vs pysiglib
"""
import numpy as np
import time
import chen
import iisignature
import pysiglib
import torch

def benchmark_speed(N=1000, d=5, m=7, runs=20):
    """Compare signature computation speed"""
    print(f"\n{'='*70}")
    print(f"SPEED BENCHMARK: N={N}, d={d}, m={m}")
    print(f"{'='*70}")
    
    path = np.ascontiguousarray(np.random.randn(N, d), dtype=np.float64)
    
    # Chen
    _ = chen.sig(path, m)
    times = [time.perf_counter() for _ in range(runs)]
    for i in range(runs):
        t0 = time.perf_counter()
        chen.sig(path, m)
        times[i] = time.perf_counter() - t0
    t_chen = min(times) * 1000
    
    # iisignature
    _ = iisignature.sig(path, m)
    times = [time.perf_counter() for _ in range(runs)]
    for i in range(runs):
        t0 = time.perf_counter()
        iisignature.sig(path, m)
        times[i] = time.perf_counter() - t0
    t_iisig = min(times) * 1000
    
    # pysiglib
    _ = pysiglib.signature(path, m)
    times = [time.perf_counter() for _ in range(runs)]
    for i in range(runs):
        t0 = time.perf_counter()
        pysiglib.signature(path, m)
        times[i] = time.perf_counter() - t0
    t_pysig = min(times) * 1000
    
    print(f"chen:        {t_chen:6.1f} ms  (baseline)")
    print(f"iisignature: {t_iisig:6.1f} ms  ({t_iisig/t_chen:.2f}x)")
    print(f"pysiglib:    {t_pysig:6.1f} ms  ({t_pysig/t_chen:.2f}x)")


def benchmark_correctness(N=100, d=3, m=5):
    """Compare signature correctness"""
    print(f"\n{'='*70}")
    print(f"CORRECTNESS: N={N}, d={d}, m={m}")
    print(f"{'='*70}")
    
    path = np.ascontiguousarray(np.random.randn(N, d), dtype=np.float64)
    
    sig_chen = chen.sig(path, m)
    sig_iisig = iisignature.sig(path, m)
    sig_pysig = pysiglib.signature(path, m)
    
    diff_iisig = np.abs(sig_chen - sig_iisig).max()
    diff_pysig = np.abs(sig_chen - sig_pysig).max()
    
    print(f"chen vs iisignature: max diff = {diff_iisig:.2e}")
    print(f"chen vs pysiglib:    max diff = {diff_pysig:.2e}")
    
    if diff_iisig < 1e-10 and diff_pysig < 1e-10:
        print("✓ All implementations agree")
    else:
        print("⚠ Significant differences detected")


def benchmark_gradients(N=50, d=3, m=3):
    """Compare PyTorch gradients: chen vs pysiglib"""
    print(f"\n{'='*70}")
    print(f"PYTORCH GRADIENTS: N={N}, d={d}, m={m}")
    print(f"{'='*70}")
    
    # Import torch integration
    from chen.torch import sig_torch
    
    # Chen gradients
    path_chen = torch.randn(N, d, requires_grad=True, dtype=torch.float64)
    sig = sig_torch(path_chen, m)
    loss = sig.sum()
    loss.backward()
    grad_chen = path_chen.grad.clone()
    
    # pysiglib gradients (finite differences)
    path_np = path_chen.detach().numpy()
    eps = 1e-6
    grad_pysig = np.zeros_like(path_np)
    
    for i in range(N):
        for j in range(d):
            path_plus = path_np.copy()
            path_plus[i, j] += eps
            path_minus = path_np.copy()
            path_minus[i, j] -= eps
            
            sig_plus = pysiglib.signature(path_plus, m).sum()
            sig_minus = pysiglib.signature(path_minus, m).sum()
            
            grad_pysig[i, j] = (sig_plus - sig_minus) / (2 * eps)
    
    diff = np.abs(grad_chen.numpy() - grad_pysig).max()
    rel_diff = diff / np.abs(grad_pysig).max()
    
    print(f"Max absolute difference: {diff:.2e}")
    print(f"Max relative difference: {rel_diff:.2e}")
    
    if rel_diff < 1e-4:
        print("✓ Gradients match (within tolerance)")
    else:
        print("⚠ Gradient differences detected")


if __name__ == "__main__":
    # Speed benchmarks at different scales
    benchmark_speed(N=100, d=3, m=5)
    benchmark_speed(N=500, d=5, m=6)
    benchmark_speed(N=1000, d=5, m=7)
    
    # Correctness
    benchmark_correctness(N=100, d=3, m=5)
    
    # Gradients
    benchmark_gradients(N=50, d=3, m=3)
    
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}\n")
