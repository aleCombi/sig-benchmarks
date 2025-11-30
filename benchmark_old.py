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
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        chen.sig(path, m)
        times.append(time.perf_counter() - t0)
    t_chen = min(times) * 1000
    
    # iisignature
    _ = iisignature.sig(path, m)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        iisignature.sig(path, m)
        times.append(time.perf_counter() - t0)
    t_iisig = min(times) * 1000
    
    # pysiglib
    _ = pysiglib.signature(path, m)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        pysiglib.signature(path, m)
        times.append(time.perf_counter() - t0)
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
    sig_pysig = pysiglib.signature(path, m)[1:]  # Strip level 0
    
    diff_iisig = np.abs(sig_chen - sig_iisig).max()
    diff_pysig = np.abs(sig_chen - sig_pysig).max()
    
    print(f"chen vs iisignature: max diff = {diff_iisig:.2e}")
    print(f"chen vs pysiglib:    max diff = {diff_pysig:.2e}")
    
    if diff_iisig < 1e-10 and diff_pysig < 1e-10:
        print("✓ All implementations agree")
    else:
        print("⚠ Significant differences detected")


def benchmark_gradient_correctness(N=50, d=3, m=3):
    """Check gradient correctness via finite differences"""
    print(f"\n{'='*70}")
    print(f"GRADIENT CORRECTNESS: N={N}, d={d}, m={m}")
    print(f"{'='*70}")
    
    from chen.torch import sig_torch
    
    # Use SAME path for all methods
    path_np = np.ascontiguousarray(np.random.randn(N, d), dtype=np.float64)
    
    # 1. Chen autodiff gradients
    path_torch = torch.from_numpy(path_np).requires_grad_(True)
    sig_chen = sig_torch(path_torch, m)
    loss = sig_chen.sum()
    loss.backward()
    grad_chen = path_torch.grad.numpy().copy()
    
    print(f"Chen signature shape: {sig_chen.shape}, sum = {sig_chen.sum().item():.6f}")
    
    # 2. Finite difference gradients (ground truth)
    eps = 1e-6
    grad_fd = np.zeros_like(path_np)
    
    for i in range(N):
        for j in range(d):
            path_plus = path_np.copy()
            path_plus[i, j] += eps
            path_minus = path_np.copy()
            path_minus[i, j] -= eps
            
            sig_plus = chen.sig(path_plus, m).sum()
            sig_minus = chen.sig(path_minus, m).sum()
            
            grad_fd[i, j] = (sig_plus - sig_minus) / (2 * eps)
    
    # 3. pysiglib sig_backprop gradients
    try:
        from pysiglib import signature, sig_backprop
        
        sig_pysig = signature(path_np, m)
        print(f"pysiglib signature shape: {sig_pysig.shape}, sum = {sig_pysig.sum():.6f}")
        
        # sig_derivs should be gradient of sum() w.r.t. signature
        # For sum(), this is all ones
        sig_derivs = np.ones_like(sig_pysig)
        grad_pysig = sig_backprop(path_np, sig_pysig, sig_derivs, m)
        
        # Compare all three
        diff_chen_fd = np.abs(grad_chen - grad_fd).max()
        diff_pysig_fd = np.abs(grad_pysig - grad_fd).max()
        diff_chen_pysig = np.abs(grad_chen - grad_pysig).max()
        
        rel_chen_fd = diff_chen_fd / np.abs(grad_fd).max()
        rel_pysig_fd = diff_pysig_fd / np.abs(grad_fd).max()
        rel_chen_pysig = diff_chen_pysig / np.abs(grad_fd).max()
        
        print(f"\nGradient comparisons:")
        print(f"  chen vs finite diff:   max diff = {diff_chen_fd:.2e}, rel = {rel_chen_fd:.2e}")
        print(f"  pysiglib vs finite diff: max diff = {diff_pysig_fd:.2e}, rel = {rel_pysig_fd:.2e}")
        print(f"  chen vs pysiglib:      max diff = {diff_chen_pysig:.2e}, rel = {rel_chen_pysig:.2e}")
        
        if rel_chen_fd < 1e-4 and rel_pysig_fd < 1e-4:
            print("  ✓ Both methods match finite differences")
        else:
            print("  ⚠ WARNING: Gradient mismatch detected!")
            if rel_chen_fd >= 1e-4:
                print(f"    chen gradients don't match FD!")
            if rel_pysig_fd >= 1e-4:
                print(f"    pysiglib gradients don't match FD!")
            
    except Exception as e:
        print(f"\npysiglib sig_backprop: ERROR - {str(e)[:100]}")


def benchmark_gradient_speed(N=100, d=5, m=4, runs=10, warmup=5):
    """Compare gradient computation speed: chen vs pysiglib"""
    print(f"\n{'='*70}")
    print(f"GRADIENT SPEED: N={N}, d={d}, m={m}")
    print(f"{'='*70}")
    
    from chen.torch import sig_torch
    
    # Test 1: chen through PyTorch - THOROUGH WARMUP for Enzyme
    print("Warming up chen (compiling Enzyme rules)...")
    for _ in range(warmup):
        path = torch.randn(N, d, requires_grad=True, dtype=torch.float64)
        sig = sig_torch(path, m)
        loss = sig.sum()
        loss.backward()
    
    times = []
    for _ in range(runs):
        path = torch.randn(N, d, requires_grad=True, dtype=torch.float64)
        t0 = time.perf_counter()
        sig = sig_torch(path, m)
        loss = sig.sum()
        loss.backward()
        times.append(time.perf_counter() - t0)
    t_chen_torch = min(times) * 1000
    
    # Test 2: chen raw Julia (check if sig_gradient exists)
    t_chen_raw = None
    
    # Test 3: pysiglib - also add warmup for fairness
    try:
        from pysiglib import signature, sig_backprop
        
        print("Warming up pysiglib...")
        for _ in range(warmup):
            path_np = np.ascontiguousarray(np.random.randn(N, d), dtype=np.float64)
            sig = signature(path_np, m)
            sig_derivs = np.ones_like(sig)
            _ = sig_backprop(path_np, sig, sig_derivs, m)
        
        times = []
        for _ in range(runs):
            path_np = np.ascontiguousarray(np.random.randn(N, d), dtype=np.float64)
            t0 = time.perf_counter()
            sig = signature(path_np, m)
            sig_derivs = np.ones_like(sig)
            grad_path = sig_backprop(path_np, sig, sig_derivs, m)
            times.append(time.perf_counter() - t0)
        t_pysig = min(times) * 1000
        
        if t_chen_raw is not None:
            print(f"\nchen (PyTorch):  {t_chen_torch:6.1f} ms  (has PyTorch overhead)")
            print(f"chen (raw):      {t_chen_raw:6.1f} ms  (baseline)")
            print(f"pysiglib:        {t_pysig:6.1f} ms  ({t_pysig/t_chen_raw:.2f}x)")
        else:
            print(f"\nchen (PyTorch):  {t_chen_torch:6.1f} ms  (baseline)")
            print(f"pysiglib:        {t_pysig:6.1f} ms  ({t_pysig/t_chen_torch:.2f}x)")
            print("\nNote: chen.sig_gradient not found, can't test raw Julia speed")
        
    except Exception as e:
        print(f"chen:     {t_chen_torch:6.1f} ms  ✓ WORKING")
        print(f"pysiglib: ERROR - {str(e)[:80]}")


if __name__ == "__main__":
    # Speed benchmarks at different scales
    benchmark_speed(N=100, d=3, m=5)
    benchmark_speed(N=500, d=5, m=6)
    benchmark_speed(N=1000, d=5, m=7)
    
    # Correctness
    benchmark_correctness(N=100, d=3, m=5)
    
    # Gradients
    try:
        benchmark_gradient_correctness(N=50, d=3, m=3)
        benchmark_gradient_speed(N=100, d=5, m=5)
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"GRADIENT TESTS SKIPPED")
        print(f"{'='*70}")
        print(f"Error: {str(e)[:200]}")
    
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}\n")