#!/usr/bin/env python3
"""chen-signatures adapter for signature benchmarks"""

import json
import sys
from typing import Any, Callable, Dict, Optional

import numpy as np
from common import BenchmarkAdapter, make_path


class ChenSignaturesAdapter(BenchmarkAdapter):
    """Adapter for chen-signatures library"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Import here to avoid import errors if not available
        import chen
        self.chen = chen

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Setup phase (untimed): ensure path is contiguous
        path = np.ascontiguousarray(path, dtype=np.float64)

        # Return kernel closure
        return lambda: self.chen.sig(path, m)

    def run_logsignature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare logsignature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Check if logsig methods are available
        if not (hasattr(self.chen, "logsig") and hasattr(self.chen, "prepare_logsig")):
            return None

        # Setup phase (untimed): prepare basis and ensure path is contiguous
        path = np.ascontiguousarray(path, dtype=np.float64)
        basis = self.chen.prepare_logsig(d, m)

        # Return kernel closure
        return lambda: self.chen.logsig(path, basis)

    def run_sigdiff(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature differentiation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Import PyTorch dependencies
        try:
            from chen.torch import sig_torch
            import torch
        except ImportError:
            return None

        # Setup phase (untimed): prepare path as numpy array
        path_np = np.ascontiguousarray(make_path(d, self.N, self.path_kind), dtype=np.float64)

        # Return kernel closure that converts to torch, computes sig, and backprop
        def kernel():
            path_t = torch.tensor(path_np, dtype=torch.float64, requires_grad=True)
            sig = sig_torch(path_t, m)
            loss = sig.sum()
            loss.backward()

        return kernel

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """Execute the benchmark"""
        # Generate path
        path = make_path(self.d, self.N, self.path_kind)

        # Select operation
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "sig"
            path_type = "ndarray"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = "logsig(prepared)"
            path_type = "ndarray"
        elif self.operation == "sigdiff":
            kernel = self.run_sigdiff(path, self.d, self.m)
            method = "sigdiff"
            path_type = "torch"
        else:
            # Operation not supported
            return None

        if kernel is None:
            return None

        # Run manual timing loop
        t_ms = self.manual_timing_loop(kernel)

        # Format and return result
        return self.output_result(
            t_ms=t_ms,
            library="chen-signatures",
            method=method,
            path_type=path_type,
            language="python"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_chen.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    # Parse configuration from command line
    config = json.loads(sys.argv[1])

    # Create and run adapter
    adapter = ChenSignaturesAdapter(config)
    adapter.run()
