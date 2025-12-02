#!/usr/bin/env python3
"""pysiglib adapter for signature benchmarks"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Add src directory to path for common module
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from common import BenchmarkAdapter, make_path


class PySigLibAdapter(BenchmarkAdapter):
    """Adapter for pysiglib library"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Import here to avoid import errors if not available
        import pysiglib
        self.pysiglib = pysiglib

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Setup phase (untimed): ensure path is contiguous
        path = np.ascontiguousarray(path, dtype=np.float64)

        # Return kernel closure
        return lambda: self.pysiglib.signature(path, degree=m)

    def run_sigdiff(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature differentiation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Setup phase (untimed): ensure path is contiguous
        path = np.ascontiguousarray(path, dtype=np.float64)

        # Return kernel closure that computes signature + backprop
        def kernel():
            sig = self.pysiglib.signature(path, degree=m)
            sig_derivs = np.ones_like(sig)
            self.pysiglib.sig_backprop(path, sig, sig_derivs, m)

        return kernel

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """Execute the benchmark"""
        # Generate path
        path = make_path(self.d, self.N, self.path_kind)

        # Select operation
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "signature"
        elif self.operation == "sigdiff":
            kernel = self.run_sigdiff(path, self.d, self.m)
            method = "sigdiff"
        else:
            # Operation not supported (logsignature not available in pysiglib)
            return None

        if kernel is None:
            return None

        # Run manual timing loop
        t_ms, alloc_bytes = self.manual_timing_loop(kernel)

        # Format and return result
        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            library="pysiglib",
            method=method,
            path_type="ndarray",
            language="python"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_pysiglib.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    # Parse configuration from command line
    config = json.loads(sys.argv[1])

    # Create and run adapter
    adapter = PySigLibAdapter(config)
    adapter.run()
