"""Base benchmark adapter with manual timing loop"""

import gc
import json
import sys
import time
from typing import Any, Callable, Dict, Optional


class BenchmarkAdapter:
    """
    Base class for benchmark adapters.

    Implements the manual timing loop pattern to ensure fairness:
    - Warmup phase (untimed)
    - Disable garbage collection
    - Manual loop with perf_counter
    - Average timing calculation
    - Re-enable garbage collection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adapter with configuration.

        Args:
            config: Configuration dictionary containing:
                - N: Number of path points
                - d: Dimension
                - m: Signature truncation level
                - path_kind: "linear" or "sin"
                - operation: "signature", "logsignature", or "sigdiff"
                - repeats: Number of timing repetitions
        """
        self.config = config
        self.N = config["N"]
        self.d = config["d"]
        self.m = config["m"]
        self.path_kind = config["path_kind"]
        self.operation = config["operation"]
        self.repeats = config["repeats"]

    def manual_timing_loop(
        self,
        func: Callable[[], Any],
        warmup_iterations: int = 3
    ) -> float:
        """
        Execute manual timing loop with warmup and GC disabled.

        Args:
            func: Function to time (should be a closure/lambda wrapping the kernel)
            warmup_iterations: Number of warmup runs before timing

        Returns:
            Average time per iteration in milliseconds
        """
        # Warmup phase (untimed)
        for _ in range(warmup_iterations):
            func()

        # Timed phase with GC disabled
        gc.disable()
        try:
            t0 = time.perf_counter()
            for _ in range(self.repeats):
                func()
            t1 = time.perf_counter()
        finally:
            gc.enable()

        # Calculate average time in milliseconds
        total_time_sec = t1 - t0
        avg_time_ms = (total_time_sec / self.repeats) * 1000.0
        return avg_time_ms

    def run_signature(self, path, d: int, m: int) -> Optional[Callable]:
        """
        Prepare and return kernel for signature computation.

        This method should be overridden by subclasses to return a callable
        that performs only the kernel computation (no setup).

        Args:
            path: The input path (format depends on library)
            d: Dimension
            m: Signature level

        Returns:
            Callable that performs the signature computation, or None if not supported
        """
        raise NotImplementedError("Subclasses must implement run_signature")

    def run_logsignature(self, path, d: int, m: int) -> Optional[Callable]:
        """
        Prepare and return kernel for logsignature computation.

        This method should be overridden by subclasses to return a callable
        that performs only the kernel computation (no setup).

        Args:
            path: The input path (format depends on library)
            d: Dimension
            m: Signature level

        Returns:
            Callable that performs the logsignature computation, or None if not supported
        """
        return None  # Default: not supported

    def run_sigdiff(self, path, d: int, m: int) -> Optional[Callable]:
        """
        Prepare and return kernel for signature differentiation.

        This method should be overridden by subclasses to return a callable
        that performs only the kernel computation (no setup).

        Args:
            path: The input path (format depends on library)
            d: Dimension
            m: Signature level

        Returns:
            Callable that performs the sigdiff computation, or None if not supported
        """
        return None  # Default: not supported

    def run(self) -> None:
        """
        Execute the benchmark and output results as JSON to stdout.

        This is the main entry point called by the orchestrator.
        """
        try:
            result = self._run_benchmark()
            if result is not None:
                # Output as single-line JSON for orchestrator to parse
                print(json.dumps(result), flush=True)
        except Exception as e:
            error_result = {
                "error": str(e),
                "N": self.N,
                "d": self.d,
                "m": self.m,
                "operation": self.operation,
            }
            print(json.dumps(error_result), file=sys.stderr, flush=True)
            sys.exit(1)

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """
        Internal method to run the actual benchmark.

        Returns:
            Dictionary with benchmark results, or None if operation not supported
        """
        raise NotImplementedError("Subclasses must implement _run_benchmark")

    def output_result(
        self,
        t_ms: float,
        library: str,
        method: str,
        path_type: str = "ndarray",
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Format benchmark result for output.

        Args:
            t_ms: Time in milliseconds
            library: Library name
            method: Method name
            path_type: Path type descriptor
            language: Programming language

        Returns:
            Formatted result dictionary
        """
        return {
            "N": self.N,
            "d": self.d,
            "m": self.m,
            "path_kind": self.path_kind,
            "operation": self.operation,
            "language": language,
            "library": library,
            "method": method,
            "path_type": path_type,
            "t_ms": t_ms,
        }
