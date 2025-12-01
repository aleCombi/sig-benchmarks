# benchmark.py
import os
import csv
import time
import tracemalloc
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from common import load_config, make_path, SCRIPT_DIR

# ------------------------------------------------------------------
# Library detection (driven by config.libraries)
# ------------------------------------------------------------------

LIB_MODULES: Dict[str, Any] = {}


def setup_libraries(libraries: List[str]) -> None:
    """
    Try to import only the libraries requested in the YAML config.
    We store successfully imported modules in LIB_MODULES keyed by
    their config name, e.g. "chen-signatures" -> chen module.
    """
    global LIB_MODULES
    LIB_MODULES = {}

    for lib in libraries:
        if lib == "chen-signatures":
            try:
                import chen  # type: ignore
                LIB_MODULES["chen-signatures"] = chen
                print("[ok] chen-signatures available")
            except ImportError as e:
                print(f"Warning: chen-signatures import failed: {e}", file=sys.stderr)
        elif lib == "pysiglib":
            try:
                import pysiglib  # type: ignore
                LIB_MODULES["pysiglib"] = pysiglib
                print("[ok] pysiglib available")
            except ImportError as e:
                print(f"Warning: pysiglib import failed: {e}", file=sys.stderr)
        elif lib == "iisignature":
            try:
                import iisignature  # type: ignore
                LIB_MODULES["iisignature"] = iisignature
                print("[ok] iisignature available")
            except ImportError as e:
                print(f"Warning: iisignature import failed: {e}", file=sys.stderr)
        else:
            print(f"Warning: unknown library in config.libraries: {lib}", file=sys.stderr)


# ------------------------------------------------------------------
# Benchmarking helpers
# ------------------------------------------------------------------

def time_and_peak_memory(func, repeats: int = 5):
    """
    Run func() multiple times, return best time and associated peak memory.
    """
    best_time = float("inf")
    best_peak = 0

    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        dur = end - start
        if dur < best_time:
            best_time = dur
            best_peak = peak

    return best_time, best_peak


# ------------------------------------------------------------------
# Per-library benchmarks
# ------------------------------------------------------------------

def bench_chen_signatures(
    d: int,
    m: int,
    N: int,
    path_kind: str,
    operation: str,
    repeats: int,
) -> Optional[Dict[str, Any]]:
    """
    Benchmarks chen-signatures for:
    - signature: chen.sig(path, m)
    - logsignature: basis = chen.prepare_logsig(d, m); chen.logsig(path, basis)
    - sigdiff:   gradient of sum(sig) via chen.torch + PyTorch
    """
    if "chen-signatures" not in LIB_MODULES:
        return None

    chen = LIB_MODULES["chen-signatures"]

    # ---------- signature ----------
    if operation == "signature":
        try:
            path = make_path(d, N, path_kind)
            path = np.ascontiguousarray(path, dtype=np.float64)

            func = lambda: chen.sig(path, m)

            # Warmup
            _ = func()

            t_sec, peak_bytes = time_and_peak_memory(func, repeats=repeats)
            t_ms = t_sec * 1000.0
            alloc_kib = peak_bytes / 1024.0

            return {
                "N": N,
                "d": d,
                "m": m,
                "path_kind": path_kind,
                "operation": operation,
                "language": "python",
                "library": "chen-signatures",
                "method": "sig",
                "path_type": "ndarray",
                "t_ms": t_ms,
                "alloc_KiB": alloc_kib,
            }
        except Exception as e:
            print(f"chen-signatures signature benchmark failed: {e}", file=sys.stderr)
            return None

    # ---------- logsignature ----------
    elif operation == "logsignature":
        # Only run if chen exposes logsig + prepare_logsig
        if not (hasattr(chen, "logsig") and hasattr(chen, "prepare_logsig")):
            print("chen-signatures: logsig/prepare_logsig not found, skipping logsignature", file=sys.stderr)
            return None

        try:
            path = make_path(d, N, path_kind)
            path = np.ascontiguousarray(path, dtype=np.float64)

            # Precompute basis outside the timed section
            basis = chen.prepare_logsig(d, m)

            func = lambda: chen.logsig(path, basis)

            # Warmup
            _ = func()

            t_sec, peak_bytes = time_and_peak_memory(func, repeats=repeats)
            t_ms = t_sec * 1000.0
            alloc_kib = peak_bytes / 1024.0

            return {
                "N": N,
                "d": d,
                "m": m,
                "path_kind": path_kind,
                "operation": operation,
                "language": "python",
                "library": "chen-signatures",
                "method": "logsig(prepared)",
                "path_type": "ndarray",
                "t_ms": t_ms,
                "alloc_KiB": alloc_kib,
            }
        except Exception as e:
            print(f"chen-signatures logsignature benchmark failed: {e}", file=sys.stderr)
            return None

    # ---------- sigdiff (autodiff) ----------
    elif operation == "sigdiff":
        # Gradient speed via PyTorch
        try:
            # Import chen.torch first (juliacall), then torch
            from chen.torch import sig_torch  # type: ignore
            import torch  # type: ignore

            path_np = make_path(d, N, path_kind)
            path_np = np.ascontiguousarray(path_np, dtype=np.float64)

            def single_run():
                path_t = torch.tensor(
                    path_np,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                sig = sig_torch(path_t, m)
                loss = sig.sum()
                loss.backward()

            # Warmup a few times (compilation, Enzyme rules, etc.)
            for _ in range(3):
                single_run()

            t_sec, peak_bytes = time_and_peak_memory(single_run, repeats=repeats)
            t_ms = t_sec * 1000.0
            alloc_kib = peak_bytes / 1024.0

            return {
                "N": N,
                "d": d,
                "m": m,
                "path_kind": path_kind,
                "operation": operation,
                "language": "python",
                "library": "chen-signatures",
                "method": "sigdiff",
                "path_type": "torch",
                "t_ms": t_ms,
                "alloc_KiB": alloc_kib,
            }
        except Exception as e:
            print(f"chen-signatures sigdiff benchmark failed: {e}", file=sys.stderr)
            return None

    return None

def bench_pysiglib(
    d: int,
    m: int,
    N: int,
    path_kind: str,
    operation: str,
    repeats: int,
) -> Optional[Dict[str, Any]]:
    """
    Benchmarks pysiglib for:
    - signature: pysiglib.signature(path, degree=m)
    - sigdiff:   signature + sig_backprop (grad of sum)
    No logsig support.
    """
    if "pysiglib" not in LIB_MODULES:
        return None

    pysiglib = LIB_MODULES["pysiglib"]

    if operation == "signature":
        try:
            path = make_path(d, N, path_kind)
            path = np.ascontiguousarray(path, dtype=np.float64)

            def func():
                pysiglib.signature(path, degree=m)

            # Warmup
            _ = pysiglib.signature(path, degree=m)

            t_sec, peak_bytes = time_and_peak_memory(func, repeats=repeats)
            t_ms = t_sec * 1000.0
            alloc_kib = peak_bytes / 1024.0

            return {
                "N": N,
                "d": d,
                "m": m,
                "path_kind": path_kind,
                "operation": operation,
                "language": "python",
                "library": "pysiglib",
                "method": "signature",
                "path_type": "ndarray",
                "t_ms": t_ms,
                "alloc_KiB": alloc_kib,
            }
        except Exception as e:
            print(f"pysiglib signature benchmark failed: {e}", file=sys.stderr)
            return None

    elif operation == "sigdiff":
        # signature + sig_backprop gradient
        try:
            from pysiglib import signature, sig_backprop  # type: ignore

            path = make_path(d, N, path_kind)
            path = np.ascontiguousarray(path, dtype=np.float64)

            def single_run():
                sig = signature(path, degree=m)
                sig_derivs = np.ones_like(sig)
                _ = sig_backprop(path, sig, sig_derivs, m)

            # Warmup
            for _ in range(3):
                single_run()

            t_sec, peak_bytes = time_and_peak_memory(single_run, repeats=repeats)
            t_ms = t_sec * 1000.0
            alloc_kib = peak_bytes / 1024.0

            return {
                "N": N,
                "d": d,
                "m": m,
                "path_kind": path_kind,
                "operation": operation,
                "language": "python",
                "library": "pysiglib",
                "method": "sigdiff",
                "path_type": "ndarray",
                "t_ms": t_ms,
                "alloc_KiB": alloc_kib,
            }
        except Exception as e:
            print(f"pysiglib sigdiff benchmark failed: {e}", file=sys.stderr)
            return None

    # logsig not supported
    return None


def bench_iisignature(
    d: int,
    m: int,
    N: int,
    path_kind: str,
    operation: str,
    logsig_method: str,
    repeats: int,
) -> Optional[Dict[str, Any]]:
    """
    Benchmarks iisignature for:
    - signature: iisignature.sig(path, m)
    - logsignature: iisignature.logsig(path, basis, method)
    No sigdiff support.
    """
    if "iisignature" not in LIB_MODULES:
        return None

    iisignature = LIB_MODULES["iisignature"]

    if operation == "signature":
        try:
            path = make_path(d, N, path_kind)
            path = np.ascontiguousarray(path, dtype=np.float64)

            func = lambda: iisignature.sig(path, m)

            # Warmup
            _ = func()

            t_sec, peak_bytes = time_and_peak_memory(func, repeats=repeats)
            t_ms = t_sec * 1000.0
            alloc_kib = peak_bytes / 1024.0

            return {
                "N": N,
                "d": d,
                "m": m,
                "path_kind": path_kind,
                "operation": operation,
                "language": "python",
                "library": "iisignature",
                "method": "sig",
                "path_type": "ndarray",
                "t_ms": t_ms,
                "alloc_KiB": alloc_kib,
            }
        except Exception as e:
            print(f"iisignature signature benchmark failed: {e}", file=sys.stderr)
            return None

    elif operation == "logsignature":
        # iisignature logsig support
        if d < 2:
            # iisignature sometimes doesn't like d=1 for logsig
            return None
        try:
            path = make_path(d, N, path_kind)
            path = np.ascontiguousarray(path, dtype=np.float64)

            basis = iisignature.prepare(d, m, logsig_method)

            def func():
                iisignature.logsig(path, basis, logsig_method)

            # Warmup
            _ = iisignature.logsig(path, basis, logsig_method)

            t_sec, peak_bytes = time_and_peak_memory(func, repeats=repeats)
            t_ms = t_sec * 1000.0
            alloc_kib = peak_bytes / 1024.0

            return {
                "N": N,
                "d": d,
                "m": m,
                "path_kind": path_kind,
                "operation": operation,
                "language": "python",
                "library": "iisignature",
                "method": f"logsig({logsig_method})",
                "path_type": "ndarray",
                "t_ms": t_ms,
                "alloc_KiB": alloc_kib,
            }
        except Exception as e:
            print(f"iisignature logsignature benchmark failed: {e}", file=sys.stderr)
            return None

    # sigdiff not supported
    return None


# ------------------------------------------------------------------
# Main sweep + CSV writer
# ------------------------------------------------------------------

def run_bench() -> Path:
    cfg = load_config()
    Ns = cfg["Ns"]
    Ds = cfg["Ds"]
    Ms = cfg["Ms"]
    path_kind = cfg["path_kind"]
    repeats = int(cfg["repeats"])
    logsig_method = cfg["logsig_method"]
    operations = cfg["operations"]
    runs_dir = cfg["runs_dir"]
    libraries_cfg = cfg.get("libraries", [])

    print("=" * 60)
    print("Python Benchmark Suite")
    print("=" * 60)
    print("Configuration:")
    print(f"  path_kind     = {path_kind}")
    print(f"  Ns            = {Ns}")
    print(f"  Ds            = {Ds}")
    print(f"  Ms            = {Ms}")
    print(f"  operations    = {operations}")
    print(f"  repeats       = {repeats}")
    print(f"  logsig_method = {logsig_method} (iisignature; chen uses prepare_logsig)")
    print(f"  libraries     = {libraries_cfg}")
    print()

    # Setup libraries based on config
    setup_libraries(libraries_cfg)

    # Check if orchestrator overrides output CSV location
    env_out_csv = os.environ.get("BENCHMARK_OUT_CSV", "")
    if env_out_csv:
        csv_path = Path(env_out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Orchestrator mode: writing results to {csv_path}")
        run_dir = None
    else:
        # Standalone mode: create own run folder
        from common import setup_run_folder, finalize_run_folder
        run_dir = setup_run_folder("benchmark_python", cfg)
        csv_path = run_dir / "results.csv"

    results: List[Dict[str, Any]] = []

    for N in Ns:
        for d in Ds:
            for m in Ms:
                for op in operations:
                    # chen-signatures
                    if "chen-signatures" in libraries_cfg:
                        res = bench_chen_signatures(d, m, N, path_kind, op, repeats=repeats)
                        if res is not None:
                            results.append(res)

                    # pysiglib
                    if "pysiglib" in libraries_cfg:
                        res = bench_pysiglib(d, m, N, path_kind, op, repeats=repeats)
                        if res is not None:
                            results.append(res)

                    # iisignature
                    if "iisignature" in libraries_cfg:
                        res = bench_iisignature(
                            d, m, N, path_kind, op, logsig_method, repeats=repeats
                        )
                        if res is not None:
                            results.append(res)

    fieldnames = [
        "N",
        "d",
        "m",
        "path_kind",
        "operation",
        "language",
        "library",
        "method",
        "path_type",
        "t_ms",
        "alloc_KiB",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("=" * 60)
    print(f"Results written to: {csv_path}")

    # Only write summary in standalone mode
    if run_dir is not None:
        from common import finalize_run_folder
        summary = {
            "total_benchmarks": len(results),
            "libraries": sorted(list(set(r["library"] for r in results))),
            "operations": sorted(list(set(r["operation"] for r in results))),
            "output_csv": str(csv_path.name),
        }
        finalize_run_folder(run_dir, summary)

    return csv_path


if __name__ == "__main__":
    run_bench()
