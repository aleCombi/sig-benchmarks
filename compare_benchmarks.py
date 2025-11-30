# compare_benchmarks.py
# Orchestrator script that runs the Python benchmark
# (iisignature, pysiglib, chen-signatures, etc. as per config)
# and generates summary CSV + performance plots.

import csv
import subprocess
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt

# Import shared utilities
from common import (
    load_config,
    SCRIPT_DIR,
    CONFIG_PATH,
    setup_run_folder,
    finalize_run_folder,
)

PYPROJECT = SCRIPT_DIR / "pyproject.toml"

# -------- uv project bootstrap --------

def ensure_uv_project(libraries):
    """
    Ensure a uv project exists and that required Python deps are installed.

    `libraries` should be the list from the config (e.g. ["chen-signatures", "iisignature", "pysiglib"]).
    We always add numpy + matplotlib as shared deps.
    """
    if not PYPROJECT.exists():
        print("No pyproject.toml found, initializing uv project in", SCRIPT_DIR)
        subprocess.run(
            ["uv", "init", "."],
            cwd=SCRIPT_DIR,
            check=True,
        )

    base_deps = ["numpy", "matplotlib"]
    # Avoid duplicates while preserving a stable order
    all_deps = base_deps + [lib for lib in libraries if lib not in base_deps]

    print(f"Ensuring Python deps via uv add ({', '.join(all_deps)})...")
    subprocess.run(
        ["uv", "add", *all_deps],
        cwd=SCRIPT_DIR,
        check=True,
    )

# -------- run Python benchmark --------

def run_python_benchmark(run_dir: Path, base_env: dict) -> Path:
    print("\n" + "=" * 60)
    print("Running Python Benchmark")
    print("=" * 60)

    env = base_env.copy()
    env["BENCHMARK_OUT_CSV"] = str(run_dir / "python_results.csv")

    result = subprocess.run(
        ["uv", "run", "benchmark.py"],
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
        env=env,
    )

    # Save logs
    (run_dir / "python_stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "python_stderr.log").write_text(result.stderr, encoding="utf-8")

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Python benchmark failed with code {result.returncode}")

    python_csv = run_dir / "python_results.csv"
    if not python_csv.exists():
        raise RuntimeError(f"Expected Python output not found: {python_csv}")
    
    print(f"✓ Python results: {python_csv.name}")
    return python_csv

# -------- loading + plotting --------

def load_python_rows(python_csv: Path):
    rows = []
    with python_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "N": int(row["N"]),
                "d": int(row["d"]),
                "m": int(row["m"]),
                "path_kind": row["path_kind"].strip(),
                "operation": row["operation"].strip(),
                "library": row["library"].strip(),
                "path_type": row.get("path_type", "").strip(),
                "t_ms": float(row["t_ms"]),
                "alloc_KiB": float(row["alloc_KiB"]),
            })
    return rows

def get_time(rows, lib, N, d, m, path_kind, operation):
    for r in rows:
        if (
            r["N"] == N and
            r["d"] == d and
            r["m"] == m and
            r["path_kind"] == path_kind and
            r["operation"] == operation and
            r["library"] == lib
        ):
            return r["t_ms"]
    return None

def make_plots(python_csv: Path, runs_dir: Path, cfg: dict):
    print("=== Making Python comparison plots ===")
    rows = load_python_rows(python_csv)

    if not rows:
        print("No rows found in Python CSV; skipping plots.")
        return

    # config-derived grids
    Ns = sorted(cfg.get("Ns", []))
    Ds = sorted(cfg.get("Ds", []))
    Ms = sorted(cfg.get("Ms", []))

    if not Ns:
        Ns = sorted({r["N"] for r in rows})
    if not Ds:
        Ds = sorted({r["d"] for r in rows})
    if not Ms:
        Ms = sorted({r["m"] for r in rows})

    # pick "max" for fixed params (worst-case scaling)
    N_fixed_for_d = max(Ns)
    N_fixed_for_m = max(Ns)

    d_fixed_for_N = max(Ds)
    d_fixed_for_m = max(Ds)

    m_fixed_for_N = max(Ms)
    m_fixed_for_d = max(Ms)

    path_kind = cfg.get("path_kind", "sin")
    operations = cfg.get("operations", ["signature", "logsignature"])

    cfg_libraries = cfg.get("libraries", [])
    # Only plot libraries that are both in the CSV and requested in config
    csv_libs = sorted({r["library"] for r in rows})
    python_libs = [lib for lib in csv_libs if not cfg_libraries or lib in cfg_libraries]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharey="col")

    op_order = ["signature", "logsignature"]
    for row_idx, vary in enumerate(["N", "d", "m"]):
        for col_idx, op in enumerate(op_order):
            ax = axes[row_idx, col_idx]

            if op not in operations:
                ax.set_visible(False)
                continue

            # Determine x-grid and fixed params for this subplot
            if vary == "N":
                xs = Ns
                d_fix = d_fixed_for_N
                m_fix = m_fixed_for_N
                xlabel = "N (number of points)"
            elif vary == "d":
                xs = Ds
                d_fix = None
                m_fix = m_fixed_for_d
                xlabel = "d (dimension)"
            else:  # vary m
                xs = Ms
                d_fix = d_fixed_for_m
                m_fix = None
                xlabel = "m (signature level)"

            # Plot each Python library
            for lib in python_libs:
                ys = []
                xs_effective = []

                for x in xs:
                    if vary == "N":
                        N = x
                        d = d_fix
                        m = m_fix
                    elif vary == "d":
                        N = N_fixed_for_d
                        d = x
                        m = m_fix
                    else:  # vary m
                        N = N_fixed_for_m
                        d = d_fix
                        m = x

                    t = get_time(rows, lib, N, d, m, path_kind, op)
                    if t is not None and t > 0.0:
                        xs_effective.append(x)
                        ys.append(t)

                if len(xs_effective) >= 2:
                    ax.plot(xs_effective, ys, marker="o", label=lib)

            # ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("time (ms)")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            title = f"{op}, vary {vary}"
            if vary == "N":
                title += f" (d={d_fixed_for_N}, m={m_fixed_for_N})"
            elif vary == "d":
                title += f" (N={N_fixed_for_d}, m={m_fixed_for_d})"
            else:
                title += f" (N={N_fixed_for_m}, d={d_fixed_for_m})"
            ax.set_title(title)

            if row_idx == 0:
                ax.legend()

    fig.tight_layout()
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_plot = runs_dir / "python_comparison_3x2.png"
    fig.savefig(out_plot, dpi=300)
    print(f"Plots written to: {out_plot}")

# -------- main --------

def main():
    cfg = load_config(CONFIG_PATH)
    
    print("=" * 60)
    print("Python-only Benchmark Comparison (Chen vs others)")
    print("=" * 60)
    print("Configuration:")
    for k, v in sorted(cfg.items()):
        print(f"  {k:15s} = {v}")
    print()

    # Setup run folder
    run_dir = setup_run_folder("benchmark_python_only", cfg)
    base_env = os.environ.copy()

    # Use list of libraries from config
    libraries = cfg.get("libraries", [])
    ensure_uv_project(libraries)

    python_csv = run_python_benchmark(run_dir, base_env)
    make_plots(python_csv, run_dir, cfg)
    
    # Build summary
    rows = load_python_rows(python_csv)
    csv_libs = sorted({r["library"] for r in rows})
    cfg_libraries = cfg.get("libraries", [])
    libs = [lib for lib in csv_libs if not cfg_libraries or lib in cfg_libraries]

    summary = {
        "python_csv": python_csv.name,
        "plots": "python_comparison_3x2.png",
        "libraries": libs,
    }

    # Simple performance summary: avg time per lib (all configs)
    times_by_lib = {lib: [] for lib in libs}
    for r in rows:
        if r["library"] in times_by_lib:
            times_by_lib[r["library"]].append(r["t_ms"])

    summary["avg_time_ms"] = {
        lib: (sum(ts) / len(ts)) if ts else None
        for lib, ts in times_by_lib.items()
    }

    finalize_run_folder(run_dir, summary)
    print(f"\n✓ Python-only benchmark complete: {run_dir}")

if __name__ == "__main__":
    main()
