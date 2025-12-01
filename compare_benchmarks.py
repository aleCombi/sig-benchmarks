# compare_benchmarks.py
# Orchestrator that runs each tool separately (Julia + per Python lib),
# stitches all rows into a single CSV, and produces comparison plots.

import csv
import subprocess
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import (
    load_config,
    SCRIPT_DIR,
    CONFIG_PATH,
    setup_run_folder,
    finalize_run_folder,
)

PYPROJECT = SCRIPT_DIR / "pyproject.toml"


# -------- uv project bootstrap --------

def ensure_uv_project(libraries, operations):
    """
    Ensure a uv project exists and that required Python deps are installed.

    `libraries` is the list from the config (e.g. ["chen-signatures", "iisignature", "pysiglib"]).
    We always add numpy + matplotlib as shared deps.
    If "sigdiff" is in operations and chen-signatures is requested, we also add torch.
    """
    if not PYPROJECT.exists():
        print("No pyproject.toml found, initializing uv project in", SCRIPT_DIR)
        subprocess.run(
            ["uv", "init", "."],
            cwd=SCRIPT_DIR,
            check=True,
        )

    base_deps = ["numpy", "matplotlib"]

    # If sigdiff is requested and chen-signatures is in play, we will need torch
    if "sigdiff" in operations and any(lib == "chen-signatures" for lib in libraries):
        base_deps.append("torch")

    # Avoid duplicates while preserving a stable order
    all_deps = base_deps + [lib for lib in libraries if lib not in base_deps]

    print(f"Ensuring Python deps via uv add ({', '.join(all_deps)})...")
    subprocess.run(
        ["uv", "add", *all_deps],
        cwd=SCRIPT_DIR,
        check=True,
    )


# -------- run Julia benchmark --------

def run_julia_benchmark(run_dir: Path) -> Path:
    print("\n" + "=" * 60)
    print("Running Julia Benchmark")
    print("=" * 60)

    env = os.environ.copy()
    env["BENCHMARK_OUT_CSV"] = str(run_dir / "julia_results.csv")

    result = subprocess.run(
        ["julia", "--project=.", "benchmark.jl"],
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
        env=env,
    )

    (run_dir / "julia_stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "julia_stderr.log").write_text(result.stderr, encoding="utf-8")

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Julia benchmark failed with code {result.returncode}")

    julia_csv = run_dir / "julia_results.csv"
    if not julia_csv.exists():
        raise RuntimeError(f"Expected Julia output not found: {julia_csv}")

    print(f"-> Julia results: {julia_csv.name}")
    return julia_csv


# -------- run per-library Python benchmark --------

def run_python_library(lib: str, run_dir: Path, base_env: dict) -> Path:
    print("\n" + "=" * 60)
    print(f"Running Python Benchmark for {lib}")
    print("=" * 60)

    env = base_env.copy()
    env["BENCHMARK_OUT_CSV"] = str(run_dir / f"{lib}_results.csv")
    env["BENCHMARK_LIBRARIES"] = lib

    result = subprocess.run(
        ["uv", "run", "benchmark.py"],
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
        env=env,
    )

    stdout_path = run_dir / f"{lib}_stdout.log"
    stderr_path = run_dir / f"{lib}_stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{lib} benchmark failed with code {result.returncode}")

    csv_path = run_dir / f"{lib}_results.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Expected output not found for {lib}: {csv_path}")

    print(f"-> {lib} results: {csv_path.name}")
    return csv_path


# -------- loading + helper --------

def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "N": int(row["N"]),
                "d": int(row["d"]),
                "m": int(row["m"]),
                "path_kind": row["path_kind"].strip(),
                "operation": row["operation"].strip(),
                "language": row.get("language", "").strip(),
                "library": row["library"].strip(),
                "method": row.get("method", "").strip(),
                "path_type": row.get("path_type", "").strip(),
                "t_ms": float(row["t_ms"]),
                "alloc_KiB": float(row["alloc_KiB"]),
            })
    return rows


def write_combined_csv(rows, csv_path: Path):
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
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


# -------- plotting --------

def make_plots(rows, runs_dir: Path, cfg: dict):
    print("=== Making comparison plots ===")

    if not rows:
        print("No rows found; skipping plots.")
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
    operations_cfg = cfg.get("operations", ["signature", "logsignature"])
    cfg_libraries = cfg.get("libraries", [])
    cfg_julia_libs = cfg.get("julia_libraries", [])

    # Only plot libraries that are both in the data and requested in config (if provided)
    csv_libs = sorted({r["library"] for r in rows})
    requested_libs = (cfg_libraries or []) + (cfg_julia_libs or [])
    if requested_libs:
        libs_for_plot = [lib for lib in csv_libs if lib in requested_libs]
    else:
        libs_for_plot = csv_libs

    # We support up to 3 operations: signature, logsignature, sigdiff
    op_order = ["signature", "logsignature", "sigdiff"]

    # Set up a 3x3 grid: rows = vary N/d/m, columns = ops
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey="col")

    for row_idx, vary in enumerate(["N", "d", "m"]):
        for col_idx, op in enumerate(op_order):
            ax = axes[row_idx, col_idx]

            # If this operation is not requested in config or no data, hide axis.
            if op not in operations_cfg:
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

            plotted_any = False
            for lib in libs_for_plot:
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
                    plotted_any = True

            if not plotted_any:
                ax.set_visible(False)
                continue

            # X axis: integer ticks
            ax.set_xlabel(xlabel)
            if xs:
                ax.set_xticks(xs)

            # Slightly denser y-axis ticks
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

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
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=8)

    fig.tight_layout()
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_plot = runs_dir / "comparison_3x3.png"
    fig.savefig(out_plot, dpi=300)
    print(f"Plots written to: {out_plot}")


# -------- main --------

def main():
    cfg = load_config(CONFIG_PATH)
    
    print("=" * 60)
    print("Tool-by-tool Benchmark Comparison")
    print("=" * 60)
    print("Configuration:")
    for k, v in sorted(cfg.items()):
        print(f"  {k:15s} = {v}")
    print()

    # Setup run folder
    run_dir = setup_run_folder("benchmark_comparison", cfg)
    base_env = os.environ.copy()

    # Lists from config
    libraries = cfg.get("libraries", []) or []
    julia_libraries = cfg.get("julia_libraries", []) or []
    operations = cfg.get("operations", ["signature", "logsignature"])

    ensure_uv_project(libraries, operations)

    all_rows = []
    produced_csvs = {}

    # Julia (one project with ChenSignatures.jl)
    if julia_libraries:
        julia_csv = run_julia_benchmark(run_dir)
        produced_csvs["julia"] = julia_csv.name
        all_rows.extend(load_rows(julia_csv))

    # Python libs one by one
    for lib in libraries:
        csv_path = run_python_library(lib, run_dir, base_env)
        produced_csvs[lib] = csv_path.name
        all_rows.extend(load_rows(csv_path))

    combined_csv = run_dir / "combined_results.csv"
    write_combined_csv(all_rows, combined_csv)

    make_plots(all_rows, run_dir, cfg)
    
    # Summary
    libs_present = sorted({r["library"] for r in all_rows})
    times_by_lib = {lib: [] for lib in libs_present}
    for r in all_rows:
        times_by_lib[r["library"]].append(r["t_ms"])

    summary = {
        "combined_csv": combined_csv.name,
        "plots": "comparison_3x3.png",
        "libraries": libs_present,
        "per_tool_csvs": produced_csvs,
        "avg_time_ms": {
            lib: (sum(ts) / len(ts)) if ts else None
            for lib, ts in times_by_lib.items()
        },
    }

    finalize_run_folder(run_dir, summary)
    print(f"\n-> Benchmark complete: {run_dir}")


if __name__ == "__main__":
    main()
