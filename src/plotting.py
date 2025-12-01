"""Plotting utilities for signature benchmarks"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def load_results(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Load benchmark results from CSV file.

    Args:
        csv_path: Path to results CSV file

    Returns:
        List of result dictionaries
    """
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
            })
    return rows


def get_time(
    rows: List[Dict[str, Any]],
    library: str,
    N: int,
    d: int,
    m: int,
    path_kind: str,
    operation: str
) -> Optional[float]:
    """
    Find timing result for specific configuration.

    Args:
        rows: List of result dictionaries
        library: Library name
        N: Number of points
        d: Dimension
        m: Signature level
        path_kind: Path type
        operation: Operation name

    Returns:
        Time in milliseconds, or None if not found
    """
    for r in rows:
        if (
            r["N"] == N
            and r["d"] == d
            and r["m"] == m
            and r["path_kind"] == path_kind
            and r["operation"] == operation
            and r["library"] == library
        ):
            return r["t_ms"]
    return None


def get_latest_run(runs_dir: Path = Path("runs")) -> Optional[Path]:
    """
    Find the most recent benchmark run directory.

    Args:
        runs_dir: Directory containing benchmark runs

    Returns:
        Path to latest run directory, or None if no runs found
    """
    if not runs_dir.exists():
        return None

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("benchmark_")]
    if not run_dirs:
        return None

    # Sort by directory name (which includes timestamp)
    return sorted(run_dirs)[-1]


def make_line_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate 3x3 line plot comparison grid (original visualization).

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path (defaults to same dir as CSV)
        config: Optional configuration dict with sweep parameters

    Returns:
        Path to saved plot
    """
    rows = load_results(csv_path)

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    # Derive grid parameters from data or config
    if config:
        Ns = sorted(config.get("Ns", []))
        Ds = sorted(config.get("Ds", []))
        Ms = sorted(config.get("Ms", []))
        path_kind = config.get("path_kind", "sin")
        operations_cfg = config.get("operations", ["signature", "logsignature"])
    else:
        Ns = sorted(set(r["N"] for r in rows))
        Ds = sorted(set(r["d"] for r in rows))
        Ms = sorted(set(r["m"] for r in rows))
        path_kind = rows[0]["path_kind"]
        operations_cfg = sorted(set(r["operation"] for r in rows))

    # Fixed parameters for each subplot (use max for worst-case scaling)
    N_fixed_for_d = max(Ns)
    N_fixed_for_m = max(Ns)
    d_fixed_for_N = max(Ds)
    d_fixed_for_m = max(Ds)
    m_fixed_for_N = max(Ms)
    m_fixed_for_d = max(Ms)

    # Libraries present in data
    libraries = sorted(set(r["library"] for r in rows))

    # Operation order for columns
    op_order = ["signature", "logsignature", "sigdiff"]

    # Create 3x3 grid: rows = vary N/d/m, columns = operations
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey="col")

    for row_idx, vary in enumerate(["N", "d", "m"]):
        for col_idx, op in enumerate(op_order):
            ax = axes[row_idx, col_idx]

            # Hide subplot if operation not in config
            if op not in operations_cfg:
                ax.set_visible(False)
                continue

            # Determine x-axis values and fixed parameters
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

            # Plot each library
            for lib in libraries:
                ys = []
                xs_effective = []

                for x in xs:
                    if vary == "N":
                        N, d, m = x, d_fix, m_fix
                    elif vary == "d":
                        N, d, m = N_fixed_for_d, x, m_fix
                    else:  # vary m
                        N, d, m = N_fixed_for_m, d_fix, x

                    t = get_time(rows, lib, N, d, m, path_kind, op)
                    if t is not None and t > 0.0:
                        xs_effective.append(x)
                        ys.append(t)

                # Only plot if we have at least 2 points
                if len(xs_effective) >= 2:
                    ax.plot(xs_effective, ys, marker="o", label=lib)
                    plotted_any = True

            # Hide subplot if no data plotted
            if not plotted_any:
                ax.set_visible(False)
                continue

            # Configure axes
            ax.set_xlabel(xlabel)
            if xs:
                ax.set_xticks(xs)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.set_ylabel("time (ms)")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            # Title with fixed parameters
            title = f"{op}, vary {vary}"
            if vary == "N":
                title += f" (d={d_fixed_for_N}, m={m_fixed_for_N})"
            elif vary == "d":
                title += f" (N={N_fixed_for_d}, m={m_fixed_for_d})"
            else:
                title += f" (N={N_fixed_for_m}, d={d_fixed_for_m})"
            ax.set_title(title)

            # Legend only on top row
            if row_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=8)

    fig.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / "plot_line.png"

    fig.savefig(output_path, dpi=300)
    print(f"Line plot saved to: {output_path}")
    plt.close(fig)

    return output_path


def make_heatmap_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate heatmap showing performance across all parameter combinations.

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path (defaults to same dir as CSV)
        config: Optional configuration dict

    Returns:
        Path to saved plot
    """
    rows = load_results(csv_path)

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    # Get unique values
    operations = sorted(set(r["operation"] for r in rows))
    libraries = sorted(set(r["library"] for r in rows))

    # Create a heatmap for each operation
    num_ops = len(operations)
    fig, axes = plt.subplots(1, num_ops, figsize=(8 * num_ops, 10))
    if num_ops == 1:
        axes = [axes]

    for op_idx, operation in enumerate(operations):
        ax = axes[op_idx]

        # Get all unique parameter combinations for this operation
        op_rows = [r for r in rows if r["operation"] == operation]
        if not op_rows:
            ax.set_visible(False)
            continue

        # Create unique combinations as row labels
        params = sorted(set((r["N"], r["d"], r["m"]) for r in op_rows))
        param_labels = [f"N={n}, d={d}, m={m}" for n, d, m in params]

        # Build matrix: rows = parameter combos, columns = libraries
        matrix = np.full((len(params), len(libraries)), np.nan)

        for row_idx, (N, d, m) in enumerate(params):
            for col_idx, lib in enumerate(libraries):
                # Find timing for this combination
                for r in op_rows:
                    if r["N"] == N and r["d"] == d and r["m"] == m and r["library"] == lib:
                        matrix[row_idx, col_idx] = r["t_ms"]
                        break

        # Plot heatmap with log scale
        im = ax.imshow(np.log10(matrix + 1e-6), aspect="auto", cmap="viridis", interpolation="nearest")

        # Configure axes
        ax.set_xticks(np.arange(len(libraries)))
        ax.set_yticks(np.arange(len(params)))
        ax.set_xticklabels(libraries, rotation=45, ha="right")
        ax.set_yticklabels(param_labels, fontsize=8)

        ax.set_xlabel("Library")
        ax.set_ylabel("Parameters")
        ax.set_title(f"{operation} - Performance (log10 ms)")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("log10(time in ms)")

        # Add text annotations with actual values
        for i in range(len(params)):
            for j in range(len(libraries)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f"{matrix[i, j]:.1f}",
                                 ha="center", va="center", color="white", fontsize=6)

    fig.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / "plot_heatmap.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap plot saved to: {output_path}")
    plt.close(fig)

    return output_path


def make_speedup_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    baseline: str = "slowest"
) -> Path:
    """
    Generate speedup plot showing relative performance (same layout as line plot).

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path
        config: Optional configuration dict
        baseline: Baseline for speedup calculation ("slowest", "fastest", or library name)

    Returns:
        Path to saved plot
    """
    rows = load_results(csv_path)

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    # Derive grid parameters from data or config
    if config:
        Ns = sorted(config.get("Ns", []))
        Ds = sorted(config.get("Ds", []))
        Ms = sorted(config.get("Ms", []))
        path_kind = config.get("path_kind", "sin")
        operations_cfg = config.get("operations", ["signature", "logsignature"])
    else:
        Ns = sorted(set(r["N"] for r in rows))
        Ds = sorted(set(r["d"] for r in rows))
        Ms = sorted(set(r["m"] for r in rows))
        path_kind = rows[0]["path_kind"]
        operations_cfg = sorted(set(r["operation"] for r in rows))

    # Fixed parameters for each subplot
    N_fixed_for_d = max(Ns)
    N_fixed_for_m = max(Ns)
    d_fixed_for_N = max(Ds)
    d_fixed_for_m = max(Ds)
    m_fixed_for_N = max(Ms)
    m_fixed_for_d = max(Ms)

    # Libraries present in data
    libraries = sorted(set(r["library"] for r in rows))

    # Operation order for columns
    op_order = ["signature", "logsignature", "sigdiff"]

    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for row_idx, vary in enumerate(["N", "d", "m"]):
        for col_idx, op in enumerate(op_order):
            ax = axes[row_idx, col_idx]

            if op not in operations_cfg:
                ax.set_visible(False)
                continue

            # Determine x-axis values and fixed parameters
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

            # Calculate speedups
            for lib in libraries:
                speedups = []
                xs_effective = []

                for x in xs:
                    if vary == "N":
                        N, d, m = x, d_fix, m_fix
                    elif vary == "d":
                        N, d, m = N_fixed_for_d, x, m_fix
                    else:  # vary m
                        N, d, m = N_fixed_for_m, d_fix, x

                    # Get times for all libraries at this point
                    times = {}
                    for lib_name in libraries:
                        t = get_time(rows, lib_name, N, d, m, path_kind, op)
                        if t is not None and t > 0.0:
                            times[lib_name] = t

                    if not times:
                        continue

                    # Calculate baseline
                    if baseline == "slowest":
                        baseline_time = max(times.values())
                    elif baseline == "fastest":
                        baseline_time = min(times.values())
                    elif baseline in times:
                        baseline_time = times[baseline]
                    else:
                        baseline_time = max(times.values())  # fallback

                    # Calculate speedup for this library
                    if lib in times:
                        speedup = baseline_time / times[lib]
                        xs_effective.append(x)
                        speedups.append(speedup)

                if len(xs_effective) >= 2:
                    ax.plot(xs_effective, speedups, marker="o", label=lib)
                    plotted_any = True

            if not plotted_any:
                ax.set_visible(False)
                continue

            # Add reference line at speedup=1.0
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

            # Configure axes
            ax.set_xlabel(xlabel)
            if xs:
                ax.set_xticks(xs)

            ax.set_ylabel(f"Speedup vs {baseline}")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            # Title
            title = f"{op}, vary {vary}"
            if vary == "N":
                title += f" (d={d_fixed_for_N}, m={m_fixed_for_N})"
            elif vary == "d":
                title += f" (N={N_fixed_for_d}, m={m_fixed_for_d})"
            else:
                title += f" (N={N_fixed_for_m}, d={d_fixed_for_m})"
            ax.set_title(title)

            # Legend only on top row
            if row_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=8)

    fig.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / f"plot_speedup_{baseline}.png"

    fig.savefig(output_path, dpi=300)
    print(f"Speedup plot saved to: {output_path}")
    plt.close(fig)

    return output_path


def make_profile_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate performance profile plot showing how often each library is competitive.

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path
        config: Optional configuration dict

    Returns:
        Path to saved plot
    """
    rows = load_results(csv_path)

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    # Get unique operations and libraries
    operations = sorted(set(r["operation"] for r in rows))
    libraries = sorted(set(r["library"] for r in rows))

    # Group by unique benchmark (N, d, m, operation combination)
    benchmarks = {}
    for r in rows:
        key = (r["N"], r["d"], r["m"], r["operation"])
        if key not in benchmarks:
            benchmarks[key] = {}
        benchmarks[key][r["library"]] = r["t_ms"]

    # Calculate performance ratios for each benchmark
    num_ops = len(operations)
    fig, axes = plt.subplots(1, num_ops, figsize=(8 * num_ops, 6))
    if num_ops == 1:
        axes = [axes]

    for op_idx, operation in enumerate(operations):
        ax = axes[op_idx]

        # Filter benchmarks for this operation
        op_benchmarks = {k: v for k, v in benchmarks.items() if k[3] == operation}

        if not op_benchmarks:
            ax.set_visible(False)
            continue

        # For each library, calculate ratio to best time for each benchmark
        library_ratios = {lib: [] for lib in libraries}

        for bench_key, times in op_benchmarks.items():
            if not times:
                continue

            best_time = min(times.values())

            for lib in libraries:
                if lib in times:
                    ratio = times[lib] / best_time
                    library_ratios[lib].append(ratio)

        # Sort ratios and plot performance profile
        for lib in libraries:
            if not library_ratios[lib]:
                continue

            ratios = sorted(library_ratios[lib])
            # Y-axis: fraction of benchmarks where ratio <= x
            y_values = np.arange(1, len(ratios) + 1) / len(ratios)

            ax.plot(ratios, y_values, marker="o", markersize=4, label=lib)

        ax.set_xlabel("Performance ratio (time / best_time)")
        ax.set_ylabel("Fraction of benchmarks")
        ax.set_title(f"{operation} - Performance Profile")
        ax.set_xlim(left=1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add vertical line at ratio=2 (2x slower than best)
        ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    fig.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / "plot_profile.png"

    fig.savefig(output_path, dpi=300)
    print(f"Performance profile plot saved to: {output_path}")
    plt.close(fig)

    return output_path


def make_box_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate box plots showing distribution of performance across all benchmarks.

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path
        config: Optional configuration dict

    Returns:
        Path to saved plot
    """
    rows = load_results(csv_path)

    if not rows:
        raise ValueError("No benchmark results found in CSV")

    # Get unique operations and libraries
    operations = sorted(set(r["operation"] for r in rows))
    libraries = sorted(set(r["library"] for r in rows))

    # Create subplots for each operation
    num_ops = len(operations)
    fig, axes = plt.subplots(1, num_ops, figsize=(6 * num_ops, 6))
    if num_ops == 1:
        axes = [axes]

    for op_idx, operation in enumerate(operations):
        ax = axes[op_idx]

        # Get data for this operation
        op_rows = [r for r in rows if r["operation"] == operation]

        if not op_rows:
            ax.set_visible(False)
            continue

        # Organize data by library
        data_by_lib = {lib: [] for lib in libraries}
        for r in op_rows:
            data_by_lib[r["library"]].append(r["t_ms"])

        # Filter out libraries with no data
        plot_data = [data_by_lib[lib] for lib in libraries if data_by_lib[lib]]
        plot_labels = [lib for lib in libraries if data_by_lib[lib]]

        if not plot_data:
            ax.set_visible(False)
            continue

        # Create box plot
        bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{operation} - Distribution Across All Benchmarks")
        ax.grid(True, axis="y", alpha=0.3)

        # Rotate x labels if needed
        if len(plot_labels) > 3:
            ax.set_xticklabels(plot_labels, rotation=45, ha="right")

        # Use log scale if data spans multiple orders of magnitude
        if plot_data:
            all_vals = [v for lib_data in plot_data for v in lib_data]
            if max(all_vals) / min(all_vals) > 100:
                ax.set_yscale("log")

    fig.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / "plot_box.png"

    fig.savefig(output_path, dpi=300)
    print(f"Box plot saved to: {output_path}")
    plt.close(fig)

    return output_path


def make_comparison_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Legacy wrapper for make_line_plot (for backwards compatibility).

    Args:
        csv_path: Path to results CSV
        output_path: Optional output path
        config: Optional configuration dict

    Returns:
        Path to saved plot
    """
    if output_path is None:
        output_path = csv_path.parent / "comparison_3x3.png"
    return make_line_plot(csv_path, output_path, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from signature benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots for latest run
  python plotting.py --plot-type all

  # Generate heatmap for specific run
  python plotting.py runs/benchmark_20251201_221922 --plot-type heatmap

  # Generate speedup plot with custom baseline
  python plotting.py --plot-type speedup --baseline iisignature

  # List available plot types
  python plotting.py --list-plots
        """
    )

    parser.add_argument(
        "run_dir",
        nargs="?",
        type=str,
        help="Path to benchmark run directory or results.csv (defaults to latest run)"
    )

    parser.add_argument(
        "--plot-type",
        "-t",
        type=str,
        default="all",
        choices=["line", "heatmap", "speedup", "profile", "box", "all"],
        help="Type of plot to generate (default: all)"
    )

    parser.add_argument(
        "--baseline",
        "-b",
        type=str,
        default="slowest",
        help="Baseline for speedup plot: 'slowest', 'fastest', or library name (default: slowest)"
    )

    parser.add_argument(
        "--list-plots",
        "-l",
        action="store_true",
        help="List available plot types and exit"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for plots (defaults to run directory)"
    )

    args = parser.parse_args()

    # List plot types if requested
    if args.list_plots:
        print("Available plot types:")
        print("  line     - 3x3 grid of line plots (original)")
        print("  heatmap  - Heatmap showing all parameter combinations")
        print("  speedup  - Relative performance vs baseline")
        print("  profile  - Performance profile (competitiveness)")
        print("  box      - Box plots showing distribution")
        print("  all      - Generate all plot types")
        sys.exit(0)

    # Determine CSV path
    if args.run_dir:
        run_path = Path(args.run_dir)
        if run_path.is_file() and run_path.suffix == ".csv":
            csv_path = run_path
            run_dir = run_path.parent
        elif run_path.is_dir():
            run_dir = run_path
            csv_path = run_dir / "results.csv"
        else:
            print(f"Error: {run_path} is not a valid directory or CSV file")
            sys.exit(1)
    else:
        # Use latest run
        latest = get_latest_run()
        if not latest:
            print("Error: No benchmark runs found in 'runs/' directory")
            sys.exit(1)
        run_dir = latest
        csv_path = run_dir / "results.csv"

    # Verify CSV exists
    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        sys.exit(1)

    print(f"Loading results from: {csv_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = run_dir

    # Generate plots
    plot_funcs = {
        "line": (make_line_plot, {}),
        "heatmap": (make_heatmap_plot, {}),
        "speedup": (make_speedup_plot, {"baseline": args.baseline}),
        "profile": (make_profile_plot, {}),
        "box": (make_box_plot, {}),
    }

    if args.plot_type == "all":
        print(f"\nGenerating all plot types in: {output_dir}\n")
        for plot_name, (plot_func, kwargs) in plot_funcs.items():
            try:
                output_path = output_dir / f"plot_{plot_name}.png"
                if plot_name == "speedup":
                    output_path = output_dir / f"plot_speedup_{args.baseline}.png"
                plot_func(csv_path, output_path, **kwargs)
            except Exception as e:
                print(f"Error generating {plot_name} plot: {e}")
    else:
        plot_func, kwargs = plot_funcs[args.plot_type]
        try:
            output_path = output_dir / f"plot_{args.plot_type}.png"
            if args.plot_type == "speedup":
                output_path = output_dir / f"plot_speedup_{args.baseline}.png"
            plot_func(csv_path, output_path, **kwargs)
        except Exception as e:
            print(f"Error generating {args.plot_type} plot: {e}")
            sys.exit(1)

    print(f"\nDone! Plots saved to: {output_dir}")
