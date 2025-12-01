"""Plotting utilities for signature benchmarks"""

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
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


def make_comparison_plot(
    csv_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate 3x3 comparison plot grid.

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
        output_path = csv_path.parent / "comparison_3x3.png"

    fig.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plotting.py <results.csv> [output.png]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    make_comparison_plot(csv_path, output_path)
