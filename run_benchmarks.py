#!/usr/bin/env python3
"""Convenience script to run benchmarks and generate plots"""

import subprocess
import sys
from pathlib import Path

# Get the repository root
REPO_ROOT = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("Running Full Benchmark Suite")
    print("=" * 60)

    # Step 1: Run orchestrator
    print("\n[1/2] Running benchmarks...")
    result = subprocess.run(
        ["uv", "run", "--with", "pyyaml", "src/orchestrator.py"],
        cwd=REPO_ROOT,
        capture_output=False,
    )

    if result.returncode != 0:
        print("Benchmark failed!", file=sys.stderr)
        sys.exit(1)

    # Find the most recent run folder
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        print("No runs directory found!", file=sys.stderr)
        sys.exit(1)

    run_folders = sorted(runs_dir.glob("benchmark_*"), key=lambda p: p.stat().st_mtime)
    if not run_folders:
        print("No benchmark runs found!", file=sys.stderr)
        sys.exit(1)

    latest_run = run_folders[-1]
    results_csv = latest_run / "results.csv"

    if not results_csv.exists():
        print(f"Results CSV not found in {latest_run}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Generate plots
    print(f"\n[2/2] Generating plots for {latest_run.name}...")
    result = subprocess.run(
        ["uv", "run", "--with", "matplotlib", "--with", "numpy", "--with", "pyyaml",
         "src/plotting.py", str(results_csv)],
        cwd=REPO_ROOT,
        capture_output=False,
    )

    if result.returncode != 0:
        print("Plotting failed!", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Results: {latest_run}")
    print(f"  - results.csv")
    print(f"  - plot_line.png")
    print(f"  - plot_heatmap.png")
    print(f"  - plot_speedup_slowest.png")
    print(f"  - plot_profile.png")
    print(f"  - plot_box.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
