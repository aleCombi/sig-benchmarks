"""Orchestrator for signature benchmark suite"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


# Get script directory (src/)
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

# Configuration paths
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_SWEEP_CONFIG = CONFIG_DIR / "benchmark_sweep.yaml"
REGISTRY_CONFIG = CONFIG_DIR / "libraries_registry.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file"""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_run_folder(runs_dir: Path) -> Path:
    """Create timestamped run folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_dir}")
    return run_dir


def run_python_adapter(
    library_name: str,
    library_config: Dict[str, Any],
    task_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a Python adapter using uv with dependency injection.

    Args:
        library_name: Name of the library
        library_config: Library configuration from registry
        task_config: Task parameters (N, d, m, etc.)

    Returns:
        Benchmark result dictionary
    """
    script_path = REPO_ROOT / library_config["script"]
    deps = library_config.get("deps", [])

    # Build uv command with dependency injection
    # uv run --with <dep1> --with <dep2> <script> '<json_config>'
    # Note: Adapters add src/ to sys.path themselves, so no need to inject common
    cmd = ["uv", "run"]

    # Add dependencies
    for dep in deps:
        cmd.extend(["--with", dep])

    # Script and config
    cmd.append("python")
    cmd.append(str(script_path))
    cmd.append(json.dumps(task_config))

    # Execute
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse JSON output from stdout
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            line = line.strip()
            if line.startswith('{'):
                return json.loads(line)

        raise RuntimeError(f"No JSON output from {library_name}")

    except subprocess.CalledProcessError as e:
        print(f"Error running {library_name}:", file=sys.stderr)
        print(f"  stdout: {e.stdout}", file=sys.stderr)
        print(f"  stderr: {e.stderr}", file=sys.stderr)
        raise


def run_julia_adapter(
    library_name: str,
    library_config: Dict[str, Any],
    task_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a Julia adapter with project environment.

    Args:
        library_name: Name of the library
        library_config: Library configuration from registry
        task_config: Task parameters (N, d, m, etc.)

    Returns:
        Benchmark result dictionary
    """
    julia_dir = REPO_ROOT / library_config["dir"]
    script = julia_dir / library_config["script"]

    # Build Julia command
    # JULIA_PROJECT=<dir> julia <script> '<json_config>'
    env = os.environ.copy()
    env["JULIA_PROJECT"] = str(julia_dir)

    cmd = [
        "julia",
        str(script),
        json.dumps(task_config)
    ]

    # Execute
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

        # Parse JSON output from stdout
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            line = line.strip()
            if line.startswith('{'):
                return json.loads(line)

        raise RuntimeError(f"No JSON output from {library_name}")

    except subprocess.CalledProcessError as e:
        print(f"Error running {library_name}:", file=sys.stderr)
        print(f"  stdout: {e.stdout}", file=sys.stderr)
        print(f"  stderr: {e.stderr}", file=sys.stderr)
        raise


def run_orchestrator(config_path: Path = None):
    """
    Main orchestrator logic

    Args:
        config_path: Optional path to benchmark sweep config (default: config/benchmark_sweep.yaml)
    """
    # Determine which sweep config to use
    sweep_config = config_path if config_path else DEFAULT_SWEEP_CONFIG

    # Load configurations
    sweep = load_yaml(sweep_config)
    registry = load_yaml(REGISTRY_CONFIG)

    # Extract sweep parameters
    Ns = sweep.get("Ns", [200, 400, 800])
    Ds = sweep.get("Ds", [2, 5, 7])
    Ms = sweep.get("Ms", [2, 3, 4])
    path_kind = sweep.get("path_kind", "sin")
    operations = sweep.get("operations", ["signature", "logsignature"])
    repeats = sweep.get("repeats", 10)
    runs_dir = REPO_ROOT / sweep.get("runs_dir", "runs")

    # Setup run folder
    run_dir = setup_run_folder(runs_dir)

    # Save configuration snapshot
    (run_dir / "benchmark_sweep.yaml").write_text(
        sweep_config.read_text(encoding="utf-8"),
        encoding="utf-8"
    )
    (run_dir / "libraries_registry.yaml").write_text(
        REGISTRY_CONFIG.read_text(encoding="utf-8"),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("Signature Benchmark Orchestrator")
    print("=" * 60)
    print(f"Path kind: {path_kind}")
    print(f"Ns: {Ns}")
    print(f"Ds: {Ds}")
    print(f"Ms: {Ms}")
    print(f"Operations: {operations}")
    print(f"Repeats: {repeats}")
    print(f"Libraries: {list(registry.get('libraries', {}).keys())}")
    print("=" * 60)

    # Collect results
    all_results: List[Dict[str, Any]] = []
    libraries = registry.get("libraries", {})

    # Run benchmarks
    total_tasks = len(Ns) * len(Ds) * len(Ms) * len(operations) * len(libraries)
    current_task = 0

    for library_name, library_config in libraries.items():
        lib_operations = set(library_config.get("operations", []))

        for N in Ns:
            for d in Ds:
                for m in Ms:
                    for operation in operations:
                        current_task += 1

                        # Skip if library doesn't support this operation
                        if operation not in lib_operations:
                            print(f"[{current_task}/{total_tasks}] Skipping {library_name}.{operation} (not supported)")
                            continue

                        print(f"[{current_task}/{total_tasks}] Running {library_name}.{operation} (N={N}, d={d}, m={m})")

                        # Prepare task configuration
                        task_config = {
                            "N": N,
                            "d": d,
                            "m": m,
                            "path_kind": path_kind,
                            "operation": operation,
                            "repeats": repeats,
                        }

                        # Run adapter based on type
                        try:
                            if library_config["type"] == "python":
                                result = run_python_adapter(library_name, library_config, task_config)
                            elif library_config["type"] == "julia":
                                result = run_julia_adapter(library_name, library_config, task_config)
                            else:
                                print(f"Unknown library type: {library_config['type']}", file=sys.stderr)
                                continue

                            all_results.append(result)

                        except Exception as e:
                            print(f"Failed: {e}", file=sys.stderr)
                            continue

    # Write results to CSV
    csv_path = run_dir / "results.csv"
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
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print("\n" + "=" * 60)
    print(f"Benchmark complete!")
    print(f"Results written to: {csv_path}")
    print(f"Total benchmarks: {len(all_results)}")
    print("=" * 60)

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signature benchmark orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  uv run src/orchestrator.py

  # Run with custom config
  uv run src/orchestrator.py config/smoke.yaml

  # Run smoke test
  uv run src/orchestrator.py config/smoke.yaml
        """
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=None,
        help="Path to benchmark sweep config (default: config/benchmark_sweep.yaml)"
    )

    args = parser.parse_args()
    run_orchestrator(config_path=args.config)
