# common.py
"""Shared utilities for benchmark suite"""

import ast
import math
import json
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import numpy as np

# Optional dependency: PyYAML
try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

# -------- Constants --------

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "benchmark_config.yaml"

# -------- Simple YAML-like parser (fallback) --------

def load_simple_yaml(path: Path) -> Dict[str, Any]:
    """
    Very simple YAML/INI-like parser used as a fallback if PyYAML
    is not available. Supports:
      - key: "string"
      - key: [1, 2, 3]
      - key: 123
    """
    cfg: Dict[str, Any] = {}
    if not path.is_file():
        return cfg
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # strip comments
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                continue

            # quoted string
            if value.startswith('"') and value.endswith('"'):
                cfg[key] = value[1:-1]
            # list syntax
            elif value.startswith("["):
                try:
                    cfg[key] = ast.literal_eval(value)
                except Exception:
                    cfg[key] = value
            else:
                # try int
                try:
                    cfg[key] = int(value)
                except ValueError:
                    cfg[key] = value
    return cfg

# -------- Config Loading --------

def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load and parse benchmark configuration (PyYAML if available, otherwise fallback)."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if yaml is not None:
        # Preferred path: full YAML support
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        # Fallback: simple parser
        print(
            "[common] Warning: PyYAML not installed, using simple config parser. "
            "Install `pyyaml` for full YAML support.",
            flush=True,
        )
        raw = load_simple_yaml(config_path)

    Ns = raw.get("Ns", [150, 1000, 2000])
    Ds = raw.get("Ds", [2, 6, 7, 8])
    Ms = raw.get("Ms", [4, 6])
    path_kind = raw.get("path_kind", "linear")
    runs_dir = raw.get("runs_dir", "runs")
    repeats = int(raw.get("repeats", 5))
    logsig_method = raw.get("logsig_method", "O")
    operations = raw.get("operations", ["signature", "logsignature"])
    libraries = _ensure_list(raw.get("libraries", []))

    path_kind = str(path_kind).lower()
    if path_kind not in ("linear", "sin"):
        raise ValueError(f"Unknown path_kind '{path_kind}', expected 'linear' or 'sin'.")

    # Normalize operations to lowercase strings
    operations = [str(op).lower() for op in operations]

    # Normalize libraries as simple strings (pip-style names)
    libraries = [str(lib) for lib in libraries]

    return {
        "Ns": Ns,
        "Ds": Ds,
        "Ms": Ms,
        "path_kind": path_kind,
        "runs_dir": runs_dir,
        "repeats": repeats,
        "logsig_method": logsig_method,
        "operations": operations,
        "libraries": libraries,
    }

# -------- Path Generators --------

def make_path_linear(d: int, N: int) -> np.ndarray:
    """Generate linear path: [t, 2t, 2t, ...]"""
    ts = np.linspace(0.0, 1.0, N)
    path = np.empty((N, d), dtype=float)
    path[:, 0] = ts
    if d > 1:
        path[:, 1:] = 2.0 * ts[:, None]
    return path

def make_path_sin(d: int, N: int) -> np.ndarray:
    """Generate sinusoidal path: [sin(2π·1·t), sin(2π·2·t), ...]"""
    ts = np.linspace(0.0, 1.0, N)
    omega = 2.0 * math.pi
    ks = np.arange(1, d + 1, dtype=float)
    path = np.sin(omega * ts[:, None] * ks[None, :])
    return path

def make_path(d: int, N: int, kind: str) -> np.ndarray:
    """Generate path of specified kind"""
    kind = kind.lower()
    if kind == "linear":
        return make_path_linear(d, N)
    elif kind == "sin":
        return make_path_sin(d, N)
    else:
        raise ValueError(f"Unknown path_kind: {kind}")

# -------- Run Folder Management --------

def setup_run_folder(run_type: str, cfg: Dict[str, Any]) -> Path:
    """
    Create a timestamped run folder with standardized structure.
    
    Args:
        run_type: e.g. 'benchmark_python', 'benchmark_python_only',
                  'benchmark_comparison', etc.
        cfg: Configuration dictionary
    
    Returns:
        Path to the created run folder
    """
    runs_root = SCRIPT_DIR / cfg.get("runs_dir", "runs")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{run_type}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config snapshot
    if CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, run_dir / "benchmark_config.yaml")
    
    # Write resolved config
    config_path = run_dir / "config_resolved.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    
    # Create metadata
    metadata = {
        "run_type": run_type,
        "timestamp": ts,
        "start_time": datetime.now().isoformat(),
    }
    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created run folder: {run_dir}")
    return run_dir

def finalize_run_folder(run_dir: Path, summary: Dict[str, Any]):
    """
    Write summary and completion metadata to run folder.
    
    Args:
        run_dir: Path to run folder
        summary: Dictionary with summary information
    """
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    metadata["end_time"] = datetime.now().isoformat()
    metadata["summary"] = summary
    
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    # Write human-readable summary
    summary_path = run_dir / "SUMMARY.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Benchmark Run Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Run Type: {metadata.get('run_type', 'unknown')}\n")
        f.write(f"Start:    {metadata.get('start_time', 'unknown')}\n")
        f.write(f"End:      {metadata.get('end_time', 'unknown')}\n\n")
        
        f.write("Results:\n")
        f.write("-" * 60 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Run completed. Summary written to: {summary_path}")
