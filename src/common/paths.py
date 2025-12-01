"""Path generation utilities"""

import math
import numpy as np


def make_path_linear(d: int, N: int) -> np.ndarray:
    """
    Generate linear path: [t, 2t, 2t, ...]

    Args:
        d: Dimension of the path
        N: Number of points

    Returns:
        Array of shape (N, d)
    """
    ts = np.linspace(0.0, 1.0, N)
    path = np.empty((N, d), dtype=float)
    path[:, 0] = ts
    if d > 1:
        path[:, 1:] = 2.0 * ts[:, None]
    return path


def make_path_sin(d: int, N: int) -> np.ndarray:
    """
    Generate sinusoidal path: [sin(2π·1·t), sin(2π·2·t), ...]

    Args:
        d: Dimension of the path
        N: Number of points

    Returns:
        Array of shape (N, d)
    """
    ts = np.linspace(0.0, 1.0, N)
    omega = 2.0 * math.pi
    ks = np.arange(1, d + 1, dtype=float)
    path = np.sin(omega * ts[:, None] * ks[None, :])
    return path


def make_path(d: int, N: int, kind: str) -> np.ndarray:
    """
    Generate path of specified kind.

    Args:
        d: Dimension of the path
        N: Number of points
        kind: "linear" or "sin"

    Returns:
        Array of shape (N, d)
    """
    kind = kind.lower()
    if kind == "linear":
        return make_path_linear(d, N)
    elif kind == "sin":
        return make_path_sin(d, N)
    else:
        raise ValueError(f"Unknown path_kind: {kind}")
