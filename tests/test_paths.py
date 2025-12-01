"""Unit tests for path generation utilities"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.paths import make_path_linear, make_path_sin, make_path


class TestPathGeneration:
    """Test suite for path generation functions"""

    def test_linear_path_shape(self):
        """Test that linear path has correct shape"""
        N, d = 100, 3
        path = make_path_linear(d, N)
        assert path.shape == (N, d), f"Expected shape ({N}, {d}), got {path.shape}"

    def test_linear_path_first_dimension(self):
        """Test that first dimension is linear time [0, 1]"""
        N, d = 50, 4
        path = make_path_linear(d, N)
        # First column should be linearly spaced from 0 to 1
        expected_first_col = np.linspace(0.0, 1.0, N)
        np.testing.assert_allclose(path[:, 0], expected_first_col, rtol=1e-10)

    def test_linear_path_other_dimensions(self):
        """Test that other dimensions are 2*t"""
        N, d = 50, 4
        path = make_path_linear(d, N)
        ts = np.linspace(0.0, 1.0, N)
        for i in range(1, d):
            np.testing.assert_allclose(path[:, i], 2.0 * ts, rtol=1e-10)

    def test_linear_path_1d(self):
        """Test linear path with d=1"""
        N, d = 30, 1
        path = make_path_linear(d, N)
        assert path.shape == (N, 1)
        expected = np.linspace(0.0, 1.0, N)[:, None]
        np.testing.assert_allclose(path, expected, rtol=1e-10)

    def test_sin_path_shape(self):
        """Test that sinusoidal path has correct shape"""
        N, d = 100, 5
        path = make_path_sin(d, N)
        assert path.shape == (N, d), f"Expected shape ({N}, {d}), got {path.shape}"

    def test_sin_path_values(self):
        """Test that sinusoidal path has correct values"""
        N, d = 10, 2
        path = make_path_sin(d, N)
        ts = np.linspace(0.0, 1.0, N)
        omega = 2.0 * np.pi

        # First dimension: sin(2π * 1 * t)
        expected_col0 = np.sin(omega * 1 * ts)
        np.testing.assert_allclose(path[:, 0], expected_col0, rtol=1e-10)

        # Second dimension: sin(2π * 2 * t)
        expected_col1 = np.sin(omega * 2 * ts)
        np.testing.assert_allclose(path[:, 1], expected_col1, rtol=1e-10)

    def test_sin_path_bounds(self):
        """Test that sinusoidal path values are within [-1, 1]"""
        N, d = 100, 3
        path = make_path_sin(d, N)
        assert np.all(path >= -1.0) and np.all(path <= 1.0), \
            "Sinusoidal path values should be in [-1, 1]"

    def test_make_path_dispatcher_linear(self):
        """Test make_path dispatcher with 'linear' kind"""
        N, d = 20, 3
        path1 = make_path(d, N, "linear")
        path2 = make_path_linear(d, N)
        np.testing.assert_array_equal(path1, path2)

    def test_make_path_dispatcher_sin(self):
        """Test make_path dispatcher with 'sin' kind"""
        N, d = 20, 3
        path1 = make_path(d, N, "sin")
        path2 = make_path_sin(d, N)
        np.testing.assert_array_equal(path1, path2)

    def test_make_path_case_insensitive(self):
        """Test that make_path is case-insensitive"""
        N, d = 15, 2
        path_lower = make_path(d, N, "linear")
        path_upper = make_path(d, N, "LINEAR")
        path_mixed = make_path(d, N, "LiNeAr")
        np.testing.assert_array_equal(path_lower, path_upper)
        np.testing.assert_array_equal(path_lower, path_mixed)

    def test_make_path_invalid_kind(self):
        """Test that make_path raises error for invalid kind"""
        with pytest.raises(ValueError, match="Unknown path_kind"):
            make_path(2, 10, "invalid")

    def test_path_dtype(self):
        """Test that paths are float type"""
        path_linear = make_path_linear(3, 50)
        path_sin = make_path_sin(3, 50)
        assert np.issubdtype(path_linear.dtype, np.floating)
        assert np.issubdtype(path_sin.dtype, np.floating)

    def test_path_contiguous(self):
        """Test that paths are C-contiguous arrays"""
        path_linear = make_path_linear(3, 50)
        path_sin = make_path_sin(3, 50)
        assert path_linear.flags['C_CONTIGUOUS']
        assert path_sin.flags['C_CONTIGUOUS']
