"""Tests for core functionality."""

from mlops.core import add


def test_add():
    """Test add function."""
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0 