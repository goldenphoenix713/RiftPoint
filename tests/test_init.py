"""Smoke tests for RiftPoint package initialization."""

import riftpoint


def test_package_version() -> None:
    """Ensure package version is defined and accessible."""
    assert riftpoint.__version__ == "0.1.0"
