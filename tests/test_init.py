"""Smoke tests for RiftPoint package initialization."""

import riftpoint


def test_package_version() -> None:
    """Ensure package version is defined and accessible."""
    assert riftpoint.__version__ == "1.0.0"


def test_package_exports() -> None:
    """Ensure core symbols and logger are exported."""
    assert hasattr(riftpoint, "RiftCheckpointSaver")
    assert hasattr(riftpoint, "BaseRiftSaver")
    assert hasattr(riftpoint, "logger")
    assert callable(getattr(riftpoint.logger, "info", None))
