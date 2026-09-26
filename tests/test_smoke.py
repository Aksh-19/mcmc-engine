"""Smoke tests to verify the package is installed and importable."""

import mcmc


def test_version():
    """Verify the package version is set."""
    assert mcmc.__version__ == "0.1.0"


def test_import_distributions():
    """Verify the distributions subpackage is importable."""
    import mcmc.distributions  # noqa: F401


def test_import_samplers():
    """Verify the samplers subpackage is importable."""
    import mcmc.samplers  # noqa: F401


def test_import_diagnostics():
    """Verify the diagnostics subpackage is importable."""
    import mcmc.diagnostics  # noqa: F401


def test_import_autodiff():
    """Verify the autodiff subpackage is importable."""
    import mcmc.autodiff  # noqa: F401
