import pytest
import os

# Global FULL_SUITE flag to control which parameter sets are used
FULL_SUITE = None

def pytest_addoption(parser):
    """Add the --full-suite command-line option for pytest."""
    parser.addoption(
        "--full-suite", action="store_true", default=False, help="Run the full test suite"
    )

def pytest_configure(config):
    """Set the FULL_SUITE flag based on --full-suite option or environment variable."""
    global FULL_SUITE
    FULL_SUITE = config.getoption("--full-suite") or os.getenv("FULL_TESTS") == "1"


def conditional_parametrize(argnames, quick_params, full_params):
    """Decorator for conditional parameterization based on FULL_SUITE flag."""
    def decorator(func):
        if FULL_SUITE:
            return pytest.mark.parametrize(argnames, full_params)(func)
        else:
            return pytest.mark.parametrize(argnames, quick_params)(func)
    return decorator


def conditional_value(quick_value, full_value):
    if FULL_SUITE:
        return quick_value
    return full_value
