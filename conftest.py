import pytest


def pytest_addoption(parser):
    """Adds the --plots command line flag."""
    parser.addoption(
        "--plots",
        action="store_true",
        default=False,
        help="Display plots during test execution.",
    )


@pytest.fixture
def plots_enabled(request):
    """Fixture that returns True if --plots is passed."""
    return request.config.getoption("--plots")
