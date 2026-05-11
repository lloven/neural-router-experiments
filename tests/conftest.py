"""Shared pytest configuration for Neural Router tests.

Registers the 'remote' marker and provides fixtures for remote VM access.
Local tests (default) run without any special flags.
Remote tests require: pytest --remote
"""

import pytest


def pytest_addoption(parser):
    """Add --remote CLI flag for enabling remote VM tests."""
    parser.addoption(
        "--remote",
        action="store_true",
        default=False,
        help="Run tests that require SSH access to the remote VM.",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.remote tests unless --remote is passed."""
    if config.getoption("--remote"):
        return  # --remote given: run everything
    skip_remote = pytest.mark.skip(reason="needs --remote flag to run")
    for item in items:
        if "remote" in item.keywords:
            item.add_marker(skip_remote)


@pytest.fixture
def ssh_host():
    """SSH host alias for the remote VM. Override via NROUTER_SSH_HOST env var."""
    import os
    return os.environ.get("NROUTER_SSH_HOST", "remote-host")


@pytest.fixture
def remote_dir():
    """Remote working directory. Override via NROUTER_REMOTE_DIR env var."""
    import os
    return os.environ.get("NROUTER_REMOTE_DIR", "~/neural-router")
