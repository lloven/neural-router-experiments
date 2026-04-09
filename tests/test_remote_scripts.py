"""Tests for scripts/remote/ shell scripts.

TDD: These tests verify existence, executability, syntax validity,
and required content of the remote execution shell scripts.
"""

import os
import subprocess
import stat

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REMOTE_DIR = os.path.join(REPO_ROOT, "scripts", "remote")

SCRIPTS = {
    "setup_vm": os.path.join(REMOTE_DIR, "setup-vm.sh"),
    "deploy": os.path.join(REMOTE_DIR, "deploy.sh"),
    "run_experiments": os.path.join(REMOTE_DIR, "run-experiments.sh"),
    "collect_results": os.path.join(REMOTE_DIR, "collect-results.sh"),
    "check_vm": os.path.join(REMOTE_DIR, "check-vm.sh"),
}


# ---------------------------------------------------------------------------
# Existence + executable
# ---------------------------------------------------------------------------


class TestScriptExistence:
    """Every remote script must exist and be executable."""

    def test_setup_vm_exists_and_executable(self):
        path = SCRIPTS["setup_vm"]
        assert os.path.isfile(path), f"{path} does not exist"
        assert os.stat(path).st_mode & stat.S_IXUSR, f"{path} is not executable"

    def test_deploy_exists_and_executable(self):
        path = SCRIPTS["deploy"]
        assert os.path.isfile(path), f"{path} does not exist"
        assert os.stat(path).st_mode & stat.S_IXUSR, f"{path} is not executable"

    def test_run_experiments_exists_and_executable(self):
        path = SCRIPTS["run_experiments"]
        assert os.path.isfile(path), f"{path} does not exist"
        assert os.stat(path).st_mode & stat.S_IXUSR, f"{path} is not executable"

    def test_collect_results_exists_and_executable(self):
        path = SCRIPTS["collect_results"]
        assert os.path.isfile(path), f"{path} does not exist"
        assert os.stat(path).st_mode & stat.S_IXUSR, f"{path} is not executable"

    def test_check_vm_exists_and_executable(self):
        path = SCRIPTS["check_vm"]
        assert os.path.isfile(path), f"{path} does not exist"
        assert os.stat(path).st_mode & stat.S_IXUSR, f"{path} is not executable"


# ---------------------------------------------------------------------------
# Syntax validity (bash -n)
# ---------------------------------------------------------------------------


class TestScriptSyntax:
    """All scripts must pass bash -n (syntax check)."""

    def test_setup_vm_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", SCRIPTS["setup_vm"]], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Syntax error:\n{result.stderr}"

    def test_deploy_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", SCRIPTS["deploy"]], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Syntax error:\n{result.stderr}"

    def test_run_experiments_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", SCRIPTS["run_experiments"]], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Syntax error:\n{result.stderr}"

    def test_collect_results_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", SCRIPTS["collect_results"]], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Syntax error:\n{result.stderr}"

    def test_check_vm_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", SCRIPTS["check_vm"]], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Syntax error:\n{result.stderr}"


# ---------------------------------------------------------------------------
# Content requirements
# ---------------------------------------------------------------------------


class TestScriptContent:
    """Scripts must contain required actions/flags."""

    def test_run_experiments_has_required_actions(self):
        with open(SCRIPTS["run_experiments"]) as f:
            content = f.read()
        for action in ("smoke", "full", "crossover", "stop", "status"):
            assert action in content, (
                f"run-experiments.sh missing required action: {action}"
            )

    def test_deploy_has_setup_flag(self):
        with open(SCRIPTS["deploy"]) as f:
            content = f.read()
        assert "--setup" in content, "deploy.sh missing --setup flag"

    def test_collect_results_has_mode_filter(self):
        with open(SCRIPTS["collect_results"]) as f:
            content = f.read()
        assert "--mode" in content, "collect-results.sh missing --mode filter"
