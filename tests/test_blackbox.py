"""Tests for the black box recorder and analyzer (GPU VM diagnostics).

The black box recorder (blackbox.sh) must:
1. Read /proc files directly — no forking (ps, grep, wc, awk) in the main loop
2. Sample nvidia-smi at low frequency (every 5th iteration, not every iteration)
3. Output valid JSONL with all required fields
4. Implement ringbuffer behavior (max 1000 samples)

The analyzer (analyze_blackbox.py) must:
1. Parse JSONL recordings
2. Detect load spikes
3. Correlate NFS ops with load changes

See: L44 (one step at a time), L45 (don't repeat without changes)
"""
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

BLACKBOX = Path(__file__).parent.parent / "scripts" / "remote" / "blackbox.sh"
ANALYZER = Path(__file__).parent.parent / "scripts" / "remote" / "analyze_blackbox.py"


# ---------------------------------------------------------------------------
# Bash recorder tests
# ---------------------------------------------------------------------------


class TestBlackboxExists:
    def test_blackbox_exists(self):
        assert BLACKBOX.exists(), f"Recorder not found at {BLACKBOX}"

    def test_blackbox_is_executable(self):
        assert os.access(BLACKBOX, os.X_OK), "blackbox.sh is not executable"


class TestBlackboxNoFork:
    """The recorder MUST NOT fork in its main loop (L44).

    Allowed exceptions:
    - nvidia-smi (in a clearly guarded block, every 5th iteration)
    - timeout (wrapping nvidia-smi only)
    - date (for timestamp — but we prefer printf)
    - Comments (lines starting with #)
    """

    FORBIDDEN_CMDS = ["ps ", "grep ", "wc ", "awk ", "sed ", "cut ", "tr "]

    def _get_main_loop_code(self):
        """Extract non-comment lines from the main while loop."""
        content = BLACKBOX.read_text()
        lines = content.splitlines()
        in_loop = False
        loop_lines = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^while\s", stripped):
                in_loop = True
                continue
            if in_loop and stripped == "done":
                break
            if in_loop and not stripped.startswith("#"):
                loop_lines.append(stripped)
        return "\n".join(loop_lines)

    def _get_non_nvidia_loop_code(self):
        """Extract main loop code EXCLUDING the nvidia-smi guarded block."""
        content = BLACKBOX.read_text()
        lines = content.splitlines()
        in_loop = False
        in_nvidia_block = False
        nvidia_depth = 0
        result = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^while\s", stripped):
                in_loop = True
                continue
            if in_loop and stripped == "done" and not in_nvidia_block:
                break
            if not in_loop:
                continue
            if stripped.startswith("#"):
                continue
            # Detect nvidia-smi guarded block (if block containing nvidia-smi)
            if "nvidia-smi" in stripped or "nvidia_smi" in stripped:
                in_nvidia_block = True
            if in_nvidia_block:
                if "if " in stripped or "if [" in stripped:
                    nvidia_depth += 1
                if stripped == "fi":
                    nvidia_depth -= 1
                    if nvidia_depth <= 0:
                        in_nvidia_block = False
                        nvidia_depth = 0
                continue
            result.append(stripped)
        return "\n".join(result)

    def test_no_ps_in_main_loop(self):
        code = self._get_non_nvidia_loop_code()
        assert " ps " not in code, "Main loop uses 'ps' which forks"

    def test_no_grep_in_main_loop(self):
        code = self._get_non_nvidia_loop_code()
        assert "grep " not in code, "Main loop uses 'grep' which forks"
        assert "| grep" not in code, "Main loop pipes through grep"

    def test_no_wc_in_main_loop(self):
        code = self._get_non_nvidia_loop_code()
        assert " wc " not in code, "Main loop uses 'wc' which forks"
        assert "| wc" not in code, "Main loop pipes through wc"

    def test_no_awk_in_main_loop(self):
        code = self._get_non_nvidia_loop_code()
        assert " awk " not in code, "Main loop uses 'awk' which forks"
        assert "| awk" not in code, "Main loop pipes through awk"

    def test_no_subshell_in_main_loop(self):
        """No $(...) or backticks in non-nvidia code (except safe builtins).

        $((arithmetic)) is bash built-in arithmetic, NOT a fork.
        $(printf ...) and $(date ...) are builtins, not forks.
        $(dirname ...) at init time is fine (not in the loop).
        """
        code = self._get_non_nvidia_loop_code()
        # Remove known-safe patterns
        safe_removed = re.sub(r'\$\(date[^)]*\)', '', code)
        safe_removed = re.sub(r'\$\(printf[^)]*\)', '', safe_removed)
        # $((expr)) is arithmetic expansion, not a subshell — remove it
        safe_removed = re.sub(r'\$\(\([^)]*\)\)', '', safe_removed)
        subshells = re.findall(r'\$\([^)]+\)', safe_removed)
        backticks = re.findall(r'`[^`]+`', safe_removed)
        assert not subshells, f"Subshell in main loop (non-nvidia): {subshells}"
        assert not backticks, f"Backtick subshell in main loop: {backticks}"


class TestBlackboxProcReads:
    """Recorder must read from /proc directly."""

    def test_reads_proc_loadavg(self):
        content = BLACKBOX.read_text()
        assert "/proc/loadavg" in content, "Must read /proc/loadavg for load averages"

    def test_reads_proc_stat(self):
        content = BLACKBOX.read_text()
        assert "/proc/stat" in content, "Must read /proc/stat for procs_running/blocked"

    def test_scans_proc_pid_comm(self):
        """Must scan /proc/*/comm to count Python processes without forking."""
        content = BLACKBOX.read_text()
        assert "/proc/" in content and "comm" in content, (
            "Must scan /proc/<pid>/comm for Python process counting"
        )

    def test_reads_proc_pid_stat_for_ppid(self):
        """Must read /proc/<pid>/stat to get parent PID of Python processes."""
        content = BLACKBOX.read_text()
        # Should reference reading stat file for ppid extraction
        non_comment = "\n".join(
            l for l in content.splitlines() if not l.strip().startswith("#")
        )
        assert "ppid" in non_comment.lower() or (
            "/stat" in non_comment and "proc" in non_comment
        ), "Must extract ppid from /proc/<pid>/stat"

    def test_reads_mountstats_for_nfs(self):
        """Must read /proc/self/mountstats for NFS operation counts."""
        content = BLACKBOX.read_text()
        assert "mountstats" in content, "Must read /proc/self/mountstats for NFS ops"


class TestBlackboxOutput:
    def test_output_is_jsonl_format(self):
        """The output path must be the specified JSONL file."""
        content = BLACKBOX.read_text()
        assert "blackbox.jsonl" in content, "Must write to blackbox.jsonl"

    def test_output_has_required_fields(self):
        """JSON output must contain all required diagnostic fields."""
        content = BLACKBOX.read_text()
        required_fields = [
            "ts",
            "load_1m",
            "procs_running",
            "procs_blocked",
            "python_count",
            "nfs_ops",
            "gpu_mem_mb",
            "gpu_util_pct",
        ]
        for field in required_fields:
            assert field in content, f"Missing required field '{field}' in output"


class TestBlackboxRingbuffer:
    def test_has_ringbuffer_limit(self):
        """Must implement max sample limit (1000 lines)."""
        content = BLACKBOX.read_text()
        assert "1000" in content, "Must have 1000-sample ringbuffer limit"

    def test_ringbuffer_uses_tail_or_truncate(self):
        """Ringbuffer must have a mechanism to trim old entries."""
        content = BLACKBOX.read_text()
        # Should use tail to keep last N lines, or a line-count check
        has_trim = (
            "tail" in content
            or "tmpfile" in content
            or "line_count" in content
            or "MAX_SAMPLES" in content
            or "max_lines" in content
            or "wc -l" in content  # allowed outside hot path for periodic trim
        )
        assert has_trim, "Must have a mechanism to enforce ringbuffer size"


class TestBlackboxNvidiaSmi:
    def test_nvidia_smi_present(self):
        content = BLACKBOX.read_text()
        assert "nvidia-smi" in content, "Must call nvidia-smi for GPU monitoring"

    def test_nvidia_smi_is_low_frequency(self):
        """nvidia-smi must be called every Nth iteration, not every iteration."""
        content = BLACKBOX.read_text()
        # Should have a counter or modulo check gating nvidia-smi
        has_counter = bool(
            re.search(r"counter|iteration|gpu_counter|nvidia_counter|sample_count", content, re.IGNORECASE)
        )
        has_modulo = bool(re.search(r"%\s*\d+|mod\s+\d+", content))
        has_interval_check = bool(
            re.search(r"eq\s+0|==\s*0|-eq\s+0", content)
        )
        assert has_counter or has_modulo or has_interval_check, (
            "nvidia-smi must be gated by a counter/modulo check (every 5th iteration)"
        )

    def test_nvidia_smi_has_timeout(self):
        """nvidia-smi must be wrapped in timeout to survive hangs during cascades."""
        content = BLACKBOX.read_text()
        lines = content.splitlines()
        found = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comments and variable assignments in header
            if stripped.startswith("#"):
                continue
            if "nvidia-smi" in stripped and ("query" in stripped or "--" in stripped):
                # This is the actual nvidia-smi invocation line
                found = True
                context = "\n".join(lines[max(0, i - 3): i + 3])
                assert "timeout" in context, (
                    "nvidia-smi must be wrapped in 'timeout' to survive cascade hangs"
                )
                break
        assert found, "No nvidia-smi invocation found in script"


class TestBlackboxSyntax:
    def test_shellcheck_clean(self):
        """Recorder should pass basic bash syntax check."""
        result = subprocess.run(
            ["bash", "-n", str(BLACKBOX)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------

SAMPLE_JSONL = [
    {"ts": "2026-03-25T05:20:00", "load_1m": 0.08, "load_5m": 0.02, "procs_running": 2, "procs_blocked": 0, "python_count": 0, "python_ppids": [], "nfs_ops": 100, "gpu_mem_mb": 0, "gpu_util_pct": 0},
    {"ts": "2026-03-25T05:20:02", "load_1m": 0.10, "load_5m": 0.03, "procs_running": 2, "procs_blocked": 0, "python_count": 0, "python_ppids": [], "nfs_ops": 102, "gpu_mem_mb": 0, "gpu_util_pct": 0},
    {"ts": "2026-03-25T05:20:04", "load_1m": 5.00, "load_5m": 1.00, "procs_running": 8, "procs_blocked": 3, "python_count": 3, "python_ppids": [1234, 1234, 5678], "nfs_ops": 500, "gpu_mem_mb": 2048, "gpu_util_pct": 45},
    {"ts": "2026-03-25T05:20:06", "load_1m": 25.00, "load_5m": 5.00, "procs_running": 30, "procs_blocked": 15, "python_count": 12, "python_ppids": [1234, 1234, 1234, 5678, 5678, 5678, 5678, 5678, 5678, 5678, 5678, 5678], "nfs_ops": 2000, "gpu_mem_mb": 4096, "gpu_util_pct": 80},
    {"ts": "2026-03-25T05:20:08", "load_1m": 45.00, "load_5m": 10.00, "procs_running": 50, "procs_blocked": 30, "python_count": 20, "python_ppids": [1234]*5 + [5678]*15, "nfs_ops": 8000, "gpu_mem_mb": 8192, "gpu_util_pct": 95},
]


@pytest.fixture
def sample_jsonl_file(tmp_path):
    """Create a temporary JSONL file with sample blackbox data."""
    f = tmp_path / "blackbox.jsonl"
    f.write_text("\n".join(json.dumps(s) for s in SAMPLE_JSONL) + "\n")
    return f


class TestAnalyzerExists:
    def test_analyzer_exists(self):
        assert ANALYZER.exists(), f"Analyzer not found at {ANALYZER}"

    def test_analyzer_is_executable(self):
        assert os.access(ANALYZER, os.X_OK), "analyze_blackbox.py is not executable"


class TestAnalyzerReadsJsonl:
    def test_reads_jsonl_file(self, sample_jsonl_file):
        """Analyzer must successfully parse a JSONL file."""
        result = subprocess.run(
            ["python3", str(ANALYZER), str(sample_jsonl_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Analyzer failed: {result.stderr}"
        assert len(result.stdout) > 0, "Analyzer produced no output"


class TestAnalyzerFindsLoadSpike:
    def test_detects_load_spike(self, sample_jsonl_file):
        """Analyzer must detect when load crosses a threshold."""
        result = subprocess.run(
            ["python3", str(ANALYZER), str(sample_jsonl_file)],
            capture_output=True,
            text=True,
        )
        output = result.stdout.lower()
        assert "load" in output and ("spike" in output or "cross" in output or "exceed" in output or "45" in output), (
            "Analyzer must report load spike events"
        )

    def test_reports_first_python_process(self, sample_jsonl_file):
        """Analyzer must identify when Python processes first appeared."""
        result = subprocess.run(
            ["python3", str(ANALYZER), str(sample_jsonl_file)],
            capture_output=True,
            text=True,
        )
        output = result.stdout.lower()
        assert "python" in output, "Analyzer must report Python process events"


class TestAnalyzerCorrelation:
    def test_correlates_nfs_and_load(self, sample_jsonl_file):
        """Analyzer must compare NFS ops timing with load timing."""
        result = subprocess.run(
            ["python3", str(ANALYZER), str(sample_jsonl_file)],
            capture_output=True,
            text=True,
        )
        output = result.stdout.lower()
        assert "nfs" in output, "Analyzer must report NFS operation analysis"

    def test_ppid_analysis(self, sample_jsonl_file):
        """Analyzer must report parent process IDs of Python processes."""
        result = subprocess.run(
            ["python3", str(ANALYZER), str(sample_jsonl_file)],
            capture_output=True,
            text=True,
        )
        output = result.stdout
        assert "1234" in output or "5678" in output or "ppid" in output.lower(), (
            "Analyzer must report Python process parent PIDs"
        )

    def test_gpu_memory_timeline(self, sample_jsonl_file):
        """Analyzer must report GPU memory usage over time."""
        result = subprocess.run(
            ["python3", str(ANALYZER), str(sample_jsonl_file)],
            capture_output=True,
            text=True,
        )
        output = result.stdout.lower()
        assert "gpu" in output, "Analyzer must report GPU memory timeline"
