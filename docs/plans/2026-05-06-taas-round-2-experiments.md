# TAAS Round-2 Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four BOUNDED FORMAL items the TAAS round-1 reviewer panel made mandatory before resubmission: (C9) a perturbation experiment that exercises the QoE adaptive loop, (H4) a calibration-fraction sweep that empirically substantiates the "calibration-noise-limited" claim, (R5) a reseeding pass that brings `tab:cross-dataset` cells from n=3 to n=5 seeds, and (R6) a stratified miss analysis on the cost-model factor-of-two band.

**Architecture:** Four narrow shells over existing implementations in `src/`. (i) **C9 perturbation** runs `scripts/run_qoe.py` twice on D1 — once with calibration drawn from a topic-restricted slice and evaluation on the full corpus (calibration-domain shift), once with full-corpus calibration but mid-run injection of backend latency degradation (failure injection). The QoE assigner already exists; the new code is a `--perturbation` flag plus three perturbation hooks in `src/qoe.py`. (ii) **H4 sweep** wires the existing `QoEAssigner.calibration_fraction` parameter (already plumbed in `src/qoe.py`) to a new `--calibration-fraction` CLI flag and runs the sweep `0.05, 0.10, 0.20, 0.50` on D1. (iii) **R5 reseeding** re-runs `scripts/run_experiment.py` for the existing tab:cross-dataset cells with `--seeds 42,123,456,789,0` instead of the current 3-seed protocol; alternative path is updating §4.10's stated protocol to honestly disclose the n=3. (iv) **R6 stratified miss** is a pure analysis pass over the existing per-cluster cost-validation parquet, no new compute.

**Tech Stack:** Python 3.11/3.12 (laptop venv / CSC `pytorch/2.9` module), pytest, existing experiment harness (`src/qoe.py`, `src/router.py`, `scripts/run_qoe.py`, `scripts/run_experiment.py`), Ollama on Mahti (Qwen 2.5 7B + 32B), SLURM `gputest` for smokes / `gpusmall` for full runs.

**Estimated cost:** ~140 BU CSC (Mahti gpusmall, 16 BU/GPU-h, 1.5× safety) for C9 + H4 + R5 combined. R6 is laptop-only (~5 min CPU). Zero Anthropic spend (all open-weight Qwen tiers).

---

## Scope Check

The four items target one consensus issue (C9, the SI's exercise-the-mechanism gate), one dual-reviewer item (H4, the calibration-noise-limited claim), one red-flag item (R5, protocol drift), and one recommended single-reviewer item (R6, methodology authority). They are independent: C9 and H4 share the `run_qoe.py` shell but have separate output CSVs and figures; R5 calls a different harness; R6 is pure analysis. Splitting into separate plans buys nothing because the operational context (Mahti, OPERATIONS_LOG, cost accounting) is shared. One plan, four task groups.

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `src/qoe.py` | **MODIFY** | Add `PerturbationSpec` dataclass + three hook callsites: `topic_restricted_calibration_sample`, `inject_latency_after_n_events`, `inject_backend_failure`. Idempotent: passes through unchanged when `perturbation=None`. |
| `tests/test_qoe_perturbations.py` | **CREATE** | TDD: invariant tests for each perturbation hook. (a) `topic_restricted` calibration sample respects the topic mask; (b) latency injection adds the configured offset to events ≥ N; (c) failure injection marks events ≥ N with `backend_failed=True` and the QoE assigner reroutes them. |
| `scripts/run_qoe.py` | **MODIFY** | Add `--perturbation {none,topic_restricted_cal,latency_injection,failure_injection}` and `--calibration-fraction FLOAT` CLI flags; thread to `QoEAssigner`. |
| `scripts/run_qoe_perturbation.py` | **CREATE** | Top-level driver that calls `run_qoe.py` once per perturbation × strategy × seed cell and aggregates output CSVs into `results/qoe_perturbation/`. |
| `scripts/slurm/mahti_qoe_perturbation_smoke.sh` | **CREATE** | gputest smoke: 1 perturbation × 1 strategy × 1 seed, ~10 min. |
| `scripts/slurm/mahti_qoe_perturbation_full.sh` | **CREATE** | gpusmall full: 2 perturbations × 3 strategies × 5 seeds × 2 backends, ~3-4 h. |
| `scripts/slurm/mahti_qoe_calfrac_smoke.sh` | **CREATE** | gputest smoke for H4: 1 calibration fraction × 1 strategy × 1 seed. |
| `scripts/slurm/mahti_qoe_calfrac_full.sh` | **CREATE** | gpusmall full for H4: 4 calibration fractions × 3 strategies × 5 seeds. |
| `scripts/slurm/mahti_reseeding_full.sh` | **CREATE** | gpusmall full for R5: re-run cross-dataset cells at seeds {123, 456} (the missing two), aggregating into existing CSVs. |
| `analysis/make_figures.py` | **MODIFY** | Add `fig_qoe_perturbation()` (2-panel: distribution-shift, failure-injection) and `fig_qoe_calfrac()` (1-panel F1 vs calibration_fraction). |
| `analysis/cost_validation_stratified.py` | **CREATE** | R6: pandas analysis on `results/cost_validation/per_cluster.parquet`; stratify the 17% miss cells by `m_c≤b_max` (trivial regime) vs `m_c>b_max` (non-trivial); emit a summary paragraph + small table for §5.4. |
| `tests/test_qoe_calibration_fraction.py` | **CREATE** | TDD: assert `--calibration-fraction 0.20` results in `n_sample = 0.20 × n_events` per cluster (rounding rule). |
| `tests/test_run_qoe_perturbation.py` | **CREATE** | TDD: validate driver script aggregates correctly; smoke 1 cell. |
| `Manuscripts/Neural Router (Elsevier FGCS)/figs/qoe_perturbation.pdf` | **GENERATE** | After C9 data lands. |
| `Manuscripts/Neural Router (Elsevier FGCS)/figs/qoe_calfrac.pdf` | **GENERATE** | After H4 data lands. |
| `Manuscripts/Neural Router (Elsevier FGCS)/txt/Results.tex` | **MODIFY** | New §5.9.1 (perturbation result) and §5.9.2 (calibration sweep); update `tab:qoe` / `Suppl.\ Tab.~2` rows for n=5 if R5 path is taken. |
| `Manuscripts/Neural Router (Elsevier FGCS)/txt/Experiment.tex` | **MODIFY** | Either re-state §4.10 protocol to "n=5 for ablation, n=3 for cross-dataset (compute-budget constraint)" OR remove the discrepancy after R5 reseeding completes. |
| `OPERATIONS_LOG.md` (local + Mahti) | **APPEND** | Per L55 + `feedback_remote_logging.md`: every launch, kill, reset, and progress backup. Same-day entries, never reconstructed post-hoc. |

## Pre-flight Checks

- [ ] **PF1: Mahti reachable + project quota healthy**

  Run: `ssh -o BatchMode=yes mahti 'hostname; date; csc-projects | head -5'`
  Expected: hostname returned, date current, project_2018951 listed with positive remaining BUs. If quota under 200 BU, surface to user and pause.

- [ ] **PF2: Existing test suite green**

  Run: `cd Experiments/neural-router && pytest -x -q tests/`
  Expected: all green. If reds, fix before adding new tests (per L31).

- [ ] **PF3: Read EXPERIMENT_PROTOCOL.md**

  Run: `cat Experiments/neural-router/EXPERIMENT_PROTOCOL.md | head -80`
  Expected: read fully (per L47). Note any pre-flight steps the protocol mandates that are not yet in this list.

- [ ] **PF4: Verify the cost-validation parquet exists for R6**

  Run: `ls -la Experiments/neural-router/results/cost_validation/per_cluster.parquet`
  Expected: file exists, non-zero size. If missing, R6 is blocked; surface to user.

- [ ] **PF5: OPERATIONS_LOG entry**

  Append to both `Experiments/neural-router/OPERATIONS_LOG.md` (local) and remote Mahti `OPERATIONS_LOG.md` (per L55):

  ```
  ## 2026-05-06 — TAAS Round-2 plan dispatched
  - **Plan:** docs/plans/2026-05-06-taas-round-2-experiments.md
  - **Items:** C9 perturbation, H4 calibration sweep, R5 reseeding, R6 stratified miss
  - **Reason:** TAAS round-1 reviewer mandate (BOUNDED FORMAL items)
  - **Estimated BU:** ~140 (C9 ~70, H4 ~50, R5 ~20, R6 ~0)
  ```

---

## Task 1: C9 — Perturbation experiment for the QoE loop

### Task 1.1: Add `PerturbationSpec` to src/qoe.py

**Files:**
- Modify: `Experiments/neural-router/src/qoe.py` (add new dataclass at the top of the QoE-types block, no callsite changes yet)
- Test: `Experiments/neural-router/tests/test_qoe_perturbations.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qoe_perturbations.py
import pytest
from src.qoe import PerturbationSpec

def test_perturbation_spec_defaults_to_none():
    """L38: a default-constructed perturbation must be a no-op."""
    p = PerturbationSpec()
    assert p.kind == "none"
    assert p.is_noop()

def test_perturbation_spec_topic_restricted_calibration():
    """L41: critical-path perturbation must declare the affected component."""
    p = PerturbationSpec(
        kind="topic_restricted_cal",
        topic_mask=[0, 1, 2],
        injection_event_index=None,
        injected_latency_s=None,
    )
    assert p.kind == "topic_restricted_cal"
    assert p.topic_mask == [0, 1, 2]
    assert not p.is_noop()
```

- [ ] **Step 2: Verify RED**

  Run: `cd Experiments/neural-router && pytest -x tests/test_qoe_perturbations.py::test_perturbation_spec_defaults_to_none -v`
  Expected: FAIL with `ImportError: cannot import name 'PerturbationSpec'`.

- [ ] **Step 3: Implement minimal**

```python
# src/qoe.py (insert after the QOE_WEIGHT_PRESETS block)
from dataclasses import dataclass, field
from typing import Optional, Sequence

@dataclass
class PerturbationSpec:
    """C9: perturbation hook for the QoE adaptive loop.

    L38/L39: every perturbation must be applied verifiably (no silent no-op
    when the user requested perturbation). The `is_noop()` predicate is the
    smoke-test contract.
    """
    kind: str = "none"
    topic_mask: Optional[Sequence[int]] = None
    injection_event_index: Optional[int] = None
    injected_latency_s: Optional[float] = None
    backend_to_fail: Optional[str] = None

    def is_noop(self) -> bool:
        return self.kind == "none"
```

- [ ] **Step 4: Verify GREEN**

  Run: same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/qoe.py tests/test_qoe_perturbations.py
  git commit -m "C9: add PerturbationSpec dataclass for QoE loop perturbations"
  ```

### Task 1.2: Topic-restricted calibration hook

**Files:**
- Modify: `Experiments/neural-router/src/qoe.py:280` — `QoEAssigner._calibrate_cluster()` accepts the `PerturbationSpec` and slices the calibration sample by topic when `kind == "topic_restricted_cal"`.
- Test: `Experiments/neural-router/tests/test_qoe_perturbations.py`

- [ ] **Step 1: Write the failing test (treatment verification per L38)**

```python
def test_topic_restricted_calibration_respects_mask(synthetic_dataset):
    """The calibration sample must contain ONLY events whose topic is in
    the mask. This is a treatment-verification assertion (L38)."""
    from src.qoe import QoEAssigner, PerturbationSpec
    perturbation = PerturbationSpec(
        kind="topic_restricted_cal",
        topic_mask=[0, 1, 2],  # only first 3 of 19 D1 topics
    )
    assigner = QoEAssigner(
        calibration_fraction=0.10,
        perturbation=perturbation,
    )
    cal_sample = assigner._sample_calibration_events(
        events=synthetic_dataset.events,  # 19-topic mix
        cluster_id=0,
    )
    # Treatment: every sampled event must have topic ∈ [0, 1, 2]
    sampled_topics = {synthetic_dataset.event_topic[e.id] for e in cal_sample}
    assert sampled_topics.issubset({0, 1, 2}), (
        f"L38: topic mask not respected. Sampled topics: {sampled_topics}"
    )
    assert len(cal_sample) > 0, "L38: empty calibration sample is silent failure"
```

  Plus a fixture in `tests/conftest.py` that produces a synthetic 19-topic D1-shaped dataset (deterministic, no external data fetch). Add it if not present.

- [ ] **Step 2: Verify RED**

  Run: `pytest -x tests/test_qoe_perturbations.py::test_topic_restricted_calibration_respects_mask -v`
  Expected: FAIL because `QoEAssigner.__init__` doesn't accept `perturbation` and `_sample_calibration_events` doesn't exist.

- [ ] **Step 3: Implement**

  Wire `perturbation: Optional[PerturbationSpec] = None` into `QoEAssigner.__init__`. Extract the existing inline calibration sampling in `QoEAssigner._calibrate_cluster` (around line 280) into a new method `_sample_calibration_events`. When `perturbation.kind == "topic_restricted_cal"`, filter events by `event.topic in perturbation.topic_mask` before applying `calibration_fraction`. Empty-result guard: `raise ValueError(...)` per L39 (injection functions must not silently swallow errors).

- [ ] **Step 4: Verify GREEN**

  Run: same pytest. Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/qoe.py tests/test_qoe_perturbations.py tests/conftest.py
  git commit -m "C9: topic-restricted calibration hook + treatment-verification test"
  ```

### Task 1.3: Latency-injection hook (mid-run shift)

**Files:**
- Modify: `Experiments/neural-router/src/qoe.py:eval_with_assignment()` — accepts `PerturbationSpec` and adds `injected_latency_s` to per-event `wall_clock` for events with index ≥ `injection_event_index`.
- Test: `Experiments/neural-router/tests/test_qoe_perturbations.py`

- [ ] **Step 1: Write the failing test**

```python
def test_latency_injection_increases_post_injection_latency():
    """Treatment-verification (L38): events ≥ injection_index must have
    latency strictly greater than baseline by at least injected_latency_s.
    """
    from src.qoe import QoEAssigner, PerturbationSpec
    perturbation = PerturbationSpec(
        kind="latency_injection",
        injection_event_index=500,
        injected_latency_s=0.5,
    )
    # Run a tiny eval pass (mock LLM client returning fixed F1 = 0.5)
    # ...
    pre_latencies = [r.wall_clock_s for r in result.events if r.idx < 500]
    post_latencies = [r.wall_clock_s for r in result.events if r.idx >= 500]
    assert (
        min(post_latencies) >= max(pre_latencies) + 0.4
    ), "L38: latency injection treatment not applied to post-injection events"
```

- [ ] **Step 2: Verify RED**

  Run: `pytest -x tests/test_qoe_perturbations.py::test_latency_injection_increases_post_injection_latency -v`. Expected: FAIL.

- [ ] **Step 3: Implement**

  In `QoEAssigner.eval_with_assignment` (or wherever per-event evaluation iterates), wrap the LLM call. If `perturbation.kind == "latency_injection"` and `event.idx >= perturbation.injection_event_index`, `time.sleep(perturbation.injected_latency_s)` after the call (sleep adds wall-clock to the event's measured latency without invalidating the F1 score).

- [ ] **Step 4: Verify GREEN.**

- [ ] **Step 5: Commit.**

### Task 1.4: Backend-failure-injection hook

Same TDD shape (test the treatment first, then implement). The failure hook sets `backend_failed=True` for events ≥ `injection_event_index`, and the QoE assigner re-routes those events to the alternate backend (round-robin fallback). Treatment verification: assert that post-injection events for the failing backend's clusters arrive at the alternate backend's logs.

(Steps 1–5 follow the same RED/GREEN cycle.)

### Task 1.5: Wire `--perturbation` CLI flag in run_qoe.py

**Files:**
- Modify: `Experiments/neural-router/scripts/run_qoe.py` — `argparse` accepts `--perturbation`, `--injection-event-index`, `--injected-latency-s`, `--topic-mask`, `--backend-to-fail`. Constructs `PerturbationSpec`. Threads to `QoEAssigner`.
- Test: `Experiments/neural-router/tests/test_run_qoe_perturbation.py`

- [ ] **Step 1: Write a CLI argument-parsing test (no full run yet).**
- [ ] **Steps 2–5:** RED → GREEN → commit per TDD.

### Task 1.6: Smoke run on Mahti gputest

**Per L23 + L32:** smoke before full.

- [ ] **Step 1: Compose smoke SLURM script**

  Create `scripts/slurm/mahti_qoe_perturbation_smoke.sh` that runs ONE perturbation × ONE strategy × seed=42 with `--max-events 200 --calibration-fraction 0.10`. Walltime: 15 min. Partition: `gputest`.

- [ ] **Step 2: Submit smoke**

  Run remotely (per L55, log first):

  ```
  ssh mahti 'cd $WRKDIR/neural-router && sbatch scripts/slurm/mahti_qoe_perturbation_smoke.sh'
  ```

  Append to local + remote `OPERATIONS_LOG.md`:

  ```
  - [13:00] Submitted mahti_qoe_perturbation_smoke.sh (jobid=XXXXX). Reason: smoke before full per L23/L32. Expected outcome: 1 row in qoe_perturbation_smoke.csv.
  ```

- [ ] **Step 3: Wait and verify treatment**

  After job completes, confirm:
  - CSV has at least one row with `success=True` and non-null `f1_macro`, non-null `latency_s` (per L30).
  - The smoke validates the perturbation was applied, not just that the run succeeded (per L38). Assertion script: `python scripts/check_smoke_perturbation_treatment.py results/qoe_perturbation_smoke.csv`. This script reads the smoke CSV and the perturbation log, asserts the topic mask was respected (or latency injection added the expected offset).

- [ ] **Step 4: If smoke fails, return to Phase 1 of systematic-debugging.** Do NOT proceed to full run. (Per L45: never repeat a failing action without changing conditions.)

- [ ] **Step 5: Log smoke outcome to OPERATIONS_LOG before proceeding.**

### Task 1.7: Full perturbation run on Mahti gpusmall

- [ ] **Step 1:** Compose `mahti_qoe_perturbation_full.sh`. Grid: 2 perturbations (`topic_restricted_cal`, `latency_injection`) × 3 strategies (homogeneous-7B, homogeneous-32B, qoe_optimised) × 5 seeds (`42, 123, 456, 789, 0`). Walltime: 4h. `--max-events 1000`. Per-batch progress to `.progress.json` (L63).
- [ ] **Step 2:** Smoke-test one cell of the full grid first (per L32 — the smoke validates the full SLURM template).
- [ ] **Step 3:** Submit full job; log to OPERATIONS_LOG (L55).
- [ ] **Step 4:** Monitor via `.progress.json` (NOT `pgrep -f` per L54).
- [ ] **Step 5:** Validate outputs by content (L30): every cell has `success=True`, non-null F1, non-null cost; treatment-verification re-checks that perturbation was applied.

### Task 1.8: Generate fig:qoe_perturbation and write up

- [ ] **Step 1:** Add `fig_qoe_perturbation()` to `analysis/make_figures.py`. 2-panel: (a) bar chart of F1 by perturbation × strategy with 95% CI; (b) latency CDF for the latency-injection perturbation, before/after.
- [ ] **Step 2:** TDD the figure: write a synthetic input frame, assert the produced PDF has the expected number of bars / CDF curves (use a content-validation test, not just file-exists per L30/L60).
- [ ] **Step 3:** Generate the figure: `python analysis/make_figures.py --figure qoe_perturbation`.
- [ ] **Step 4:** Insert §5.9.1 in `txt/Results.tex` with the figure cite + 1-paragraph narrative.
- [ ] **Step 5:** Commit + recompile manuscript + verify cross-refs.

---

## Task 2: H4 — Calibration-fraction sweep

### Task 2.1: Expose `--calibration-fraction` flag

**Files:**
- Modify: `Experiments/neural-router/scripts/run_qoe.py` — add `--calibration-fraction FLOAT` to argparse, default `0.10`. Thread to `QoEAssigner.__init__`.
- Test: `Experiments/neural-router/tests/test_qoe_calibration_fraction.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cli_calibration_fraction_flag(tmp_path):
    """Treatment-verification (L38, L53): the --calibration-fraction flag
    must be both set AND consumed. We assert the QoEAssigner sees it via
    the CSV calibration_fraction column."""
    import subprocess, pandas as pd
    out = tmp_path / "out.csv"
    subprocess.run([
        "python", "scripts/run_qoe.py",
        "--dataset", "D1", "--dry-run",
        "--calibration-fraction", "0.20",
        "--output", str(out),
    ], check=True)
    df = pd.read_csv(out)
    assert (df["calibration_fraction"] == 0.20).all(), (
        "L53: --calibration-fraction set but not consumed. "
        f"Found {df['calibration_fraction'].unique()}"
    )
```

- [ ] **Step 2: Verify RED.** Expected: KeyError or AssertionError because the column doesn't exist or value isn't propagated.
- [ ] **Step 3: Implement.** Add the flag, thread it into `QoEAssigner`, ensure each output row records `calibration_fraction`. Per L53: write the smoke that observes the value end-to-end, do not trust env-var-style plumbing.
- [ ] **Step 4: Verify GREEN.**
- [ ] **Step 5: Commit.**

### Task 2.2: Smoke + full sweep (mirrors Task 1.6 + 1.7)

- [ ] **Step 1:** `mahti_qoe_calfrac_smoke.sh`: 1 fraction × 1 seed × 1 strategy on D1 max-events 200. Validate treatment.
- [ ] **Step 2:** `mahti_qoe_calfrac_full.sh`: 4 fractions × 3 strategies × 5 seeds × 1 backend tier (Qwen 2.5 7B+32B). Walltime 3h.
- [ ] **Step 3:** Smoke + full as in Task 1, with OPERATIONS_LOG entries each step.
- [ ] **Step 4:** Treatment-verification: confirm each output row's `calibration_fraction` column matches the sweep value (L38 + L53).

### Task 2.3: Figure + writeup

- [ ] **Step 1:** Add `fig_qoe_calfrac()` to `make_figures.py`. 1-panel: F1 (y) vs calibration_fraction (x), one line per strategy with 95% CI ribbons. Hypothesis: QoE-optimised's line crosses round-robin's somewhere in the 0.20–0.50 range, validating "calibration-noise-limited".
- [ ] **Step 2:** TDD as in Task 1.8.
- [ ] **Step 3:** Insert §5.9.2 in `txt/Results.tex` with the figure cite + 1-paragraph narrative.
- [ ] **Step 4:** Commit + recompile.

---

## Task 3: R5 — Reseeding tab:cross-dataset (or honest spec update)

### Task 3.1: Decision point — reseed or update spec

- [ ] **Step 1: Read the current spec**

  Run: `grep -n "5 seeds\|n=5\|three seeds\|n=3" Manuscripts/Neural\ Router\ \(Elsevier\ FGCS\)/txt/Experiment.tex`

- [ ] **Step 2: Check existing data**

  Run: `python scripts/inspect_seeds.py results/cross_dataset/` (CREATE this small inspection script if absent: prints unique seeds per (dataset, backend, config) cell). If 3 seeds exist, decide between reseed (~5h Mahti, more rigorous) or spec update (~1 min, honest but reduces statistical power claim).

- [ ] **Step 3: Surface decision to human partner.** Default: reseed if Mahti quota ≥ 50 BU; update spec if quota tight. Per L3 (use plan mode for major tasks): the cost-vs-rigor trade is a judgment call; surface to the user with the BU estimate.

### Task 3.2 (Path A: Reseed)

- [ ] **Step 1:** TDD: write a smoke test that runs `scripts/run_experiment.py` for one (dataset, backend, config) cell at seeds 123 and 456 (the missing two) with `--resume` (avoids re-running existing seeds 42, 789, 0). Verify the smoke output rows appear in the existing CSV.
- [ ] **Step 2:** `mahti_reseeding_smoke.sh` (1 cell, 2 seeds), then `mahti_reseeding_full.sh` (all D2/D3 cells, 2 seeds each). Walltime 5h.
- [ ] **Step 3:** Aggregate: `python scripts/aggregate_seeds.py` — produces a new `tab:cross-dataset` data row dataset with mean ± half-CI95 over n=5 seeds.
- [ ] **Step 4:** Update §4.10 to remove the "n=3 for cross-dataset" disclosure (no longer needed). Recompile.

### Task 3.2 (Path B: Spec update only)

- [ ] **Step 1:** Edit §4.10 in `txt/Experiment.tex` to honestly state: "Cross-dataset comparison cells (Suppl.\ Tab.~1) use n=3 seeds rather than the n=5 of `tab:ablation`, due to a 3h-per-task SLURM walltime budget on shared HPC partitions; we report mean ± half-CI95 with the n=3 caveat."
- [ ] **Step 2:** No compute. Compile + done.

---

## Task 4: R6 — Cost-model 17% miss stratified analysis

This is local-only, no Mahti compute. Pure analysis on the existing parquet.

### Task 4.1: TDD the stratification

**Files:**
- Create: `Experiments/neural-router/analysis/cost_validation_stratified.py`
- Test: `Experiments/neural-router/tests/test_cost_validation_stratified.py`

- [ ] **Step 1: Write the failing test (with synthetic frame)**

```python
def test_stratified_miss_split_trivial_vs_nontrivial():
    """R6: the 17% miss cells must be split by m_c <= b_max(c)
    (trivial regime where I_pred = I_meas = 1 mechanically) vs
    m_c > b_max(c) (non-trivial regime where the cost model is
    actually under test)."""
    import pandas as pd
    from analysis.cost_validation_stratified import stratify_misses
    df = pd.DataFrame([
        {"cluster": 0, "m_c": 5, "b_max": 10, "I_pred": 1, "I_meas": 1},   # trivial, hit
        {"cluster": 1, "m_c": 50, "b_max": 10, "I_pred": 5, "I_meas": 5},  # non-trivial, hit
        {"cluster": 2, "m_c": 50, "b_max": 10, "I_pred": 5, "I_meas": 12}, # non-trivial, MISS (>2x)
        {"cluster": 3, "m_c": 5, "b_max": 10, "I_pred": 1, "I_meas": 3},   # trivial, MISS
    ])
    summary = stratify_misses(df)
    assert summary["n_total"] == 4
    assert summary["n_miss"] == 2
    assert summary["n_miss_trivial"] == 1
    assert summary["n_miss_nontrivial"] == 1
    assert "miss_rate_nontrivial" in summary
```

- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement** `stratify_misses(df)`: defines miss as `|I_pred/I_meas − 1| > 1`, splits by `m_c > b_max`, returns counts, miss rate, and per-stratum median ratio.
- [ ] **Step 4: Verify GREEN.**
- [ ] **Step 5:** Run on the actual parquet: `python analysis/cost_validation_stratified.py results/cost_validation/per_cluster.parquet`. Output: a 1-paragraph stratified summary suitable for §5.4.

### Task 4.2: Insert summary into manuscript

- [ ] **Step 1:** Read the analysis output. Identify whether the 17% miss is dominated by trivial-regime cells (good — model passes the meaningful test) or non-trivial-regime cells (concerning — cost-model has failure modes worth discussing).
- [ ] **Step 2:** Insert 3-4 sentences into `txt/Results.tex` §5.4 immediately after the existing "83% within factor-of-two band" claim, with the stratified breakdown. If the non-trivial miss rate is high, add a one-sentence honest caveat to §6.4 Limitations.
- [ ] **Step 3:** Recompile, verify cross-refs.

---

## Cross-Issue Synthesis

After Tasks 1–4 land:

- [ ] **Conflict check:** read the modified §5.9 and §5.4 in sequence. Does the perturbation result (§5.9.1) contradict or strengthen the calibration-fraction-noise claim (§5.9.2)? Hypothesis: yes — the topic-restricted-calibration perturbation is a parametric instance of the calibration-quality dimension, so the two results jointly support a stronger story than either alone.
- [ ] **Narrative arc:** re-read abstract → intro → §5.9 → conclusion. Does the new perturbation result need an abstract hook? If the result strengthens the QoE story, add one sentence to the QoE side of the abstract.
- [ ] **Page budget:** the existing 32 pp main matter expects to grow by ~1 pp from §5.9.1 + §5.9.2 + R6 paragraph. Compensating compression must come from §6.2 Discrimination Capacity (per the round-1 revision summary).

## Verification — End-to-End

Before marking the round-2 plan complete:

- [ ] All BOUNDED FORMAL tasks (C9, H4, R5, R6) have a result CSV + figure + manuscript paragraph.
- [ ] Every smoke + full SLURM run logged in OPERATIONS_LOG.md within the same day (L55).
- [ ] No silent failures: per L30, every CSV inspected for `success=True` rate, non-null metrics, expected row count.
- [ ] Treatment-verification (L38) passed for each perturbation cell (topic mask respected, latency injected, backend failed).
- [ ] `pytest -q tests/` all green (L31).
- [ ] Page count under 30 pp main matter (or with explicit compression strategy documented in revision summary).
- [ ] Cross-refs (`\cref{}` and supplementary-material textual refs) all resolve in `pdflatex` output.

## Anti-Patterns to Avoid (from Tasks/lessons.md)

- **L23/L32:** never skip from unit-smoke directly to full Mahti run.
- **L31:** every src/ change has a test that failed before the implementation existed.
- **L38:** every perturbation has a treatment-verification assertion in its smoke test, not just an outcome assertion.
- **L39:** perturbation hooks `raise` on invalid input, never log-and-continue.
- **L41:** the failure-injection perturbation kills exactly one of the two backends (partial degradation), not both (trivial) and not none (off-path).
- **L45:** if the smoke fails twice, do NOT submit a third attempt without a written explanation of what changed.
- **L47:** read EXPERIMENT_PROTOCOL.md before every action.
- **L51:** the perturbation driver must propagate cell-level failures to the aggregate; never silently report 100% success.
- **L53:** every CLI flag added must be smoke-tested end-to-end; the CSV must visibly carry the flag's value.
- **L54:** monitor jobs via `.progress.json`, not `pgrep -f`.
- **L55:** OPERATIONS_LOG entries before walking away from the keyboard.
- **L60:** every new figure must be tied to a specific manuscript claim; otherwise drop it.
- **L61:** any new synthetic-data helper preserves the description-aware F1 invariants tested in I4.
- **L63:** every long run emits per-batch progress to `.progress.json`.

## Execution Handoff

Plan complete and saved to `Experiments/neural-router/docs/plans/2026-05-06-taas-round-2-experiments.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, two-stage review between tasks, fast iteration. Best for the four-task structure here because tasks are independent and each has a tight TDD scope.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batched with checkpoints. Better if the user wants to watch the smoke runs land before authorising the full Mahti runs.

**Recommended:** Subagent-Driven for Tasks 1, 2, 4 (independent TDD scope); inline for Task 3 path-decision (judgment call belongs to the human-in-the-loop). After each subagent reports a task complete, two-stage code review per `superpowers:requesting-code-review` before submitting the SLURM full run.
