# Figure Repair + L23-Compliant Smoke Methodology Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Apply superpowers:test-driven-development at every implementation step (assert data invariants in pytest BEFORE editing the figure code or caption). Apply superpowers:verification-before-completion before checking off any step. Apply superpowers:systematic-debugging if any verification fails. Apply L60: every figure must justify a manuscript claim — the question is never "fix or drop" before "what claim does this support, and is the claim load-bearing?"

**Goal:** Bring every figure in `Manuscripts/Neural Router (Elsevier FGCS)` into provable agreement with the underlying CSV data and the caption + body text that reference it; close the orphan-figure problem by tying every figure to its load-bearing claim; restore promised but un-logged data (per-cluster invocations for fig:cost-validation); and replace the failed single-point "smokes" of 2026-05-04 with proper L23 unit/integration smokes before any further GPU spend.

**Architecture:** Four workstreams executed roughly in series so that downstream tasks see the upstream fixes. (1) **Figure-claim audit (recorded in §Figure-claim map below)** — every figure is mapped to its manuscript promise; load-bearing figures are kept and repaired, never dropped. (2) **Logging gap fix + small re-run** — RouterStats gains `per_cluster_invocations` / `per_cluster_events` / `per_cluster_active_subs`, and a ~12-BU Mahti gputest run reproduces the cost-model validation data the §3.5 prose promises. (3) **Data-vs-figure repair hardened by pytest assertions** — every figure gets a test in `tests/test_figure_data_consistency.py` pinning the invariants its caption claims, then the figure code and caption are repaired together. (4) **L23 smokes** — replace the broken single-point smokes with unit (8 events, no GPU, <60 s) → integration (50 events, gputest, <3 min) → full-proxy (one production task, gputest, <15 min) levels gated behind explicit success of the previous level.

## Figure-claim map (L60 audit, 2026-05-04)

| Figure | Manuscript promise (§) | Claim it justifies | Status | Action |
|---|---|---|---|---|
| `fig:k-sensitivity` | §4.5 Parameter Sensitivity | k=19 is the chosen operating point; F1/I trade-off shape | Orphan — needs prose tie | Add `\cref` + one-sentence claim in §4.5 |
| `fig:tau-sensitivity` | §4.5 + §5 | τ≈0.3 plateau due to high-d cosine concentration | Tied (Discussion.tex L36) | Keep |
| `fig:kappa-sensitivity` | §4.5 | Any fixed κ is a compromise; κ=3 chosen | Orphan — needs prose tie | Add `\cref` + claim |
| `fig:embedding-sensitivity` | §4.5 | Embedding choice has modest F1 impact (LLM is final arbiter) | Orphan — needs prose tie | Add `\cref` + claim |
| `fig:cost-validation` | §3.5/§4.6 explicitly promise "I_meas vs I_pred" | Cost model Eq. 4 is empirically accurate | **Data gap** — per-cluster I and m_c not logged | **Add logging, re-run small representative campaign (~12 BU), then plot honestly** |
| `fig:scaling` | §3.6 (empirical 50–2000 + analytical 5000–100000) | F1 behaviour with `\|E\|`; cost grows predictably | **Caption-vs-data mismatch** — caption claims F1 stable, data shows decline; 5000-event point methodology mismatch | Rewrite caption to match data (F1 declines as batch density rises); drop inconsistent empirical 5000 point; ADD analytical-projection curve per §3.6 promise |
| `fig:pareto` | §4.7 Accuracy-Cost Trade-off | Backends + configs occupy interpretable Pareto positions | **Data interpretation suspect** — hardcoded events 100/1000, latency_s interpretation unclear | Investigate `latency_s` semantics in `src/llm_async.py`; fix event counts per actual `MAX_EVENTS`; re-plot honestly |
| `fig:d2-discrim` | §4.8 Cross-dataset + dual-factor discrimination | Parameter × training-generation joint determines `\|S\|_cross` | Tied (Results.tex L156) | Keep |
| `fig:crossover` (planned, smokes pending) | §4.7 Crossover Validation | A4 beats A0 above context-window threshold | Pending data | Implement after L23 smokes pass |
| `fig:qoe` (planned, smokes pending) | §5 Contribution C5 | QoE-aware mixed strategies dominate homogeneous Pareto | Pending data | Implement after L23 smokes pass |

**No figure is dropped.** Every figure justifies a load-bearing claim. The wrong ones need data, honest captions, or prose ties — not removal. Per L60, "fix or drop" is the wrong frame; the right frame is "what claim does this support, and is the data sufficient to honestly plot it?"

**Tech Stack:** pytest, pandas (data assertions), matplotlib (figure regeneration), pdftotext / regex (LaTeX caption checks), bash + SLURM (smoke jobs on Mahti gputest).

**Estimated cost:** Zero CSC spend until smokes pass. Cost-model-validation re-run with per-cluster logging: one Mahti gputest job at ~30 min × 8 BU/GPU-h ≈ 4 BU (one config × one seed × one dataset is enough — the variation we want is across configs A0–A4 and across (k, |S|), all available in a single ~30-min job that re-runs the existing k-sensitivity D1 sweep with the new logging). Smoke layer 3 (parametric proxy) is at most 1 GPU-h × 8 BU/GPU-h ≈ 8 BU per family (crossover + QoE) — total ≤ 16 BU. Combined: ~20 BU < 0.01 % of the 240 k envelope.

---

## Scope Check

This plan covers two independent subsystems (figures and smoke methodology) that share one constraint: both must complete before the user can trust the manuscript or re-launch GPU jobs. They are sequenced rather than split because the smoke methodology depends on the QoE `--max-events` flag whose introduction is itself a bite-sized TDD task. Splitting would only duplicate the operations-log book-keeping.

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `tests/test_figure_data_consistency.py` | **CREATE** | TDD: one test per figure asserting the data invariant the caption claims (e.g., "F1 monotone with `\|E\|`", "throughput is `events / latency_s` with realistic per-call times", "predicted I matches `Σ ⌈mc/bmax⌉`") |
| `analysis/make_figures.py` | **MODIFY** (functions: `fig_scaling`, `fig_cost_validation`, `fig_pareto`, plus minor cleanup) | Make every figure plot what its caption claims, OR remove the figure if no caption can honestly hold |
| `Manuscripts/Neural Router (Elsevier FGCS)/txt/Results.tex` | **MODIFY** | Captions for `fig:scaling`, `fig:cost-validation`, `fig:pareto`; add `\cref{fig:…}` ties (or `\includegraphics` removal) for the 6 orphan figures; remove any figure dropped by §Workstream 1 |
| `Manuscripts/Neural Router (Elsevier FGCS)/txt/Discussion.tex` | **MODIFY (if needed)** | If `fig:scaling`'s narrative changes from "stable F1, linear cost" to "F1 declines with batch density", §5.5 (limitations / future work) must reflect it |
| `Manuscripts/Neural Router (Elsevier FGCS)/figs/*.pdf` | **REGENERATE** | After every `make_figures.py` change |
| `tests/test_run_qoe_max_events.py` | **CREATE** | TDD: `--max-events 8` truncates the event corpus to 8 in `run_qoe.py` |
| `scripts/run_qoe.py` | **MODIFY** | Add `--max-events` argument, propagate to dataset loader |
| `scripts/slurm/mahti_crossover_smoke.sh` | **REWRITE** | Layered: `--smoke-level={unit,integration,full}` with progressively larger event counts and configs |
| `scripts/slurm/mahti_qoe_smoke.sh` | **REWRITE** | Same pattern: `--smoke-level={unit,integration,full}`, progressive event counts |
| `tests/test_smoke_scripts_layered.py` | **CREATE** | Static checks: each smoke script accepts the three levels, the `unit` level passes `--max-events 8` (or equivalent), the `full` level matches the production task |
| `OPERATIONS_LOG.md` | **APPEND** | Log every figure regeneration, smoke run, and outcome (per L55) |

## Pre-flight Checks

- [ ] **PF1: All Mahti GPU jobs are terminal**

  Run: `ssh mahti 'squeue -u $USER'`
  Expected: empty (no in-flight jobs). If a job is still running, do not touch it; confirm whether to cancel.

- [ ] **PF2: Local pytest passes baseline**

  Run: `cd Experiments/neural-router && .venv/bin/pytest -x -q --no-header 2>&1 | tail -20`
  Expected: existing tests green (or only tests known to need GPU/Ollama skipped). If unrelated failures, fix before continuing.

- [ ] **PF3: Capture current figure data invariants for the audit**

  Read each figure-producing CSV and write down (in this plan, in §Workstream 1) the *truth* the caption must claim — F1 range, |I| range, latency range. This is the spec the tests will lock in.

---

## Workstream 1 — Figure data audit and repair

### Task 1.1: Lock down `fig:scaling` truth via pytest

**Files:**
- Create: `tests/test_figure_data_consistency.py`
- Read: `results/puhti_mirror/scaling_by_task/qwen7b_scaling_events_D3/scaling_events_D3.csv`
- Read: `results/puhti_mirror/scaling_by_task/qwen7b_scale_evt5000_D3/scaling_events_D3.csv`

**Caption truth to assert (from CSVs, verified 2026-05-04):**
- F1 across `|E| ∈ {50, 100, 200, 500, 1000, 2000}` is **NOT** stable: F1 = (0.250, 0.152, 0.016, 0.008, 0.007, 0.013).
- The `|E|=5000` point uses ~1.5 events/call (3383 invocations / 5000 events) vs ~5–7 events/call for the 50–2000 sweep — methodology mismatch.
- LLM invocations grow ~linearly with `|E|` for the 50–2000 sweep: I ∈ (14, 18, 34, 71, 144, 276).

**The test asserts these invariants.** If the data ever changes (re-run), the test catches it.

- [ ] **Step 1.1.1: Write the failing test**

```python
# tests/test_figure_data_consistency.py
"""Pin down the invariants that figure captions claim, so the captions cannot
silently desync from the data they describe.

The 2026-05-04 audit found three figures whose captions stated the opposite
of what the data shows; this file is the regression net."""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCALE_50_2000 = ROOT / "results/puhti_mirror/scaling_by_task/qwen7b_scaling_events_D3/scaling_events_D3.csv"
SCALE_5000 = ROOT / "results/puhti_mirror/scaling_by_task/qwen7b_scale_evt5000_D3/scaling_events_D3.csv"


def test_scaling_50_to_2000_f1_is_not_stable():
    """Caption must not claim 'F1 stable' across 50..2000 events: it isn't.
    F1 collapses from 0.25 (|E|=50) to <=0.02 (|E|>=200)."""
    df = pd.read_csv(SCALE_50_2000).sort_values("n_events")
    f1 = df["f1"].tolist()
    assert max(f1) >= 0.20, f"F1 max should reflect the 50-event point: got {f1}"
    assert min(f1) <= 0.02, f"F1 min should reflect the >=200-event collapse: got {f1}"
    # The 50-event F1 must be at least 5x the 1000-event F1: a 'stability' caption is false.
    f1_50 = df.loc[df["n_events"] == 50, "f1"].iloc[0]
    f1_1000 = df.loc[df["n_events"] == 1000, "f1"].iloc[0]
    assert f1_50 / max(f1_1000, 1e-6) >= 5.0, \
        f"F1 collapse must be at least 5x: f1(50)={f1_50}, f1(1000)={f1_1000}"


def test_scaling_5000_uses_different_batching():
    """The 5000-event point uses much smaller batches (~1.5 events/call) vs
    the 50..2000 sweep (~5..7 events/call). Plotting them together without
    documenting this is the methodology mismatch the audit caught."""
    df_low = pd.read_csv(SCALE_50_2000).sort_values("n_events")
    df_high = pd.read_csv(SCALE_5000)
    evt_per_call_low = (df_low["n_events"] / df_low["invocations"]).mean()
    evt_per_call_high = (df_high["n_events"] / df_high["invocations"]).iloc[0]
    assert evt_per_call_low > 3.0, f"low-|E| sweep should average >3 evt/call: {evt_per_call_low}"
    assert evt_per_call_high < 2.0, f"5000-event point must average <2 evt/call: {evt_per_call_high}"


def test_scaling_invocations_grow_linearly_in_50_2000():
    """The cost-model claim 'I grows linearly in |E|' is true for the
    homogeneous-batching 50..2000 sweep. Lock that in."""
    df = pd.read_csv(SCALE_50_2000).sort_values("n_events")
    # Linear regression of I on |E|: should have R^2 >= 0.99
    import numpy as np
    n = df["n_events"].values
    I = df["invocations"].values
    coef = np.polyfit(n, I, 1)
    pred = np.polyval(coef, n)
    ss_res = ((I - pred) ** 2).sum()
    ss_tot = ((I - I.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    assert r2 >= 0.99, f"I should be linear in |E| (50..2000): R^2 = {r2}"
```

- [ ] **Step 1.1.2: Run the test, verify it fails**

  Run: `cd Experiments/neural-router && .venv/bin/pytest tests/test_figure_data_consistency.py -v 2>&1 | tail -30`
  Expected: tests **PASS** (the data already matches these claims). If they fail, the data files have moved — fix paths first.
  *Note: this is "lock-in" rather than RED-GREEN. The test pins truth so any future data change forces a caption review.*

- [ ] **Step 1.1.3: Commit**

  ```bash
  git -C "Experiments/neural-router" add tests/test_figure_data_consistency.py
  git -C "Experiments/neural-router" commit -m "test: lock down fig:scaling data invariants"
  ```

### Task 1.2: Repair `fig:scaling` figure code

**Decision:** Drop `|E|=5000` from the figure (methodology mismatch); honestly plot F1 declining with `|E|` due to per-event-attention-budget shrinkage; relabel the right axis.

**Files:**
- Modify: `analysis/make_figures.py` lines 195–224 (function `fig_scaling`)

- [ ] **Step 1.2.1: Edit `fig_scaling()`**

```python
def fig_scaling():
    """Event-count scaling on D3 (Qwen-2.5-7B, A3) over |E| ∈ {50,100,200,500,1000,2000}.

    The 5000-event point from `qwen7b_scale_evt5000_D3` is intentionally
    excluded because it used a different batch size (~1.5 events/call vs ~5-7
    events/call for the 50..2000 sweep). Including it without the methodology
    caveat would mislead — see tests/test_figure_data_consistency.py.
    """
    csv = SCALE / "qwen7b_scaling_events_D3" / "scaling_events_D3.csv"
    if not csv.exists():
        print("  scaling: NO DATA"); return
    df = pd.read_csv(csv).sort_values("n_events").drop_duplicates("n_events")
    fig, ax1 = plt.subplots(figsize=(3.6, 2.6))
    ax2 = ax1.twinx()
    ax1.plot(df["n_events"], df["f1"], marker="o", color="C0", label="F1")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Number of events $|\mathcal{E}|$")
    ax1.set_ylabel("Macro-F1", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(0, max(df["f1"].max() * 1.2, 0.3))
    ax2.plot(df["n_events"], df["invocations"], marker="s", color="C3", linestyle="--", label="$I$")
    ax2.set_yscale("log")
    ax2.set_ylabel("LLM invocations $I$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "scaling.pdf")
    plt.close(fig)
    print(f"  saved scaling.pdf ({len(df)} event-count points)")
```

- [ ] **Step 1.2.2: Regenerate the figure**

  Run: `cd Experiments/neural-router && .venv/bin/python analysis/make_figures.py 2>&1 | grep -E "(scaling|error|ERROR)"`
  Expected: `saved scaling.pdf (6 event-count points)`. If error, return to systematic-debugging Phase 1.

- [ ] **Step 1.2.3: Visually inspect the new PDF**

  Run: `open "Manuscripts/Neural Router (Elsevier FGCS)/figs/scaling.pdf"` (manual). Confirm: 6 points, F1 declines from ~0.25 to <0.02, I grows linearly. **Do not check off until visually confirmed.**

- [ ] **Step 1.2.4: Commit**

### Task 1.3: Rewrite `fig:scaling` caption + body text

**Files:**
- Modify: `Manuscripts/Neural Router (Elsevier FGCS)/txt/Results.tex` lines 100–105 (caption block)
- Read & possibly modify: `Manuscripts/Neural Router (Elsevier FGCS)/txt/Discussion.tex` (any §5 reference to "stable F1")

- [ ] **Step 1.3.1: Replace the false "F1 stable" caption with the truthful version**

  New caption (replace verbatim in `Results.tex`):

  > Event-count scaling on D3 for configuration A3 with the Qwen-2.5-7B backend, sweeping `|E| ∈ {50, 100, 200, 500, 1000, 2000}` at a fixed events-per-call budget (\~5–7 events/call). F1 (left axis) decreases from 0.25 at `|E|=50` to below 0.02 once `|E| ≥ 200` because the per-event share of the 4096-token attention budget shrinks as more events compete inside one call — a direct prediction of the discrimination-capacity argument of §3.6. LLM invocations $I$ (right axis, log) grow linearly with event count, consistent with the cost model $I = \sum_c \lceil m_c / b_{\max}(c) \rceil$ when batch size is held fixed.

- [ ] **Step 1.3.2: Update Discussion.tex if it references the old "stable F1" claim**

  Run: `grep -n "stable" "Manuscripts/Neural Router (Elsevier FGCS)/txt/Discussion.tex"`. If a hit refers to scaling, rewrite to match the new caption.

- [ ] **Step 1.3.3: Verify caption-vs-data with a quick LaTeX text test**

  Add to `tests/test_figure_data_consistency.py`:
  ```python
  def test_scaling_caption_does_not_claim_stability():
      """fig:scaling caption must not contain 'remains stable' phrasing —
      that was the 2026-05-04 audit finding."""
      tex = (ROOT.parent / "Manuscripts/Neural Router (Elsevier FGCS)/txt/Results.tex").read_text()
      # Find the caption for fig:scaling
      import re
      m = re.search(r"fig:scaling.*?\\caption\{(.+?)\}\s*\\label", tex, re.DOTALL)
      if m is None:
          # caption may be before \label rather than after; search both directions
          m = re.search(r"\\caption\{([^}]*scaling[^}]*)\}.{0,200}fig:scaling", tex, re.DOTALL | re.IGNORECASE)
      assert m, "Could not locate fig:scaling caption block"
      caption = m.group(1)
      assert "remains stable" not in caption.lower(), \
          f"caption still claims F1 stability: {caption!r}"
      assert "decreases" in caption or "declines" in caption or "shrinks" in caption, \
          f"caption should describe the F1 decline: {caption!r}"
  ```
  Run the test, verify GREEN.

- [ ] **Step 1.3.4: Compile manuscript, confirm no LaTeX errors**

  Run: `cd "Manuscripts/Neural Router (Elsevier FGCS)" && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | tail -10`
  Expected: PDF builds. If `Paragraph ended before \end was complete` (L58), find and fix typo immediately.

- [ ] **Step 1.3.5: Commit**

### Task 1.4: Add per-cluster logging to RouterStats (data gap fix for `fig:cost-validation`)

**Per L60: `fig:cost-validation` justifies the empirical-validation promise of §3.5/§4.6. The cost model (Eq. 4) is a load-bearing manuscript claim. The data gap is in the logging path (`RouterStats` only persists aggregates), not in the experiment design. Adding per-cluster logging is a tiny code change; the re-run is ~4 BU.**

**Files:**
- Modify: `src/router.py` lines 90–111 (`RouterStats` dataclass) + the per-cluster increment site (`_match_cluster`, ~lines 688–770)
- Modify: `src/evaluation.py` lines ~167–198 (propagate per-cluster fields into `EvaluationResult`)
- Modify: `scripts/run_experiment.py` lines ~432–500 (write per-cluster fields to result CSV — JSON-encoded list column)
- Create: `tests/test_router_per_cluster_stats.py`

- [ ] **Step 1.4.1: Write the failing test**

```python
# tests/test_router_per_cluster_stats.py
"""Per-cluster invocation logging is required to honestly populate
fig:cost-validation. The cost model claim (Eq. 4) is per-cluster:
I = Σ_c ⌈m_c / b_max(c)⌉. Aggregate I is insufficient.
"""
import pytest
from src.router import NeuralRouter, RouterConfig, RouterStats
from src.data import Subscription, Event


def test_router_stats_records_per_cluster_invocations():
    """RouterStats must expose lists indexed by cluster: invocations,
    events processed, active subscription count after CoverAndMerge.
    """
    s = RouterStats()
    # Fields exist and default to empty lists
    assert s.per_cluster_invocations == []
    assert s.per_cluster_events == []
    assert s.per_cluster_active_subs == []


@pytest.mark.slow
def test_per_cluster_invocations_sum_to_total(small_router_fixture):
    """Per-cluster I_c values must sum to the aggregate llm_invocations.
    This is the cost-model identity Σ_c I_c = I_total.
    """
    router, events = small_router_fixture
    router.match(events)
    s = router.stats
    assert sum(s.per_cluster_invocations) == s.llm_invocations
    assert len(s.per_cluster_invocations) == s.num_clusters
```

- [ ] **Step 1.4.2: Run the test, verify it FAILS**

  Run: `pytest tests/test_router_per_cluster_stats.py::test_router_stats_records_per_cluster_invocations -v`
  Expected: FAIL with `AttributeError: 'RouterStats' object has no attribute 'per_cluster_invocations'`.

- [ ] **Step 1.4.3: Add the three list fields to RouterStats**

```python
@dataclass
class RouterStats:
    # ... existing fields unchanged ...
    per_cluster_invocations: list[int] = field(default_factory=list)
    per_cluster_events: list[int] = field(default_factory=list)
    per_cluster_active_subs: list[int] = field(default_factory=list)
```

  Run the unit test → GREEN.

- [ ] **Step 1.4.4: Increment per-cluster fields in `_match_cluster`**

  At entry of `_match_cluster`, append `len(cluster.queue)` to `per_cluster_events` and `len(cluster.active_subscriptions)` to `per_cluster_active_subs`. Replace the existing `self.stats.llm_invocations += 1` with both:
  ```python
  self.stats.llm_invocations += 1
  self.stats.per_cluster_invocations[-1] += 1  # current cluster's slot
  ```
  Initialize the per-cluster slot at `_match_cluster` entry: `self.stats.per_cluster_invocations.append(0)`.

- [ ] **Step 1.4.5: Run the slow integration test on a stub fixture**

  Define `small_router_fixture` in `tests/conftest.py` if not already: 5 subs, 10 events, k=2, dummy LLM that returns empty matches. Verify `sum(per_cluster_invocations) == llm_invocations`.

- [ ] **Step 1.4.6: Propagate per-cluster fields into `EvaluationResult` and CSV**

  Add three columns to the result CSV header: `per_cluster_invocations` (JSON-encoded list), `per_cluster_events`, `per_cluster_active_subs`. Use `json.dumps()` for serialisation and `json.loads()` when reading back.

- [ ] **Step 1.4.7: Commit**

### Task 1.5: Re-run cost-model validation campaign with per-cluster logging

**Goal:** Produce ~30 (config, k, dataset) cells with per-cluster I, m, |S| logged. Use existing k-sensitivity sweep on D1 (k ∈ {1, 5, 19, 50}) plus ablation A0..A4 on D1, both with seed 42 only — that's ~12 task-rows × ~2-min/task ≈ 25 min wall-clock at one Mahti gputest A100 ≈ 4 BU.

**Files:**
- Create: `scripts/slurm/mahti_cost_validation.sh`
- Modify: `analysis/make_figures.py` `fig_cost_validation` (rewrite to use per-cluster columns)

- [ ] **Step 1.5.1: Write SLURM script** (gputest, 30-min walltime, single task — no array)

- [ ] **Step 1.5.2: Smoke locally with stub LLM** (verify CSV has per-cluster columns populated, sum matches aggregate)

- [ ] **Step 1.5.3: Submit on Mahti gputest, monitor**

- [ ] **Step 1.5.4: Verify CSV per L30** — non-empty, per-cluster lists are non-empty, lengths match `num_clusters`

- [ ] **Step 1.5.5: Rewrite `fig_cost_validation()` to plot honest predicted-vs-measured per cluster**

```python
def fig_cost_validation():
    """Predicted vs measured I per cluster, across the cost-validation
    re-run (config × k × dataset cells). Predicted I_c = ⌈m_c / b_max(|S_c|)⌉
    where b_max(|S_c|) = floor((W - t_inst - |S_c|·t_s - t_resp) / t_e),
    using the production constants W=4096, t_inst=200, t_resp=500, t_s=80,
    t_e estimated from per-row tokens_prompt / sum(per_cluster_events)."""
    csv = Path("results/cost_validation/cost_validation_D1.csv")
    if not csv.exists():
        print("  cost_validation: NO DATA (re-run pending)"); return
    import json, math
    df = pd.read_csv(csv)
    rows = []
    for _, r in df.iterrows():
        per_inv = json.loads(r["per_cluster_invocations"])
        per_evt = json.loads(r["per_cluster_events"])
        per_sub = json.loads(r["per_cluster_active_subs"])
        avg_tok_per_evt = r["tokens_prompt"] / max(sum(per_evt), 1) / 1.5  # rough t_e
        for I_meas, m, S_c in zip(per_inv, per_evt, per_sub):
            avail = 4096 - 200 - S_c * 80 - 500
            b_max = max(1, math.floor(avail / max(avg_tok_per_evt, 10)))
            I_pred = math.ceil(m / b_max) if m > 0 else 0
            rows.append({"config": r["config"], "I_meas": I_meas, "I_pred": I_pred})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(plot_df["I_meas"], plot_df["I_pred"], alpha=0.6, s=20)
    lims = [1, max(plot_df[["I_meas","I_pred"]].max().max() * 1.5, 100)]
    ax.plot(lims, lims, "k--", alpha=0.5, label="$I_{pred}=I_{meas}$")
    ax.fill_between(lims, [x*0.9 for x in lims], [x*1.1 for x in lims], alpha=0.15, color="grey", label=r"$\pm 10\%$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Measured invocations $I_{meas}$ per cluster")
    ax.set_ylabel("Predicted invocations $I_{pred}$ per cluster")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "cost_validation.pdf"); plt.close(fig)
    print(f"  saved cost_validation.pdf ({len(plot_df)} per-cluster points)")
```

- [ ] **Step 1.5.6: Add data-invariant test** — `(I_pred / I_meas).median()` is in [0.7, 1.4] (the model is order-of-magnitude correct; tight ±10% band is for visual emphasis only)

- [ ] **Step 1.5.7: Update caption to honestly describe the per-cluster scatter**

- [ ] **Step 1.5.8: Compile + commit**

### Task 1.6: Repair `fig:pareto` (data interpretation, then plot)

**Per L60: §4.7 Accuracy-Cost Trade-off is the central operational claim. The figure is load-bearing — keep, repair, do not drop.**

**Files:**
- Modify: `analysis/make_figures.py` lines 229–279 (`fig_pareto`)
- Read: `src/router.py:570–605`, `src/llm_async.py` (`latency_s` semantics)

- [ ] **Step 1.6.1: Read `src/llm_async.py` and `src/router.py:520–610` to determine what `total_time_s` measures for async LLM clients.**

  Record findings (wall-clock vs sum-of-calls; whether async-batch concurrency reduces wall-clock).

- [ ] **Step 1.6.2: Determine actual `MAX_EVENTS` used per backend × dataset.**

  Already verified for Qwen-7B (D1=1000, D2=300, D3=1000). For Haiku and Sonnet, grep the laptop run logs / OPERATIONS_LOG / any `run_*.py` invocation for the actual `--max-events` flag used. Record per backend.

- [ ] **Step 1.6.3: Decide the throughput axis.**

  Two plausible definitions:
  - (a) Wall-clock throughput: `events / latency_s` — interpretable as "how many events does this backend process per second of pipeline wall-clock"; honest if `latency_s` is wall-clock.
  - (b) Cost-normalised throughput: `events / cost_usd` — F1 vs `$ / 1000 events` — interpretable for §4.7 directly.
  
  Choose (b) because it ties cleanly to the cost model and avoids the `latency_s`-as-throughput interpretation question. Record decision here before checking off.

- [ ] **Step 1.6.4: Write the data-invariant test first (TDD)**

```python
def test_pareto_data_uses_actual_max_events():
    """The Pareto figure must not assume 100/1000 hardcoded events: it
    must read the actual MAX_EVENTS for each backend × dataset cell."""
    # The repaired fig_pareto should expose the data it uses (e.g., write a sidecar
    # JSON with the events count it used, OR compute throughput from cost rather
    # than from a hardcoded events number)
    sidecar = ROOT.parent / "Manuscripts/Neural Router (Elsevier FGCS)/figs/pareto_data.json"
    if not sidecar.exists():
        pytest.skip("regenerate fig_pareto first")
    import json
    data = json.loads(sidecar.read_text())
    # Haiku/Sonnet/Qwen-7B all use MAX_EVENTS that depends on dataset; assert
    # the sidecar records per-backend events != 100 universally
    events_per_backend = {row["backend"]: row["events"] for row in data}
    assert all(v >= 100 for v in events_per_backend.values())
    assert len(set(events_per_backend.values())) >= 1, "events should reflect actual run config"
```

- [ ] **Step 1.6.5: Rewrite `fig_pareto()` to plot F1 vs $/1000 events (cost-normalised throughput).**

  Use `cost_per_1k` directly from the CSV (already computed by `evaluation.py`). For Qwen, cost is hardware-only — compute or cite a reasonable per-token rate (the manuscript already does this in §4.6 worked example). Drop the hardcoded `events` constant entirely.

- [ ] **Step 1.6.6: Visually verify; rewrite caption to match.**

- [ ] **Step 1.6.7: Add a `\cref{fig:pareto}` to §4.7 prose to remove the orphan status.**

- [ ] **Step 1.6.8: Compile, commit.**

### Task 1.7: Tie sensitivity orphans + add analytical projection to `fig:scaling`

**Per L60: `fig:k-sensitivity`, `fig:kappa-sensitivity`, `fig:embedding-sensitivity` each support a load-bearing claim in §4.5 — they're orphans only because §4.5 prose was never written. Add prose ties; do not drop. `fig:scaling` also needs the §3.6-promised analytical-projection curve added.**

**Files:**
- Modify: `Manuscripts/Neural Router (Elsevier FGCS)/txt/Results.tex` §4.5 prose
- Modify: `analysis/make_figures.py` `fig_scaling` (add analytical projection curve)

- [ ] **Step 1.7.1: Add §4.5 prose tying each sensitivity figure to its claim.**

  Insert after the §4.5 subsection header in `Results.tex`. Example wording:
  
  > Across the four sensitivity dimensions, the chosen operating point ($k=19$, $\tau{=}0.1$, $\kappa{=}3$, embedding all-MiniLM-L6-v2) sits at or near the F1 maximum while keeping LLM invocation count $I$ low. \Cref{fig:k-sensitivity} shows that F1 saturates around $k=19$ on D1 (one cluster per native subscription) while $I$ grows roughly linearly in $k$; \cref{fig:tau-sensitivity} shows the cosine-threshold plateau above $\tau \approx 0.3$ that justifies the operating choice $\tau{=}0.1$ in the high-recall regime; \cref{fig:kappa-sensitivity} shows that $\kappa{=}3$ minimises the FP/recall trade-off on the variable-label-density D1; and \cref{fig:embedding-sensitivity} shows the modest spread (\textless0.05 F1) across mainstream sentence-transformer embeddings, justifying our choice of MiniLM as the lightweight default.

- [ ] **Step 1.7.2: Add analytical-projection curve to `fig_scaling()`** (per §3.6 promise of `|E| ∈ {5000, 10000, 50000, 100000}` projection)

  Project I from the linear regression already locked in by `test_scaling_invocations_grow_linearly_in_50_2000`: I_pred = a × |E| + b. Plot as a dashed extension of the I curve from |E|=2000 onwards. Annotate as "analytical projection". The empirical 5000-point is dropped per Task 1.2 (methodology mismatch); the analytical curve is the §3.6 deliverable.

- [ ] **Step 1.7.3: Update fig:scaling caption to mention analytical projection.**

- [ ] **Step 1.7.4: Add the orphan-detector test** (already in §1.7 of the original plan):

```python
def test_no_orphan_figures():
    """Every figure that survives the 2026-05-04 audit must be \\cref'd
    from at least one body file."""
    import re
    tex_dir = ROOT.parent / "Manuscripts/Neural Router (Elsevier FGCS)/txt"
    main_tex = ROOT.parent / "Manuscripts/Neural Router (Elsevier FGCS)/main.tex"
    all_text = "\n".join(p.read_text() for p in tex_dir.glob("*.tex"))
    all_text += main_tex.read_text()
    labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", all_text))
    referenced = set(re.findall(r"\\(?:cref|Cref|ref)\{(fig:[^}]+)\}", all_text))
    orphans = labels - referenced
    assert not orphans, f"orphan figures (defined but not referenced): {orphans}"
```

- [ ] **Step 1.7.5: Compile, commit.**

### Task 1.8: Run all figure-consistency tests + visually verify the manuscript

- [ ] **Step 1.8.1:** `pytest tests/test_figure_data_consistency.py tests/test_router_per_cluster_stats.py -v`. All green.
- [ ] **Step 1.8.2:** `latexmk -pdf -interaction=nonstopmode main.tex`. Zero errors, zero "undefined reference" warnings for figures.
- [ ] **Step 1.8.3:** Open the produced `main.pdf` and skim §4 (Results) and §5 (Discussion). Confirm no figure feels stale or unmoored, and that every claim that needs a figure has one.

---

## Workstream 2 — `--max-events` for `run_qoe.py` (TDD)

### Task 2.1: Add `--max-events` to `run_qoe.py`

**Files:**
- Create: `tests/test_run_qoe_max_events.py`
- Modify: `scripts/run_qoe.py`

- [ ] **Step 2.1.1: Write the failing test**

  ```python
  # tests/test_run_qoe_max_events.py
  """Verify that scripts/run_qoe.py respects --max-events for L23 unit smoke."""
  import subprocess
  import sys
  from pathlib import Path

  ROOT = Path(__file__).resolve().parents[1]


  def test_run_qoe_help_lists_max_events():
      """The --max-events flag must be visible in --help (otherwise smokes
      can't reduce the corpus)."""
      out = subprocess.run(
          [sys.executable, "scripts/run_qoe.py", "--help"],
          cwd=ROOT, capture_output=True, text=True, check=True,
      )
      assert "--max-events" in out.stdout, out.stdout
  ```

- [ ] **Step 2.1.2: Run the test, verify it FAILS**

  Run: `cd Experiments/neural-router && .venv/bin/pytest tests/test_run_qoe_max_events.py -v 2>&1 | tail -20`
  Expected: FAIL with `--max-events` not found in help.

- [ ] **Step 2.1.3: Implement `--max-events` in `run_qoe.py`**

  Read the current arg parser, add `parser.add_argument("--max-events", type=int, default=None, help="Truncate the dataset to this many events (smoke testing). Default: full corpus.")`. Propagate to dataset loader: `if args.max_events is not None: dataset.events = dataset.events[: args.max_events]`.

- [ ] **Step 2.1.4: Verify test passes**

- [ ] **Step 2.1.5: Add a behavioral test (1 event must run end-to-end with a stub backend)**

  Marked `@pytest.mark.slow`. Skip on CI by default; run manually before claiming done.

- [ ] **Step 2.1.6: Commit**

---

## Workstream 3 — L23-compliant smokes for crossover and QoE

### Task 3.1: Layered crossover smoke

**Files:**
- Create: `tests/test_smoke_scripts_layered.py`
- Rewrite: `scripts/slurm/mahti_crossover_smoke.sh`

**Three levels:**
- **Unit** (target: <60 s on a CPU): Python-only, no Ollama. Runs the `_subsample_subscriptions` + cluster + cost-model-prediction path on 8 events. **No SLURM.** Lives in `tests/`.
- **Integration** (target: <3 min on Mahti gputest): 1 task = `(|S|=50, A0+A4, seed 42)` with `--max-events 50`. Real Ollama, real qwen2.5:7b. SLURM gputest 5-min walltime.
- **Full proxy** (target: <15 min on Mahti gputest): 1 task at `(|S|=200, A0+A4, seed 42)` with `--max-events 1000` (the actual D1 cap). Validates the production task before launching the 36-task array.

- [ ] **Step 3.1.1: Write the layered-smoke static check test (FAILS until rewrite)**

```python
# tests/test_smoke_scripts_layered.py
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / "scripts/slurm/mahti_crossover_smoke.sh"
QOE = ROOT / "scripts/slurm/mahti_qoe_smoke.sh"


def _shell(p): return p.read_text() if p.exists() else ""


def test_crossover_smoke_supports_three_levels():
    s = _shell(CROSS)
    assert "SMOKE_LEVEL" in s, "crossover smoke must take a SMOKE_LEVEL parameter"
    assert "unit" in s and "integration" in s and "full" in s, \
        "crossover smoke must define unit, integration, full levels"


def test_crossover_unit_smoke_is_cheap():
    s = _shell(CROSS)
    # Unit level should NOT call sbatch ollama (it is python-only) — gate by
    # SMOKE_LEVEL=unit selecting --max-events 8 and skipping ollama.
    assert "--max-events 8" in s or "MAX_EVENTS=8" in s, \
        "unit-level crossover smoke must use --max-events 8"


def test_crossover_integration_smoke_uses_50_events():
    s = _shell(CROSS)
    assert "--max-events 50" in s or "MAX_EVENTS=50" in s, \
        "integration-level crossover smoke must use --max-events 50"


def test_qoe_smoke_supports_three_levels():
    s = _shell(QOE)
    assert "SMOKE_LEVEL" in s
    assert "unit" in s and "integration" in s and "full" in s


def test_qoe_unit_smoke_uses_max_events_8():
    s = _shell(QOE)
    assert "--max-events 8" in s or "MAX_EVENTS=8" in s


def test_qoe_integration_smoke_uses_max_events_50():
    s = _shell(QOE)
    assert "--max-events 50" in s or "MAX_EVENTS=50" in s
```

  Run: `pytest tests/test_smoke_scripts_layered.py -v` → FAIL.

- [ ] **Step 3.1.2: Rewrite `mahti_crossover_smoke.sh` to take `SMOKE_LEVEL`**

  Skeleton:
  ```bash
  #!/bin/bash
  #SBATCH --job-name=nrouter-cross-smoke
  # ... usual SBATCH ...
  set -eo pipefail
  SMOKE_LEVEL=${SMOKE_LEVEL:-integration}  # unit | integration | full
  case "$SMOKE_LEVEL" in
    unit) MAX_EVENTS=8;   N_SUBS=50;  CONFIGS=A0    ;;  # python-only fast path
    integration) MAX_EVENTS=50;  N_SUBS=50;  CONFIGS=A0,A4 ;;
    full) MAX_EVENTS=1000; N_SUBS=200; CONFIGS=A0,A4 ;;
    *) echo "unknown SMOKE_LEVEL=$SMOKE_LEVEL" >&2; exit 2 ;;
  esac
  # ... environment setup, ollama serve (skip for SMOKE_LEVEL=unit), pull ...
  python scripts/run_crossover.py --dataset D1 --configs $CONFIGS \
      --sub-volumes $N_SUBS --seeds 42 --max-context-tokens 4096 \
      --llm-model ollama/qwen2.5:7b --output-dir $SMOKE_OUT \
      --batch-size 50 --llm-timeout 600 --max-events $MAX_EVENTS
  ```

- [ ] **Step 3.1.3: Run the static test, verify GREEN**

- [ ] **Step 3.1.4: Add `--max-events` to `scripts/run_crossover.py` if not already present (TDD: test first)**

  Mirrors §2.1 for `run_qoe.py`.

- [ ] **Step 3.1.5: Commit**

### Task 3.2: Layered QoE smoke

Same pattern as §3.1 for `mahti_qoe_smoke.sh`. Each level invokes `run_qoe.py` with `--max-events {8, 50, full}` plus progressively more strategies + presets.

- [ ] **Step 3.2.1: Rewrite `mahti_qoe_smoke.sh`** (test already failing from §3.1.1)
- [ ] **Step 3.2.2: Run static tests, verify GREEN**
- [ ] **Step 3.2.3: Commit**

### Task 3.3: Run unit smokes locally (no SLURM, no GPU)

- [ ] **Step 3.3.1: Run unit-level crossover locally**

  Run: `SMOKE_LEVEL=unit bash scripts/slurm/mahti_crossover_smoke.sh` (gated to skip the ollama serve when `SMOKE_LEVEL=unit`, OR run a python-only equivalent: `.venv/bin/python scripts/run_crossover.py --dataset D1 --configs A0 --sub-volumes 50 --seeds 42 --max-context-tokens 4096 --llm-model dummy --output-dir /tmp/cross-unit --max-events 8`).
  Expected: <60 s, CSV row produced, F1 column non-null.

- [ ] **Step 3.3.2: Run unit-level QoE locally**

  Same pattern for QoE.

- [ ] **Step 3.3.3: Verify both unit smokes succeed before submitting integration smoke**

### Task 3.4: Submit integration smokes to Mahti gputest

- [ ] **Step 3.4.1:** `ssh mahti 'sbatch --export=ALL,SMOKE_LEVEL=integration scripts/slurm/mahti_crossover_smoke.sh'`. Wait for completion (notify).
- [ ] **Step 3.4.2:** Verify CSV content (per L30): non-empty, F1 column non-null, latency reasonable.
- [ ] **Step 3.4.3:** Same for QoE.

### Task 3.5: Submit full proxy smokes (only after §3.4 passes)

- [ ] **Step 3.5.1:** Submit `SMOKE_LEVEL=full` for each. Verify.
- [ ] **Step 3.5.2:** **Only then** submit the production arrays.

---

## Workstream 4 — Cleanup

### Task 4.1: Update `OPERATIONS_LOG.md` with full session record

- [ ] Per L55: append every step taken to `Experiments/neural-router/OPERATIONS_LOG.md` and to `~/OPERATIONS_LOG.md` on Mahti (when remote).

### Task 4.2: Update memory

- [ ] If anything is durable across sessions (e.g., "scaling figure now drops 5000-event point — methodology mismatch documented"), add to memory under a `feedback_*.md` or `project_neural_router_status.md`.

---

## Done when

- [ ] `tests/test_figure_data_consistency.py`: all green
- [ ] `tests/test_smoke_scripts_layered.py`: all green
- [ ] `tests/test_run_qoe_max_events.py`: all green
- [ ] `latexmk -pdf -interaction=nonstopmode main.tex`: zero errors, zero unresolved figure refs
- [ ] Visual inspection of every figure PDF: caption matches plot
- [ ] Unit smoke (crossover + QoE): <60 s each, locally green
- [ ] Integration smoke (crossover + QoE): <5 min each on Mahti, F1 non-null
- [ ] No "deferred to future work" placeholder for figures we promised in body
- [ ] OPERATIONS_LOG appended for every session step
