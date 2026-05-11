"""TDD for R6: cost-model 17% miss stratified analysis.

Splits the misses (factor-of-two band violations) by whether the cluster
operates in the trivial regime (m_c <= b_max(c) — model collapses to
identity, miss is uninformative) or the non-trivial regime (m_c > b_max(c)
— cost model is genuinely under test, miss matters).

This test was written BEFORE analysis/cost_validation_stratified.py
existed (RED-phase TDD). The stratification justifies a §5.4
paragraph claim.
"""
from __future__ import annotations

import pandas as pd
import pytest


# Production constants from manuscript §3.5 (post-revision):
#   W=4096, t_inst=200, t_s=80, t_e=30, t_resp=500
PROD_CONSTANTS = dict(W=4096, t_inst=200, t_s=80, t_e=30, t_resp=500, derate=1.0)


def test_stratify_misses_synthetic_4_cells():
    """R6: split the misses (|I_pred/I_meas - 1| > 1) by stratum.

    Synthetic frame, 4 clusters total:
      - cluster 0: trivial regime, hit
      - cluster 1: non-trivial regime, hit
      - cluster 2: non-trivial regime, MISS (>2x)
      - cluster 3: trivial regime, MISS
    """
    from analysis.cost_validation_stratified import stratify_misses

    # One row per (config, k, seed); per_cluster_* are list-valued
    df = pd.DataFrame([
        {
            "config": "A1",
            "dataset": "D1",
            "seed": 42,
            "k": 4,
            "per_cluster_invocations": [1, 5, 12, 3],   # I_meas
            "per_cluster_events": [5, 50, 50, 5],       # m_c
            "per_cluster_active_subs": [4, 4, 4, 4],    # |S_c'|
        },
    ])
    summary = stratify_misses(df, **PROD_CONSTANTS)

    # b_max for |S_c'|=4: (4096 - 200 - 320 - 500) / 30 = 102 events/call
    # cluster 0: m=5 <= 102 → trivial; I_pred=ceil(5/102)=1, I_meas=1 → hit
    # cluster 1: m=50 <= 102 → trivial; I_pred=ceil(50/102)=1, I_meas=5 → MISS (4x)
    # cluster 2: m=50 <= 102 → trivial; I_pred=1, I_meas=12 → MISS (11x)
    # cluster 3: m=5 <= 102 → trivial; I_pred=1, I_meas=3 → MISS (2x)
    # All cells in this synthetic example are trivial regime (b_max=102, m<=50).
    assert summary["n_total"] == 4
    # All trivial — synthetic was set up to verify the regime split logic;
    # we now make a more demanding synthetic case below.

    # Stress: vary |S_c'| to push some clusters non-trivial.
    df2 = pd.DataFrame([
        {
            "config": "A1",
            "dataset": "D1",
            "seed": 42,
            "k": 4,
            # Cluster 0: m=5, |S|=4 → trivial; b_max=102. I_pred=1, I_meas=1 → hit
            # Cluster 1: m=200, |S|=40 → non-trivial; b_max=floor((4096-200-3200-500)/30)=floor(196/30)=6;
            #            I_pred=ceil(200/6)=34, I_meas=34 → hit
            # Cluster 2: m=200, |S|=40 → non-trivial; same b_max=6; I_pred=34, I_meas=80 → MISS (2.4x)
            # Cluster 3: m=5, |S|=4 → trivial; b_max=102. I_pred=1, I_meas=4 → MISS (3x)
            "per_cluster_invocations": [1, 34, 80, 4],
            "per_cluster_events": [5, 200, 200, 5],
            "per_cluster_active_subs": [4, 40, 40, 4],
        },
    ])
    summary2 = stratify_misses(df2, **PROD_CONSTANTS)
    assert summary2["n_total"] == 4
    assert summary2["n_miss"] == 2, f"expected 2 misses, got {summary2}"
    assert summary2["n_miss_nontrivial"] == 1, (
        f"non-trivial miss should count cluster 2 only: {summary2}"
    )
    assert summary2["n_miss_trivial"] == 1, (
        f"trivial miss should count cluster 3 only: {summary2}"
    )
    assert "miss_rate_nontrivial" in summary2
    assert "miss_rate_trivial" in summary2


def test_stratify_misses_handles_string_list_columns():
    """The CSV stores per_cluster_* as Python-literal strings (e.g.
    '[5, 50]'); stratify_misses must parse them transparently."""
    from analysis.cost_validation_stratified import stratify_misses
    df = pd.DataFrame([
        {
            "config": "A1",
            "dataset": "D1",
            "seed": 42,
            "k": 2,
            "per_cluster_invocations": "[1, 1]",
            "per_cluster_events": "[5, 50]",
            "per_cluster_active_subs": "[4, 4]",
        },
    ])
    summary = stratify_misses(df, **PROD_CONSTANTS)
    assert summary["n_total"] == 2  # 2 clusters total


def test_stratify_misses_factor_of_two_band():
    """Miss is defined as |I_pred / I_meas - 1| > 1 (i.e., outside the
    factor-of-two band). I_meas=0 cells are excluded (per fig:cost-validation
    caption: 'Cells with I_meas=0 (a single empty-batch artefact) are
    excluded from the ratio')."""
    from analysis.cost_validation_stratified import stratify_misses
    # All four cells trivial regime (small |S|, small m); only the I_meas
    # value matters for the band.
    df = pd.DataFrame([
        {
            "config": "A1", "dataset": "D1", "seed": 42, "k": 4,
            # I_pred = 1 for all (m_c=5, b_max=102)
            # cell 0: I_meas=1 → ratio 1.0 → hit
            # cell 1: I_meas=2 → ratio 0.5 → in band (boundary at 0.5..2)
            # cell 2: I_meas=3 → ratio 0.333 → MISS
            # cell 3: I_meas=0 → EXCLUDED (empty-batch artefact)
            "per_cluster_invocations": [1, 2, 3, 0],
            "per_cluster_events": [5, 5, 5, 5],
            "per_cluster_active_subs": [4, 4, 4, 4],
        },
    ])
    summary = stratify_misses(df, **PROD_CONSTANTS)
    assert summary["n_total"] == 3, "I_meas=0 cell must be excluded"
    assert summary["n_miss"] == 1, f"only cell 2 should miss: {summary}"
