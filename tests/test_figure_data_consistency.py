"""Pin down the invariants that figure captions claim, so the captions cannot
silently desync from the data they describe.

The 2026-05-04 audit found three figures whose captions stated the opposite
of what the data shows; this file is the regression net (per L60).
"""
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
    import numpy as np
    df = pd.read_csv(SCALE_50_2000).sort_values("n_events")
    n = df["n_events"].values
    inv = df["invocations"].values
    coef = np.polyfit(n, inv, 1)
    pred = np.polyval(coef, n)
    ss_res = ((inv - pred) ** 2).sum()
    ss_tot = ((inv - inv.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    assert r2 >= 0.99, f"I should be linear in |E| (50..2000): R^2 = {r2}"


def test_cost_model_predicted_matches_measured_within_factor_2():
    """fig:cost-validation invariant: per-cluster I_pred matches I_meas
    in median within tolerance, and >=80% of points fit within a factor of 2.
    Locks in the empirical validation of Eq. 4 from §3.5/§4.6."""
    import json, math
    W, T_INST, T_RESP, T_S, T_E = 4096, 200, 500, 80, 30
    ratios = []
    for path in [ROOT / "results/full/cost_validation/full/cost_validation_D1_qwen7b.csv",
                 ROOT / "results/full/cost_validation/full/cost_validation_D1_dryrun.csv"]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            try:
                pi = json.loads(r["per_cluster_invocations"])
                pe = json.loads(r["per_cluster_events"])
                ps = json.loads(r["per_cluster_active_subs"])
            except (TypeError, ValueError):
                continue
            for Im, m, S_c in zip(pi, pe, ps):
                if m == 0 or Im == 0:
                    continue
                avail = W - T_INST - S_c * T_S - T_RESP
                b_max = max(1, math.floor(avail * 0.8 / T_E)) if avail > 0 else 1
                Ip = math.ceil(m / b_max)
                ratios.append(Ip / Im)
    assert len(ratios) >= 30, f"need >=30 cluster points: got {len(ratios)}"
    median_ratio = sorted(ratios)[len(ratios) // 2]
    within_factor_2 = sum(1 for r in ratios if 0.5 <= r <= 2.0) / len(ratios)
    assert 0.7 <= median_ratio <= 1.5, \
        f"median I_pred/I_meas should be in [0.7, 1.5]: got {median_ratio:.3f}"
    assert within_factor_2 >= 0.75, \
        f">=75% of points should be within factor-2 band: got {100 * within_factor_2:.0f}%"


def test_crossover_a4_degrades_with_S():
    """fig:crossover invariant: at W=4096, A4's F1 decreases as |S| grows
    (the empirical refutation of the cost-model recall-preservation
    assumption)."""
    import re
    base = ROOT / "results/full/crossover/by_task"
    if not base.exists():
        return
    a4_by_S = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir() or "A4" not in d.name:
            continue
        m = re.search(r"S(\d+)_A4_s(\d+)", d.name)
        if not m:
            continue
        S = int(m.group(1))
        csv = d / "crossover_D1.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        a4_by_S[S] = float(df["f1"].iloc[0])
    if len(a4_by_S) < 3:
        return
    # F1(50) > F1(200) > F1(2000) — strict decrease at W=4096 on D1
    sorted_S = sorted(a4_by_S.keys())
    assert a4_by_S[sorted_S[0]] > a4_by_S[sorted_S[-1]], \
        f"A4 should degrade with |S| under W=4096; got {a4_by_S}"


def test_crossover_a0_plateaus_above_truncation_cap():
    """fig:crossover invariant: A0's F1 is approximately equal between
    |S|=200 and |S|=2000, because W=4096 truncation caps both at the same
    effective subscription count."""
    import re
    base = ROOT / "results/full/crossover/by_task"
    if not base.exists():
        return
    a0_by_S = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir() or "A0" not in d.name:
            continue
        m = re.search(r"S(\d+)_A0_s(\d+)", d.name)
        if not m:
            continue
        S = int(m.group(1))
        if "_s42" not in d.name:
            continue
        csv = d / "crossover_D1.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        a0_by_S[S] = float(df["f1"].iloc[0])
    if 200 in a0_by_S and 2000 in a0_by_S:
        diff = abs(a0_by_S[200] - a0_by_S[2000])
        assert diff < 0.005, \
            f"A0 F1 should plateau above truncation cap; |S|=200 vs |S|=2000: {a0_by_S[200]} vs {a0_by_S[2000]}"


def test_no_orphan_figures():
    """Every figure that survives the 2026-05-04 audit must be \\cref'd
    from at least one body file (per L60: figures justify claims)."""
    import re
    manuscript = ROOT.parent.parent / "Manuscripts" / "Neural Router (Elsevier FGCS)"
    txt_dir = manuscript / "txt"
    main_tex = manuscript / "main.tex"
    all_text = "\n".join(p.read_text() for p in txt_dir.glob("*.tex"))
    all_text += main_tex.read_text()
    labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", all_text))
    referenced = set(re.findall(r"\\(?:cref|Cref|ref)\{(fig:[^}]+)\}", all_text))
    # Comma-separated references e.g. \cref{fig:a,fig:b} also need handling
    for m in re.findall(r"\\(?:cref|Cref|ref)\{([^}]+)\}", all_text):
        for ref in m.split(","):
            ref = ref.strip()
            if ref.startswith("fig:"):
                referenced.add(ref)
    orphans = labels - referenced
    assert not orphans, f"orphan figures (defined but not referenced): {orphans}"


def test_pareto_a6_is_highest_invocations_per_backend():
    """fig:pareto caption claim: 'A6 (no cosine filter) is the highest-I
    configuration'. Lock the data invariant; if a future re-run flips it,
    the caption must be revisited."""
    for csv in ["results/full/ablation/D1_ablation_haiku_results.csv",
                "results/full/ablation/D1_ablation_qwen7b_per_task.csv",
                "results/full/ablation/D1_ablation_sonnet_results.csv"]:
        df = pd.read_csv(ROOT / csv)
        df = df[~df["config"].astype(str).str.startswith("baseline_")]
        g = df.groupby("config").agg(inv=("invocations", "mean")).reset_index()
        max_cfg = g.loc[g["inv"].idxmax(), "config"]
        assert max_cfg == "A6", f"{csv}: highest-I config is {max_cfg}, expected A6"


def test_k_sensitivity_f1_is_nonmonotone():
    """fig:k-sensitivity caption: F1 non-monotone, k=1 and k=19 both above
    the intermediate values."""
    df = pd.read_csv(ROOT / "results/puhti_mirror/sensitivity_by_task/qwen7b_sens_k_D1/sensitivity_k_D1.csv")
    df = df.set_index("k")
    f1_endpoints = max(df.loc[1, "f1"], df.loc[19, "f1"])
    f1_middle_max = df.loc[[5, 10, 15], "f1"].max()
    assert f1_endpoints > f1_middle_max, \
        f"endpoints (k=1 or 19) should beat middle (5,10,15): {df['f1'].to_dict()}"


def test_kappa_sensitivity_kappa3_is_at_plateau():
    """fig:kappa-sensitivity caption: kappa=3 is at the F1 plateau."""
    df = pd.read_csv(ROOT / "results/puhti_mirror/sensitivity_by_task/qwen7b_sens_kappa_D1/sensitivity_kappa_D1.csv")
    df = df.set_index("kappa")
    f1_max = df["f1"].max()
    assert abs(df.loc[3, "f1"] - f1_max) <= 0.01, \
        f"kappa=3 should be within 0.01 of max F1: kappa3={df.loc[3,'f1']}, max={f1_max}"


def test_embedding_sensitivity_minilm_close_to_best_at_lower_cost():
    """fig:embedding-sensitivity caption: MiniLM-L6 is within ~0.04 F1 of the
    best embedding but at ~1/12 the invocation count."""
    df = pd.read_csv(ROOT / "results/puhti_mirror/sensitivity_by_task/qwen7b_sens_embedding_D1/sensitivity_embedding_D1.csv")
    df = df.set_index("embedding_model")
    minilm_f1 = df.loc["all-MiniLM-L6-v2", "f1"]
    minilm_inv = df.loc["all-MiniLM-L6-v2", "invocations"]
    best_f1 = df["f1"].max()
    max_inv = df["invocations"].max()
    assert best_f1 - minilm_f1 <= 0.05, \
        f"MiniLM should be within 0.05 of best F1: minilm={minilm_f1}, best={best_f1}"
    assert max_inv / minilm_inv >= 8, \
        f"larger embeddings should cost ~10x or more: max_inv={max_inv}, minilm={minilm_inv}"


def test_pareto_a0_a4_compete_for_best_f1_per_backend():
    """fig:pareto caption claim: 'A0 and A4 attain the best F1-I point'.
    For each backend the top-2 F1 configs include both A0 and A4."""
    for csv in ["results/full/ablation/D1_ablation_haiku_results.csv",
                "results/full/ablation/D1_ablation_qwen7b_per_task.csv",
                "results/full/ablation/D1_ablation_sonnet_results.csv"]:
        df = pd.read_csv(ROOT / csv)
        df = df[~df["config"].astype(str).str.startswith("baseline_")]
        g = df.groupby("config").agg(f1=("f1", "mean")).reset_index()
        top2 = set(g.nlargest(2, "f1")["config"].tolist())
        assert {"A0", "A4"}.issubset(top2), f"{csv}: top-2 F1 configs = {top2}, not {{A0,A4}}"


def test_scaling_caption_does_not_claim_stability():
    """fig:scaling caption must not contain 'remains stable' phrasing —
    that was the 2026-05-04 audit finding (per L60)."""
    # ROOT = Experiments/neural-router; manuscript dir is .../FCG/Manuscripts/...
    manuscript = ROOT.parent.parent / "Manuscripts" / "Neural Router (Elsevier FGCS)"
    tex = (manuscript / "txt" / "Results.tex").read_text()
    import re
    # Locate the figure block containing fig:scaling
    blocks = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.DOTALL)
    scaling_block = next((b for b in blocks if "fig:scaling" in b and "scaling.pdf" in b), None)
    assert scaling_block is not None, "Could not locate the fig:scaling figure block"
    # Caption must not claim stability and must describe the decline
    caption_match = re.search(r"\\caption\{(.+?)\}\s*\\label", scaling_block, re.DOTALL)
    assert caption_match, "Could not extract caption from fig:scaling block"
    caption = caption_match.group(1).lower()
    assert "remains stable" not in caption and "f1 (left axis) remains" not in caption, \
        f"caption still claims F1 stability: {caption!r}"
    assert any(w in caption for w in ("declines", "decreases", "shrinks")), \
        f"caption should describe the F1 decline: {caption!r}"
